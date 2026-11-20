import os
import io
import gc
import re
import zipfile
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import boto3
from botocore.config import Config
import numpy as np
import cv2

try:
    import pytesseract
except ImportError:
    pytesseract = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CastilloEnfoca")

app = FastAPI(
    title="CastilloEnfoca - Organizador Universal de Fotos",
    description="Agrupación biométrica universal de fotos por jugador y dorsal",
    version="2.2.0"
)

# Variables de entorno Cloudflare R2
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "agente-1-clasificador-fotos")

# Roster configurable (Dorsal -> Nombre)
roster_map = {
    "1": "Portero_Trujillo",
    "10": "Capitan_10"
}

def get_r2_client():
    if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY or not R2_ENDPOINT_URL:
        return None
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4")
    )

processing_status = {
    "is_processing": False,
    "last_run": None,
    "total_photos": 0,
    "processed_photos": 0,
    "current_action": "Listo para recibir fotos del partido",
    "folders_created": {},
    "logs": []
}

def log_event(message: str):
    logger.info(message)
    processing_status["logs"].append(message)
    processing_status["current_action"] = message
    if len(processing_status["logs"]) > 50:
        processing_status["logs"].pop(0)

def extract_universal_features(img_bytes: bytes) -> Dict[str, Any]:
    """Extrae rasgos faciales, histograma del uniforme y dorsal numérico de cualquier jugador."""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr
        if img is None:
            return {"has_face": False, "face_patch": None, "torso_hist": None, "dorsal": None}

        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        has_face = len(faces) > 0

        face_patch = None
        torso_hist = None
        detected_dorsal = None

        if has_face:
            # Tomar la cara principal
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_crop = gray[fy:fy+fh, fx:fx+fw]
            if face_crop.size > 0:
                face_patch = cv2.resize(face_crop, (64, 64))
                face_patch = cv2.equalizeHist(face_patch)

            # Zona del torso debajo del rostro
            ty1 = min(fy + fh, h - 10)
            ty2 = min(fy + int(fh * 3.0), h)
            tx1 = max(0, fx - int(fw * 0.5))
            tx2 = min(w, fx + int(fw * 1.5))
            
            torso_crop = hsv[ty1:ty2, tx1:tx2]
            if torso_crop.size > 100:
                hist = cv2.calcHist([torso_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                torso_hist = hist

        # Si no hay cara frontal (ej. foto de espalda), analizar el torso central
        else:
            cy1, cy2 = int(h * 0.15), int(h * 0.70)
            cx1, cx2 = int(w * 0.20), int(w * 0.80)
            center_crop = hsv[cy1:cy2, cx1:cx2]
            if center_crop.size > 100:
                hist = cv2.calcHist([center_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                torso_hist = hist

        # Detección de dorsal deportivo con OCR estricto
        if pytesseract:
            cy1, cy2 = int(h * 0.15), int(h * 0.75)
            cx1, cx2 = int(w * 0.20), int(w * 0.80)
            crop_ocr = gray[cy1:cy2, cx1:cx2]
            if crop_ocr.size > 0:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(crop_ocr)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                cfg = "--psm 11 -c tessedit_char_whitelist=0123456789"
                text = pytesseract.image_to_string(thresh, config=cfg).strip()
                # Filtrar dorsales válidos y aislados (ej. 1 al 99)
                nums = re.findall(r'\b([1-9][0-9]?)\b', text)
                if nums:
                    num_val = int(nums[0])
                    if 1 <= num_val <= 99:
                        detected_dorsal = str(num_val)

        del gray, hsv, img
        gc.collect()

        return {
            "has_face": has_face,
            "face_patch": face_patch,
            "torso_hist": torso_hist,
            "dorsal": detected_dorsal
        }
    except Exception as e:
        logger.error(f"Error extrayendo rasgos: {e}")
        return {"has_face": False, "face_patch": None, "torso_hist": None, "dorsal": None}

def cluster_and_assign_photos(photo_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Agrupa TODAS las fotos del mismo jugador en UNA SOLA CARPETA universalmente.
    Compara dorsales y similitud biométrica (rostro e histograma del uniforme).
    """
    clusters = []  # Lista de clusters: {"dorsal": ..., "face_patch": ..., "torso_hist": ..., "files": []}
    general_files = []

    for item in photo_items:
        key = item["key"]
        feat = item["features"]

        # Si no tiene rostro ni torso ni dorsal identificado, es foto general de cancha/equipo
        if not feat["has_face"] and feat["torso_hist"] is None and feat["dorsal"] is None:
            general_files.append(key)
            continue

        matched_cluster = None

        # Comparar con clusters existentes
        for c in clusters:
            # 1. Si comparten el mismo número de dorsal -> ES EL MISMO JUGADOR
            if feat["dorsal"] and c["dorsal"] and feat["dorsal"] == c["dorsal"]:
                matched_cluster = c
                break

            # 2. Si ambos tienen rostro, comparar correlación facial
            if feat["face_patch"] is not None and c["face_patch"] is not None:
                res = cv2.matchTemplate(feat["face_patch"], c["face_patch"], cv2.TM_CCOEFF_NORMED)
                if res[0][0] > 0.60:
                    matched_cluster = c
                    break

            # 3. Si comparten similitud de uniforme y uno es foto de espalda del mismo atleta
            if feat["torso_hist"] is not None and c["torso_hist"] is not None:
                sim = cv2.compareHist(feat["torso_hist"], c["torso_hist"], cv2.HISTCMP_CORREL)
                if sim > 0.85:
                    matched_cluster = c
                    break

        if matched_cluster:
            matched_cluster["files"].append(key)
            if not matched_cluster["dorsal"] and feat["dorsal"]:
                matched_cluster["dorsal"] = feat["dorsal"]
            if matched_cluster["face_patch"] is None and feat["face_patch"] is not None:
                matched_cluster["face_patch"] = feat["face_patch"]
        else:
            # Crear nuevo cluster
            clusters.append({
                "dorsal": feat["dorsal"],
                "face_patch": feat["face_patch"],
                "torso_hist": feat["torso_hist"],
                "files": [key]
            })

    # Nombrar las carpetas unificadas
    folders_result = {}
    player_counter = 1

    for c in clusters:
        # Si el cluster tiene un dorsal confirmado:
        if c["dorsal"]:
            d_num = int(c["dorsal"])
            name_suffix = roster_map.get(str(d_num), f"Jugador_{d_num}")
            folder_name = f"organizadas/Dorsal_{d_num:02d}_{name_suffix}"
        elif len(c["files"]) >= 2:
            # Si un jugador aparece en 2 o más fotos pero no se le vio el dorsal
            folder_name = f"organizadas/Jugador_Identificado_{player_counter:02d}"
            player_counter += 1
        else:
            # Foto aislada sin número confirmado -> se asigna a fotos generales
            general_files.extend(c["files"])
            continue

        folders_result[folder_name] = c["files"]

    if general_files:
        folders_result["organizadas/00_Fotos_Generales_Equipo"] = general_files

    return folders_result

def process_photos_task():
    """Ejecuta la clasificación universal y copia los archivos en sus carpetas unificadas."""
    global processing_status
    processing_status["is_processing"] = True
    processing_status["logs"] = []
    log_event("Analizando las fotos del partido...")

    s3 = get_r2_client()
    if not s3:
        log_event("ERROR: Cloudflare R2 no conectado.")
        processing_status["is_processing"] = False
        return

    try:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME)
        objects = response.get("Contents", [])
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        pending_photos = [
            obj["Key"] for obj in objects 
            if obj["Key"].lower().endswith(image_extensions) 
            and not obj["Key"].startswith("organizadas/")
        ]

        processing_status["total_photos"] = len(pending_photos)
        processing_status["processed_photos"] = 0

        if not pending_photos:
            log_event("No hay fotos nuevas para organizar.")
            processing_status["is_processing"] = False
            return

        # Paso 1: Analizar rasgos de todas las fotos
        photo_items = []
        for idx, key in enumerate(pending_photos, 1):
            filename = os.path.basename(key)
            log_event(f"[{idx}/{len(pending_photos)}] Analizando rasgos: {filename}...")
            
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            img_bytes = obj['Body'].read()
            feat = extract_universal_features(img_bytes)
            photo_items.append({"key": key, "features": feat})
            del img_bytes
            gc.collect()

        # Paso 2: Agrupación universal en una sola carpeta por jugador
        log_event("Agrupando jugadores por biometría y dorsales...")
        folders_map = cluster_and_assign_photos(photo_items)

        # Paso 3: Guardar en Cloudflare R2
        for folder_name, files in folders_map.items():
            for key in files:
                filename = os.path.basename(key)
                destination_key = f"{folder_name}/{filename}"
                s3.copy_object(
                    Bucket=R2_BUCKET_NAME,
                    CopySource={'Bucket': R2_BUCKET_NAME, 'Key': key},
                    Key=destination_key
                )
                processing_status["processed_photos"] += 1

        processing_status["folders_created"] = {k: len(v) for k, v in folders_map.items()}
        log_event("¡Listo! Todas las fotos de cada jugador quedaron unificadas en su carpeta.")

    except Exception as e:
        log_event(f"Error procesando fotos: {str(e)}")
    finally:
        processing_status["is_processing"] = False
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/estado")
def get_status():
    return processing_status

@app.get("/api/carpetas")
def get_folders():
    s3 = get_r2_client()
    if not s3:
        return {"folders": {}}
    try:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix="organizadas/")
        objects = response.get("Contents", [])
        result = {}
        for obj in objects:
            key = obj["Key"]
            parts = key.split("/")
            if len(parts) >= 3:
                folder_name = parts[1]
                filename = parts[2]
                result.setdefault(folder_name, []).append(filename)
        return {"folders": result}
    except Exception as e:
        return {"folders": {}}

@app.post("/api/procesar")
def trigger_process(background_tasks: BackgroundTasks):
    if processing_status["is_processing"]:
        return JSONResponse({"status": "busy", "message": "Ya hay un proceso en marcha."})
    background_tasks.add_task(process_photos_task)
    return {"status": "started", "message": "Clasificación universal iniciada."}

@app.post("/api/limpiar")
def clean_previous_session():
    """Borra las fotos organizadas viejas para dejar la pantalla completamente limpia."""
    s3 = get_r2_client()
    if not s3:
        return JSONResponse({"error": "No R2"}, status_code=500)
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix="organizadas/"):
            contents = page.get("Contents", [])
            if contents:
                delete_keys = [{'Key': obj['Key']} for obj in contents]
                s3.delete_objects(Bucket=R2_BUCKET_NAME, Delete={'Objects': delete_keys})

        processing_status["folders_created"] = {}
        processing_status["logs"] = []
        processing_status["current_action"] = "Listo para recibir fotos del partido"
        return {"status": "ok", "message": "Sesión limpiada con éxito."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/subir-y-procesar")
async def upload_and_process(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    s3 = get_r2_client()
    if not s3:
        return JSONResponse({"error": "R2 no conectado."}, status_code=500)

    clean_previous_session()

    count = 0
    for file in files:
        if file.filename:
            s3.upload_fileobj(file.file, R2_BUCKET_NAME, file.filename)
            count += 1

    background_tasks.add_task(process_photos_task)
    return {"status": "ok", "uploaded": count, "message": f"{count} fotos subidas."}

@app.get("/api/descargar-zip")
def download_all_zip():
    s3 = get_r2_client()
    if not s3:
        return JSONResponse({"error": "No R2"}, status_code=500)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix="organizadas/")
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                file_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
                zip_file.writestr(key.replace("organizadas/", ""), file_obj["Body"].read())

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=Fotos_Organizadas_CastilloEnfoca.zip"}
    )

@app.get("/", response_class=HTMLResponse)
def clean_dashboard():
    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CastilloEnfoca - Clasificador de Fotos</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b1329; color: #ffffff; padding: 16px; }
            .container { max-width: 550px; margin: 0 auto; }
            
            .header { text-align: center; margin-bottom: 20px; }
            .header h1 { font-size: 24px; color: #38bdf8; margin-bottom: 4px; }
            .header p { color: #94a3b8; font-size: 14px; }

            .card { background: #1e293b; border-radius: 16px; padding: 20px; margin-bottom: 16px; border: 1px solid #334155; }
            
            .upload-box { border: 2px dashed #38bdf8; border-radius: 12px; padding: 32px 16px; text-align: center; cursor: pointer; background: #0f172a; }
            .upload-box:active { background: #1e293b; }
            .upload-icon { font-size: 44px; margin-bottom: 8px; }
            .upload-text { font-size: 16px; font-weight: bold; color: #f8fafc; }
            .upload-hint { font-size: 13px; color: #94a3b8; margin-top: 4px; }
            
            input[type="file"] { display: none; }
            
            .btn-action { width: 100%; background: #0284c7; color: white; border: none; padding: 16px; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; margin-top: 14px; display: none; }
            .btn-action:active { background: #0369a1; }
            .btn-action:disabled { background: #475569; }

            .btn-download { width: 100%; background: #10b981; color: white; border: none; padding: 16px; border-radius: 12px; font-size: 17px; font-weight: bold; cursor: pointer; margin-top: 14px; text-align: center; text-decoration: none; display: none; }
            
            .btn-clean { width: 100%; background: transparent; color: #94a3b8; border: 1px solid #334155; padding: 10px; border-radius: 10px; font-size: 13px; cursor: pointer; margin-top: 10px; text-align: center; }
            .btn-clean:hover { color: #f8fafc; border-color: #64748b; }

            .status-box { background: #090d16; border-radius: 10px; padding: 12px; font-family: monospace; font-size: 13px; color: #38bdf8; text-align: center; }
            
            .folder-item { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
            .folder-name { font-weight: bold; font-size: 15px; color: #f8fafc; }
            .folder-count { background: #0284c7; padding: 4px 10px; border-radius: 20px; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📸 CastilloEnfoca</h1>
                <p>Organizador de Fotos del Partido</p>
            </div>

            <!-- PASO 1: SELECCIONAR FOTOS -->
            <div class="card">
                <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text" id="uploadLabel">Toca aquí para seleccionar las fotos</div>
                    <div class="upload-hint">Puedes seleccionar todas las fotos del partido juntas</div>
                </div>
                <input type="file" id="fileInput" multiple accept="image/*" onchange="fotosSeleccionadas(event)">
                <button id="btnSubir" class="btn-action" onclick="subirFotos()">🚀 Subir y Organizar con IA</button>
            </div>

            <!-- PASO 2: ESTADO (SOLO SE MUESTRA AL PROCESAR) -->
            <div class="card" id="cardEstado" style="display: none;">
                <div id="statusText" class="status-box">Listo para recibir fotos...</div>
            </div>

            <!-- PASO 3: CARPETAS ORGANIZADAS (SOLO CUANDO EXISTEN) -->
            <div class="card" id="cardCarpetas" style="display: none;">
                <h3 style="font-size: 15px; color: #38bdf8; margin-bottom: 12px;">📂 Tus Carpetas Listas:</h3>
                <div id="foldersList"></div>
                <a href="/api/descargar-zip" id="btnDescargar" class="btn-download">📦 Descargar Todas las Carpetas</a>
                <button class="btn-clean" onclick="limpiarTodo()">🧹 Limpiar para el siguiente partido</button>
            </div>
        </div>

        <script>
            let selectedFiles = [];

            function fotosSeleccionadas(e) {
                selectedFiles = Array.from(e.target.files);
                const label = document.getElementById("uploadLabel");
                const btn = document.getElementById("btnSubir");
                
                if (selectedFiles.length > 0) {
                    label.innerText = `✅ ${selectedFiles.length} fotos seleccionadas`;
                    btn.style.display = "block";
                    btn.innerText = `🚀 Subir ${selectedFiles.length} fotos y Organizar`;
                }
            }

            async function subirFotos() {
                if (selectedFiles.length === 0) return;
                const btn = document.getElementById("btnSubir");
                const cardEstado = document.getElementById("cardEstado");
                const status = document.getElementById("statusText");
                
                btn.disabled = true;
                btn.innerText = "⏳ Subiendo fotos...";
                cardEstado.style.display = "block";
                status.innerText = `Subiendo ${selectedFiles.length} fotos a la nube...`;

                const formData = new FormData();
                for (let i = 0; i < selectedFiles.length; i++) {
                    formData.append("files", selectedFiles[i]);
                }

                try {
                    const res = await fetch("/api/subir-y-procesar", {
                        method: "POST",
                        body: formData
                    });
                    const data = await res.json();
                    status.innerText = "Fotos subidas. El robot las está agrupando...";
                    selectedFiles = [];
                    document.getElementById("uploadLabel").innerText = "Toca aquí para seleccionar fotos";
                    btn.style.display = "none";
                    btn.disabled = false;
                } catch(err) {
                    status.innerText = "Error: " + err;
                    btn.disabled = false;
                }
            }

            async function limpiarTodo() {
                if (!confirm("¿Deseas limpiar la sesión para empezar un nuevo partido?")) return;
                try {
                    await fetch("/api/limpiar", { method: "POST" });
                    document.getElementById("cardCarpetas").style.display = "none";
                    document.getElementById("cardEstado").style.display = "none";
                    document.getElementById("foldersList").innerHTML = "";
                    document.getElementById("btnDescargar").style.display = "none";
                } catch(e) {}
            }

            async function refrescarCarpetas() {
                try {
                    const res = await fetch("/api/carpetas");
                    const data = await res.json();
                    const list = document.getElementById("foldersList");
                    const card = document.getElementById("cardCarpetas");
                    const btnDescargar = document.getElementById("btnDescargar");
                    
                    if (data.folders && Object.keys(data.folders).length > 0) {
                        card.style.display = "block";
                        btnDescargar.style.display = "block";
                        list.innerHTML = Object.entries(data.folders).map(([name, files]) => `
                            <div class="folder-item">
                                <div class="folder-name">📁 ${name}</div>
                                <div class="folder-count">${files.length} fotos</div>
                            </div>
                        `).join("");
                    } else {
                        card.style.display = "none";
                        btnDescargar.style.display = "none";
                    }
                } catch(e) {}
            }

            async function refrescarEstado() {
                try {
                    const res = await fetch("/api/estado");
                    const data = await res.json();
                    const status = document.getElementById("statusText");
                    const cardEstado = document.getElementById("cardEstado");
                    
                    if (data.is_processing) {
                        cardEstado.style.display = "block";
                        status.innerText = `⏳ [${data.processed_photos}/${data.total_photos}] ${data.current_action}`;
                    } else if (data.logs && data.logs.length > 0) {
                        status.innerText = "✅ " + data.logs[data.logs.length - 1];
                    }
                } catch(e) {}
            }

            setInterval(refrescarEstado, 2000);
            setInterval(refrescarCarpetas, 3000);
            refrescarCarpetas();
        </script>
    </body>
    </html>
    """
    return html
