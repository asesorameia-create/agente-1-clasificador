import os
import io
import gc
import re
import zipfile
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
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
    title="CastilloEnfoca - Organizador por Dorsal y Nombre",
    description="Clasificación estricta por número de dorsal y nombre del jugador",
    version="3.1.0"
)

# Variables de entorno Cloudflare R2
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "agente-1-clasificador-fotos")

# Nómina oficial del partido: ÚNICA DISTINCIÓN (Dorsal -> Nombre del Jugador)
roster_map = {
    "1": "Portero_Trujillo",
    "4": "Defensa_4",
    "7": "Extremo_7",
    "10": "Delantero_10"
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

def detect_dorsal_and_face(img_bytes: bytes) -> Dict[str, Any]:
    """Detecta estrictamente el número de dorsal y el rostro del atleta."""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr
        if img is None:
            return {"dorsal": None, "face_patch": None}

        h, w = img.shape[:2]
        if max(h, w) > 850:
            scale = 850.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        face_patch = None

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_crop = gray[fy:fy+fh, fx:fx+fw]
            if face_crop.size > 0:
                face_patch = cv2.resize(face_crop, (64, 64))
                face_patch = cv2.equalizeHist(face_patch)

        # Búsqueda estricta de dorsal numérico (pecho, espalda o shorts)
        detected_dorsal = None
        if pytesseract:
            # 1. Zona central (torso/espalda)
            cy1, cy2 = int(h * 0.15), int(h * 0.70)
            cx1, cx2 = int(w * 0.20), int(w * 0.80)
            crop_torso = gray[cy1:cy2, cx1:cx2]

            # 2. Zona de shorts (donde suele estar el número pequeño)
            sy1, sy2 = int(h * 0.45), int(h * 0.85)
            sx1, sx2 = int(w * 0.25), int(w * 0.75)
            crop_shorts = gray[sy1:sy2, sx1:sx2]

            for target in [crop_torso, crop_shorts]:
                if target.size == 0:
                    continue
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(target)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh_inv = cv2.bitwise_not(thresh)

                for t_img in [enhanced, thresh, thresh_inv]:
                    cfg = "--psm 11 -c tessedit_char_whitelist=0123456789"
                    txt = pytesseract.image_to_string(t_img, config=cfg).strip()
                    nums = re.findall(r'\b([1-9][0-9]?)\b', txt)
                    for n in nums:
                        if n in roster_map:  # Validar estrictamente contra los dorsales del partido
                            detected_dorsal = str(int(n))
                            break
                    if detected_dorsal:
                        break
                if detected_dorsal:
                    break

        del gray, img
        gc.collect()

        return {
            "dorsal": detected_dorsal,
            "face_patch": face_patch
        }
    except Exception as e:
        logger.error(f"Error detectando dorsal: {e}")
        return {"dorsal": None, "face_patch": None}

def classify_strictly_by_dorsal_and_name(photo_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Clasificación ESTRICTA únicamente por DORSAL y NOMBRE.
    Sin distinción de colores de uniforme ni etiquetas raras.
    """
    folders_result = {}
    general_files = []

    # Registrar el rostro de cada dorsal confirmado para juntar fotos de frente
    dorsal_faces = {}  # {dorsal: face_patch}

    # Pase 1: Identificar fotos con dorsal directo confirmado
    unassigned = []
    for item in photo_items:
        key = item["key"]
        dorsal = item["features"]["dorsal"]
        face = item["features"]["face_patch"]

        if dorsal and dorsal in roster_map:
            d_val = int(dorsal)
            folder_name = f"organizadas/Dorsal_{d_val:02d}_{roster_map[str(d_val)]}"
            folders_result.setdefault(folder_name, []).append(key)
            if face is not None and dorsal not in dorsal_faces:
                dorsal_faces[dorsal] = face
        else:
            unassigned.append(item)

    # Pase 2: Para fotos sin dorsal visible (ej. de frente o tapadas), comparar rostro con los dorsales confirmados
    for item in unassigned:
        key = item["key"]
        face = item["features"]["face_patch"]
        matched_dorsal = None

        if face is not None:
            best_score = 0.0
            for d, d_face in dorsal_faces.items():
                res = cv2.matchTemplate(face, d_face, cv2.TM_CCOEFF_NORMED)
                score = float(res[0][0])
                if score > 0.55 and score > best_score:
                    best_score = score
                    matched_dorsal = d

        if matched_dorsal:
            d_val = int(matched_dorsal)
            folder_name = f"organizadas/Dorsal_{d_val:02d}_{roster_map[str(d_val)]}"
            folders_result.setdefault(folder_name, []).append(key)
        else:
            # Foto sin dorsal identificado ni rostro de jugador confirmado -> Fotos Generales del Equipo
            general_files.append(key)

    if general_files:
        folders_result["organizadas/00_Fotos_Generales_Equipo"] = general_files

    return folders_result

def process_photos_task():
    """Ejecuta la clasificación estricta por dorsal y nombre garantizando que el 100% de las fotos se guardan."""
    global processing_status
    processing_status["is_processing"] = True
    processing_status["logs"] = []
    log_event("Analizando fotos del partido por dorsal y nombre...")

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
            and not obj["Key"].startswith("boveda_respaldo/")
        ]

        processing_status["total_photos"] = len(pending_photos)
        processing_status["processed_photos"] = 0

        if not pending_photos:
            log_event("No hay fotos nuevas para organizar.")
            processing_status["is_processing"] = False
            return

        # 1. Copia de seguridad en la Bóveda de Respaldo
        for key in pending_photos:
            filename = os.path.basename(key)
            s3.copy_object(
                Bucket=R2_BUCKET_NAME,
                CopySource={'Bucket': R2_BUCKET_NAME, 'Key': key},
                Key=f"boveda_respaldo/{filename}"
            )

        # 2. Detección estricta de dorsal y rostro
        photo_items = []
        for idx, key in enumerate(pending_photos, 1):
            filename = os.path.basename(key)
            log_event(f"[{idx}/{len(pending_photos)}] Leyendo dorsal: {filename}...")
            
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            img_bytes = obj['Body'].read()
            feat = detect_dorsal_and_face(img_bytes)
            photo_items.append({"key": key, "features": feat})
            del img_bytes
            gc.collect()

        # 3. Clasificación estricta por Dorsal y Nombre
        log_event("Asignando cada foto a su carpeta por dorsal y nombre...")
        folders_map = classify_strictly_by_dorsal_and_name(photo_items)

        # 4. Guardar en Cloudflare R2
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
        log_event("¡Listo! Todas las fotos quedaron organizadas por dorsal y nombre.")

    except Exception as e:
        log_event(f"Error procesando fotos: {str(e)}")
    finally:
        processing_status["is_processing"] = False
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/logo.png")
def get_logo():
    if os.path.exists("logo.png"):
        return FileResponse("logo.png", media_type="image/png")
    return JSONResponse({"error": "No logo"}, status_code=404)

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

@app.get("/api/boveda")
def get_backup_vault():
    s3 = get_r2_client()
    if not s3:
        return {"backup_photos": []}
    try:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix="boveda_respaldo/")
        objects = response.get("Contents", [])
        files = [os.path.basename(obj["Key"]) for obj in objects if not obj["Key"].endswith("/")]
        return {"backup_photos": files, "count": len(files)}
    except Exception as e:
        return {"backup_photos": [], "count": 0}

@app.post("/api/procesar")
def trigger_process(background_tasks: BackgroundTasks):
    if processing_status["is_processing"]:
        return JSONResponse({"status": "busy", "message": "Ya hay un proceso en marcha."})
    background_tasks.add_task(process_photos_task)
    return {"status": "started", "message": "Clasificación estricta por dorsal y nombre iniciada."}

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

@app.get("/api/descargar-boveda-zip")
def download_backup_vault_zip():
    s3 = get_r2_client()
    if not s3:
        return JSONResponse({"error": "No R2"}, status_code=500)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix="boveda_respaldo/")
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                file_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
                zip_file.writestr(key.replace("boveda_respaldo/", "Respaldo_Seguridad/"), file_obj["Body"].read())

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=Boveda_Respaldo_CastilloEnfoca.zip"}
    )

@app.get("/", response_class=HTMLResponse)
def clean_dashboard():
    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CastilloEnfoca - Organizador de Fotos</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #09090b; color: #ffffff; padding: 16px; }
            .container { max-width: 520px; margin: 0 auto; }
            
            .header { text-align: center; margin-bottom: 24px; padding-top: 10px; }
            .logo-img { width: 110px; height: 110px; border-radius: 50%; margin: 0 auto 12px auto; display: block; box-shadow: 0 0 25px rgba(239, 68, 68, 0.45); border: 2px solid #ef4444; }
            .header h1 { font-size: 26px; font-weight: 900; letter-spacing: 1px; margin-bottom: 2px; }
            .brand-white { color: #ffffff; }
            .brand-red { color: #ef4444; }
            .header p { color: #a1a1aa; font-size: 14px; }

            .card { background: #141416; border-radius: 18px; padding: 22px; margin-bottom: 16px; border: 1px solid #27272a; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
            
            .upload-box { border: 2px dashed #f97316; border-radius: 14px; padding: 34px 18px; text-align: center; cursor: pointer; background: #18181b; transition: 0.2s; }
            .upload-box:active { background: #27272a; border-color: #ef4444; }
            .upload-icon { font-size: 46px; margin-bottom: 8px; }
            .upload-text { font-size: 17px; font-weight: bold; color: #ffffff; }
            .upload-hint { font-size: 13px; color: #a1a1aa; margin-top: 4px; }
            
            input[type="file"] { display: none; }
            
            .btn-action { width: 100%; background: linear-gradient(135deg, #e11d48 0%, #ea580c 100%); color: white; border: none; padding: 18px; border-radius: 14px; font-size: 18px; font-weight: 900; cursor: pointer; margin-top: 16px; display: none; box-shadow: 0 4px 15px rgba(225, 29, 72, 0.4); text-transform: uppercase; }
            .btn-action:active { transform: scale(0.98); }
            .btn-action:disabled { background: #3f3f46; box-shadow: none; }

            .btn-download { width: 100%; background: linear-gradient(135deg, #ef4444 0%, #f97316 100%); color: white; border: none; padding: 18px; border-radius: 14px; font-size: 18px; font-weight: 900; cursor: pointer; margin-top: 14px; text-align: center; text-decoration: none; display: none; box-shadow: 0 4px 20px rgba(239, 68, 68, 0.45); text-transform: uppercase; letter-spacing: 0.5px; transition: transform 0.15s, box-shadow 0.15s; }
            .btn-download:active { transform: scale(0.98); }
            
            .btn-clean { width: 100%; background: transparent; color: #a1a1aa; border: 1px solid #27272a; padding: 12px; border-radius: 12px; font-size: 13px; cursor: pointer; margin-top: 12px; text-align: center; }
            .btn-clean:hover { color: #f43f5e; border-color: #f43f5e; }

            .status-box { background: #09090b; border-radius: 10px; padding: 14px; font-family: monospace; font-size: 13px; color: #f97316; text-align: center; border: 1px solid #27272a; }
            
            .folder-item { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
            .folder-name { font-weight: bold; font-size: 15px; color: #ffffff; }
            .folder-count { background: linear-gradient(135deg, #ef4444 0%, #f97316 100%); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }

            .vault-card { background: #111114; border: 1px solid #27272a; border-radius: 14px; padding: 18px; margin-top: 18px; text-align: center; }
            .vault-title { font-size: 15px; font-weight: bold; color: #facc15; margin-bottom: 6px; display: flex; align-items: center; justify-content: center; gap: 8px; }
            .vault-desc { font-size: 13px; color: #a1a1aa; margin-bottom: 12px; line-height: 1.4; }
            .btn-vault { background: #1c1917; color: #facc15; border: 1px solid #78350f; padding: 12px 20px; border-radius: 10px; font-size: 14px; font-weight: bold; text-decoration: none; display: inline-block; cursor: pointer; transition: 0.2s; }
            .btn-vault:hover { background: #292524; border-color: #facc15; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="/logo.png" alt="CastilloEnfoca" class="logo-img" onerror="this.style.display='none'">
                <h1><span class="brand-white">CASTILLO</span><span class="brand-red">ENFOCA</span></h1>
                <p>Organizador por Dorsal y Nombre</p>
            </div>

            <div class="card">
                <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📸</div>
                    <div class="upload-text" id="uploadLabel">Toca aquí para seleccionar las fotos</div>
                    <div class="upload-hint">Puedes seleccionar todas las fotos del partido juntas</div>
                </div>
                <input type="file" id="fileInput" multiple accept="image/*" onchange="fotosSeleccionadas(event)">
                <button id="btnSubir" class="btn-action" onclick="subirFotos()">🚀 Subir y Organizar Fotos</button>
            </div>

            <div class="card" id="cardEstado" style="display: none;">
                <div id="statusText" class="status-box">Listo para recibir fotos...</div>
            </div>

            <div class="card" id="cardCarpetas" style="display: none;">
                <h3 style="font-size: 15px; color: #f97316; margin-bottom: 14px; font-weight: bold;">📂 Tus Carpetas por Dorsal y Nombre:</h3>
                <div id="foldersList"></div>
                
                <a href="/api/descargar-zip" id="btnDescargar" class="btn-download" onclick="descargaIniciada()">📦 Descargar Todas las Carpetas en ZIP</a>
                
                <button class="btn-clean" onclick="limpiarTodo()">🧹 Limpiar pantalla para el próximo partido</button>
            </div>

            <div class="vault-card" id="cardBoveda" style="display: none;">
                <div class="vault-title"><span>🛡️</span> Bóveda de Respaldo en la Nube (Anti-Pérdidas)</div>
                <p class="vault-desc" id="vaultDesc">¿Borraste una foto por error en tu computadora? Tu copia de seguridad está a salvo en la nube.</p>
                <a href="/api/descargar-boveda-zip" class="btn-vault" id="btnRestaurar">Restaurar Copia de Seguridad</a>
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
                    status.innerText = "Fotos subidas. El robot las está agrupando por dorsal y nombre...";
                    selectedFiles = [];
                    document.getElementById("uploadLabel").innerText = "Toca aquí para seleccionar fotos";
                    btn.style.display = "none";
                    btn.disabled = false;
                } catch(err) {
                    status.innerText = "Error: " + err;
                    btn.disabled = false;
                }
            }

            function descargaIniciada() {
                const status = document.getElementById("statusText");
                const cardEstado = document.getElementById("cardEstado");
                cardEstado.style.display = "block";
                status.innerText = "⏳ Descarga iniciada. Limpiando pantalla automáticamente en 4 segundos...";

                setTimeout(async () => {
                    try {
                        await fetch("/api/limpiar", { method: "POST" });
                        document.getElementById("cardCarpetas").style.display = "none";
                        status.innerText = "✅ Descarga completa. Pantalla limpia para el próximo partido. (Tu respaldo sigue a salvo en la Bóveda abajo).";
                        refrescarBoveda();
                    } catch(e) {}
                }, 4000);
            }

            async function limpiarTodo() {
                if (!confirm("¿Deseas limpiar la pantalla para empezar un nuevo partido?")) return;
                try {
                    await fetch("/api/limpiar", { method: "POST" });
                    document.getElementById("cardCarpetas").style.display = "none";
                    document.getElementById("cardEstado").style.display = "none";
                    document.getElementById("foldersList").innerHTML = "";
                    document.getElementById("btnDescargar").style.display = "none";
                    refrescarBoveda();
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
                        list.innerHTML = Object.entries(data.folders).map(([name, files]) => {
                            const cant = files.length;
                            const label = cant === 1 ? "1 foto" : `${cant} fotos`;
                            return `
                                <div class="folder-item">
                                    <div class="folder-name">📁 ${name}</div>
                                    <div class="folder-count">${label}</div>
                                </div>
                            `;
                        }).join("");
                    } else {
                        card.style.display = "none";
                        btnDescargar.style.display = "none";
                    }
                } catch(e) {}
            }

            async function refrescarBoveda() {
                try {
                    const res = await fetch("/api/boveda");
                    const data = await res.json();
                    const cardBoveda = document.getElementById("cardBoveda");
                    const vaultDesc = document.getElementById("vaultDesc");
                    if (data.count && data.count > 0) {
                        cardBoveda.style.display = "block";
                        const textoFotos = data.count === 1 ? "1 foto protegida" : `${data.count} fotos protegidas`;
                        vaultDesc.innerText = `¿Borraste una foto por error en tu computadora? Tienes ${textoFotos} en la nube listas para recuperar.`;
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
            setInterval(refrescarBoveda, 4000);
            refrescarCarpetas();
            refrescarBoveda();
        </script>
    </body>
    </html>
    """
    return html