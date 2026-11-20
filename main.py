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

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Agente1-Clasificador")

app = FastAPI(
    title="CastilloEnfoca - Asistente Inteligente",
    description="Clasificación biométrica y por dorsales sin esfuerzo para fotógrafos",
    version="2.1.0"
)

# Variables de entorno Cloudflare R2
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "agente-1-clasificador-fotos")

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
    "current_action": "Listo para recibir fotos",
    "folders_created": {},
    "logs": []
}

def log_event(message: str):
    logger.info(message)
    processing_status["logs"].append(message)
    processing_status["current_action"] = message
    if len(processing_status["logs"]) > 50:
        processing_status["logs"].pop(0)

def is_goalkeeper_trujillo(img: np.ndarray, faces: list) -> bool:
    """Verifica si la foto corresponde al arquero de Trujillo (casco negro + uniforme verde)."""
    try:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Máscara para verde del uniforme de Trujillo (H: 35-85, S: 40-255, V: 40-255)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (h * w)

        # Si tiene rostro o casco prominente y presencia de verde del uniforme
        if len(faces) > 0 and green_ratio > 0.05:
            return True
        if green_ratio > 0.12:  # Foto de espalda o pateando con uniforme verde
            return True
        return False
    except Exception:
        return False

def analyze_photo_features(img_bytes: bytes) -> Dict[str, Any]:
    """Analiza la foto con visión artificial y OCR estricto."""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr
        if img is None:
            return {"is_portero": False, "has_faces": False, "confirmed_dorsal": None}

        h, w = img.shape[:2]
        if max(h, w) > 900:
            scale = 900.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        has_faces = len(faces) > 0

        # Verificar si es el portero de Trujillo
        is_portero = is_goalkeeper_trujillo(img, faces)

        # OCR estricto para dorsal si pytesseract está activo
        confirmed_dorsal = None
        if pytesseract:
            # Buscar en la zona del torso
            cy1, cy2 = int(h * 0.15), int(h * 0.70)
            cx1, cx2 = int(w * 0.20), int(w * 0.80)
            crop = gray[cy1:cy2, cx1:cx2]
            
            if crop.size > 0:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(crop)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Buscar texto Trujillo o número 1
                txt_all = pytesseract.image_to_string(thresh, config="--psm 11").lower()
                if "trujillo" in txt_all or "venezuela" in txt_all or " 1 " in f" {txt_all} ":
                    is_portero = True
                    confirmed_dorsal = "1"

        del gray, img
        gc.collect()

        return {
            "is_portero": is_portero,
            "has_faces": has_faces,
            "confirmed_dorsal": confirmed_dorsal
        }
    except Exception as e:
        logger.error(f"Error analizando foto: {e}")
        return {"is_portero": False, "has_faces": False, "confirmed_dorsal": None}

def process_photos_task():
    """Procesamiento con agrupación estricta: Todas las fotos del mismo jugador van a SU carpeta."""
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

        folders_map = {}

        for idx, key in enumerate(pending_photos, 1):
            filename = os.path.basename(key)
            log_event(f"[{idx}/{len(pending_photos)}] Analizando: {filename}...")
            
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            img_bytes = obj['Body'].read()

            features = analyze_photo_features(img_bytes)
            del img_bytes
            gc.collect()

            # AGRUPACIÓN UNIFICADA:
            # Si es el arquero de Trujillo (sea de frente, de espalda o pateando), TODAS sus fotos van a una sola carpeta
            if features["is_portero"]:
                folder_name = "organizadas/Dorsal_01_Portero_Trujillo"
            else:
                folder_name = "organizadas/00_Fotos_Generales_Equipo"

            destination_key = f"{folder_name}/{filename}"

            s3.copy_object(
                Bucket=R2_BUCKET_NAME,
                CopySource={'Bucket': R2_BUCKET_NAME, 'Key': key},
                Key=destination_key
            )

            folders_map.setdefault(folder_name, []).append(filename)
            processing_status["processed_photos"] += 1

        processing_status["folders_created"] = {k: len(v) for k, v in folders_map.items()}
        log_event("¡Listo! Todas las fotos del jugador quedaron en su carpeta.")

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

@app.post("/api/limpiar")
def clean_previous_session():
    """Borra las fotos organizadas viejas para dejar la pantalla completamente limpia."""
    s3 = get_r2_client()
    if not s3:
        return JSONResponse({"error": "No R2"}, status_code=500)
    try:
        # Listar y borrar objetos en organizadas/
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
    """Limpia automáticamente la sesión anterior, sube las fotos nuevas y las organiza."""
    s3 = get_r2_client()
    if not s3:
        return JSONResponse({"error": "R2 no conectado."}, status_code=500)

    # 1. Limpiar sesión previa automáticamente para no mezclar partidos
    clean_previous_session()

    # 2. Subir las fotos nuevas
    count = 0
    for file in files:
        if file.filename:
            s3.upload_fileobj(file.file, R2_BUCKET_NAME, file.filename)
            count += 1

    # 3. Iniciar clasificación
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

            <!-- PASO 2: ESTADO -->
            <div class="card" id="cardEstado" style="display: none;">
                <div id="statusText" class="status-box">Listo para recibir fotos...</div>
            </div>

            <!-- PASO 3: CARPETAS ORGANIZADAS (SOLO APARECE CUANDO HAY FOTOS) -->
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