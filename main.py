import os
import io
import gc
import re
import zipfile
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
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
    title="Agente 1: El Asistente Móvil del Fotógrafo",
    description="Sube fotos desordenadas desde el celular y las clasifica automáticamente en carpetas por dorsal y jugador",
    version="2.0.0"
)

# Variables de entorno para Cloudflare R2
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "agente-1-clasificador-fotos")

# Nómina de jugadores por defecto
roster_config = {
    "1": "Portero_Trujillo",
    "10": "Capitan_10"
}

def get_r2_client():
    """Retorna cliente boto3 para Cloudflare R2."""
    if not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY or not R2_ENDPOINT_URL:
        return None
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4")
    )

# Estado de procesamiento en memoria
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
    if len(processing_status["logs"]) > 100:
        processing_status["logs"].pop(0)

def detect_player_and_dorsal(img_bytes: bytes, current_roster: Dict[str, str]) -> Dict[str, Any]:
    """Detecta rostro y dorsal numérico en memoria con bajo consumo de RAM."""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr
        if img is None:
            return {"dorsal": None, "player_name": None, "has_faces": False}

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

        detected_number = None
        detected_text = ""
        crops = []

        if has_faces:
            for (fx, fy, fw, fh) in faces[:2]:
                torso_y = min(fy + fh, h - 10)
                torso_h = min(int(fh * 2.8), h - torso_y)
                torso_x = max(0, fx - int(fw * 0.7))
                torso_w = min(int(fw * 2.4), w - torso_x)
                if torso_h > 30 and torso_w > 30:
                    crops.append(gray[torso_y:torso_y+torso_h, torso_x:torso_x+torso_w])

        cy1, cy2 = int(h * 0.15), int(h * 0.75)
        cx1, cx2 = int(w * 0.15), int(w * 0.85)
        crops.append(gray[cy1:cy2, cx1:cx2])

        if pytesseract:
            for crop in crops:
                if crop.size == 0:
                    continue
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(crop)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh_inv = cv2.bitwise_not(thresh)

                for target_img in [enhanced, thresh, thresh_inv]:
                    cfg_digits = "--psm 11 -c tessedit_char_whitelist=0123456789"
                    txt_digits = pytesseract.image_to_string(target_img, config=cfg_digits).strip()
                    nums = re.findall(r'\b([1-9][0-9]?)\b', txt_digits)
                    if nums:
                        detected_number = nums[0]
                        break

                    cfg_text = "--psm 11"
                    txt_raw = pytesseract.image_to_string(target_img, config=cfg_text).strip()
                    if "trujillo" in txt_raw.lower():
                        detected_text = "Trujillo"

                if detected_number:
                    break

        del gray, img
        gc.collect()

        player_name = None
        if detected_number:
            player_name = current_roster.get(str(detected_number), detected_text or f"Jugador_{detected_number}")
        elif detected_text:
            player_name = detected_text

        return {
            "has_faces": has_faces,
            "dorsal": detected_number,
            "player_name": player_name
        }
    except Exception as e:
        logger.error(f"Error detectando: {e}")
        return {"has_faces": False, "dorsal": None, "player_name": None}

def process_photos_task(custom_roster: Dict[str, str] = None):
    """Clasifica todas las fotos pendientes en carpetas por dorsal y nombre."""
    global processing_status, roster_config
    if custom_roster:
        roster_config.update(custom_roster)

    processing_status["is_processing"] = True
    processing_status["logs"] = []
    log_event("Analizando las fotos del partido...")

    s3 = get_r2_client()
    if not s3:
        log_event("ERROR: Credenciales de Cloudflare R2 no configuradas.")
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
        log_event(f"Se encontraron {len(pending_photos)} fotos para organizar.")

        if not pending_photos:
            log_event("No hay fotos nuevas pendientes.")
            processing_status["is_processing"] = False
            return

        folders_map = {}
        last_identified_folder = None

        for idx, key in enumerate(pending_photos, 1):
            filename = os.path.basename(key)
            log_event(f"[{idx}/{len(pending_photos)}] Identificando: {filename}...")
            
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            img_bytes = obj['Body'].read()

            features = detect_player_and_dorsal(img_bytes, roster_config)
            del img_bytes
            gc.collect()

            if features["dorsal"]:
                d_num = int(features["dorsal"])
                name_tag = features["player_name"] or roster_config.get(str(d_num), f"Jugador_{d_num}")
                folder_name = f"organizadas/Dorsal_{d_num:02d}_{name_tag}"
                last_identified_folder = folder_name
            elif features["has_faces"]:
                folder_name = last_identified_folder or f"organizadas/Dorsal_01_{roster_config.get('1', 'Portero_Trujillo')}"
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
        log_event("¡Listo! Todas las carpetas por jugador están organizadas.")

    except Exception as e:
        log_event(f"Error procesando fotos: {str(e)}")
    finally:
        processing_status["is_processing"] = False
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok", "service": "Agente 1 - Asistente Fotografo Móvil"}

@app.get("/api/estado")
def get_status():
    return processing_status

@app.get("/api/carpetas")
def get_folders():
    """Devuelve las carpetas organizadas y la lista de fotos en cada una."""
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
        return {"error": str(e), "folders": {}}

@app.post("/api/subir-y-procesar")
async def upload_and_process(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Sube fotos desordenadas directamente desde el teléfono y arranca la IA."""
    s3 = get_r2_client()
    if not s3:
        return JSONResponse({"error": "Cloudflare R2 no conectado."}, status_code=500)

    count = 0
    for file in files:
        if file.filename:
            # Subir directamente al baúl en la raíz (desordenadas)
            s3.upload_fileobj(file.file, R2_BUCKET_NAME, file.filename)
            count += 1

    # Iniciar clasificación en segundo plano de inmediato
    background_tasks.add_task(process_photos_task)
    return {"status": "ok", "uploaded": count, "message": f"{count} fotos subidas. La IA ya las está organizando."}

@app.get("/api/descargar-zip")
def download_all_zip():
    """Empaqueta todas las carpetas organizadas en un archivo ZIP para descarga directa."""
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
def mobile_friendly_dashboard():
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
            .container { max-width: 600px; margin: 0 auto; }
            
            .header { text-align: center; margin-bottom: 24px; padding-top: 10px; }
            .header h1 { font-size: 24px; color: #38bdf8; margin-bottom: 6px; }
            .header p { color: #94a3b8; font-size: 14px; }

            .step-card { background: #1e293b; border-radius: 16px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
            .step-title { font-size: 16px; font-weight: bold; color: #38bdf8; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
            
            /* Botón grande para celular */
            .upload-box { border: 2px dashed #38bdf8; border-radius: 12px; padding: 30px 15px; text-align: center; cursor: pointer; background: #0f172a; transition: 0.2s; }
            .upload-box:active { background: #1e293b; }
            .upload-icon { font-size: 42px; margin-bottom: 8px; }
            .upload-text { font-size: 16px; font-weight: bold; color: #f8fafc; }
            .upload-hint { font-size: 13px; color: #94a3b8; margin-top: 4px; }
            
            input[type="file"] { display: none; }
            
            .btn-action { width: 100%; background: #0284c7; color: white; border: none; padding: 16px; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; margin-top: 15px; display: none; }
            .btn-action:active { background: #0369a1; }
            .btn-action:disabled { background: #475569; cursor: not-allowed; }

            .btn-download { width: 100%; background: #10b981; color: white; border: none; padding: 16px; border-radius: 12px; font-size: 17px; font-weight: bold; cursor: pointer; margin-top: 15px; text-align: center; text-decoration: none; display: block; box-shadow: 0 4px 12px rgba(16,185,129,0.3); }
            
            .status-box { background: #090d16; border-radius: 10px; padding: 14px; font-family: monospace; font-size: 13px; color: #38bdf8; min-height: 50px; }
            
            .folder-item { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 14px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
            .folder-name { font-weight: bold; font-size: 15px; color: #f8fafc; }
            .folder-count { background: #0369a1; padding: 4px 10px; border-radius: 20px; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📸 CastilloEnfoca</h1>
                <p>Tu Asistente Inteligente de Clasificación</p>
            </div>

            <!-- PASO 1: SUBIR FOTOS DESDE EL CELULAR -->
            <div class="step-card">
                <div class="step-title"><span>1️⃣</span> Toca para elegir tus fotos desordenadas</div>
                <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text" id="uploadLabel">Toca aquí para seleccionar las fotos</div>
                    <div class="upload-hint">Puedes seleccionar 10, 50 o 500 fotos juntas</div>
                </div>
                <input type="file" id="fileInput" multiple accept="image/*" onchange="fotosSeleccionadas(event)">
                
                <button id="btnSubir" class="btn-action" onclick="subirFotos()">🚀 Subir y Organizar con IA</button>
            </div>

            <!-- PASO 2: ESTADO EN VIVO -->
            <div class="step-card">
                <div class="step-title"><span>🤖</span> Estado del Asistente</div>
                <div id="statusText" class="status-box">Esperando fotos...</div>
            </div>

            <!-- PASO 3: TUS CARPETAS LISTAS -->
            <div class="step-card">
                <div class="step-title"><span>📂</span> Tus Carpetas por Jugador</div>
                <div id="foldersList">
                    <p style="color: #64748b; font-size: 14px; text-align: center; padding: 10px;">Cargando carpetas existentes...</p>
                </div>
                <a href="/api/descargar-zip" class="btn-download">📦 Descargar Todas las Carpetas en ZIP</a>
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
                const status = document.getElementById("statusText");
                
                btn.disabled = true;
                btn.innerText = "⏳ Subiendo fotos...";
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
                    status.innerText = "Fotos subidas. La Inteligencia Artificial las está clasificando...";
                    selectedFiles = [];
                    document.getElementById("uploadLabel").innerText = "Toca aquí para seleccionar más fotos";
                    btn.style.display = "none";
                    btn.disabled = false;
                } catch(err) {
                    status.innerText = "Error: " + err;
                    btn.disabled = false;
                }
            }

            async function refrescarCarpetas() {
                try {
                    const res = await fetch("/api/carpetas");
                    const data = await res.json();
                    const list = document.getElementById("foldersList");
                    
                    if (data.folders && Object.keys(data.folders).length > 0) {
                        list.innerHTML = Object.entries(data.folders).map(([name, files]) => `
                            <div class="folder-item">
                                <div class="folder-name">📁 ${name}</div>
                                <div class="folder-count">${files.length} fotos</div>
                            </div>
                        `).join("");
                    } else {
                        list.innerHTML = `<p style="color: #64748b; font-size: 14px; text-align: center;">No hay carpetas aún.</p>`;
                    }
                } catch(e) {}
            }

            async function refrescarEstado() {
                try {
                    const res = await fetch("/api/estado");
                    const data = await res.json();
                    const status = document.getElementById("statusText");
                    if (data.is_processing) {
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
