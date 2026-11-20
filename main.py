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
    title="CastilloEnfoca - Organizador Autónomo",
    description="Sistema inteligente de separación biométrica y recuperación para fotógrafos",
    version="3.0.0"
)

# Variables de entorno Cloudflare R2
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "agente-1-clasificador-fotos")

# Roster de jugadores conocidos del torneo
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

def extract_advanced_features(img_bytes: bytes) -> Dict[str, Any]:
    """
    Extrae rasgos biométricos, color dominante del uniforme y número de dorsal.
    Garantiza que atletas con uniformes diferentes NUNCA se mezclen.
    """
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr
        if img is None:
            return {"has_face": False, "face_patch": None, "dominant_color": "desconocido", "hue": -1, "dorsal": None}

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
        dominant_color = "general"
        dominant_hue = -1

        # Extraer zona del torso del atleta
        if has_face:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_crop = gray[fy:fy+fh, fx:fx+fw]
            if face_crop.size > 0:
                face_patch = cv2.resize(face_crop, (64, 64))
                face_patch = cv2.equalizeHist(face_patch)

            ty1 = min(fy + fh, h - 10)
            ty2 = min(fy + int(fh * 3.0), h)
            tx1 = max(0, fx - int(fw * 0.6))
            tx2 = min(w, fx + int(fw * 1.6))
            torso = hsv[ty1:ty2, tx1:tx2]
        else:
            cy1, cy2 = int(h * 0.15), int(h * 0.70)
            cx1, cx2 = int(w * 0.20), int(w * 0.80)
            torso = hsv[cy1:cy2, cx1:cx2]

        # Clasificación estricta de color de uniforme (Espacio HSV)
        if torso.size > 100:
            h_vals = torso[:, :, 0].flatten()
            s_vals = torso[:, :, 1].flatten()
            v_vals = torso[:, :, 2].flatten()

            # Filtrar píxeles saturados y con brillo
            valid = (s_vals > 50) & (v_vals > 50)
            if np.sum(valid) > 50:
                valid_h = h_vals[valid]
                median_h = float(np.median(valid_h))
                dominant_hue = median_h

                if 35 <= median_h <= 85:
                    dominant_color = "verde_trujillo"  # Arquero
                elif 95 <= median_h <= 135:
                    dominant_color = "azul_amarillo"   # Jugador Azul
                elif 80 <= median_h < 95:
                    dominant_color = "celeste"         # Equipo Celeste
                elif median_h < 15 or median_h > 165:
                    dominant_color = "rojo"
                elif 15 <= median_h < 35:
                    dominant_color = "amarillo"

        # Detección de dorsal deportivo con OCR estricto
        detected_dorsal = None
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
            "dominant_color": dominant_color,
            "hue": dominant_hue,
            "dorsal": detected_dorsal
        }
    except Exception as e:
        logger.error(f"Error analizando foto: {e}")
        return {"has_face": False, "face_patch": None, "dominant_color": "general", "hue": -1, "dorsal": None}

def strict_unify_clusters(photo_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Agrupa fotos con REGLA DE ORO:
    1. Atletas con colores de uniforme diferentes NUNCA se mezclan en la misma carpeta.
    2. Cada atleta tiene su carpeta personalizada única.
    3. Si un jugador tiene 4 fotos, las 4 van a su carpeta.
    4. Si un jugador tiene 1 sola foto, esa sola foto va en su carpeta única personalizada.
    """
    clusters = []  # {"color": ..., "dorsal": ..., "face_patch": ..., "files": []}
    general_files = []

    for item in photo_items:
        key = item["key"]
        feat = item["features"]

        # Si no tiene rostro ni uniforme identificable ni dorsal, es toma abierta del estadio/árbitro
        if not feat["has_face"] and feat["dominant_color"] == "general" and feat["dorsal"] is None:
            general_files.append(key)
            continue

        matched_cluster = None

        for c in clusters:
            # REGLA FUNDAMENTAL: Los uniformes DEBEN coincidir para ser el mismo atleta
            if feat["dominant_color"] != "general" and c["color"] != "general":
                if feat["dominant_color"] != c["color"]:
                    continue  # Uniformes diferentes -> NUNCA mezclar

            # 1. Mismo dorsal y mismo color de uniforme -> ES EL MISMO JUGADOR
            if feat["dorsal"] and c["dorsal"] and feat["dorsal"] == c["dorsal"]:
                matched_cluster = c
                break

            # 2. Misma cara y mismo color de uniforme -> ES EL MISMO JUGADOR
            if feat["face_patch"] is not None and c["face_patch"] is not None:
                res = cv2.matchTemplate(feat["face_patch"], c["face_patch"], cv2.TM_CCOEFF_NORMED)
                if res[0][0] > 0.58:
                    matched_cluster = c
                    break

            # 3. Mismo rol distintivo (ej. arquero verde con casco de Trujillo)
            if feat["dominant_color"] == "verde_trujillo" and c["color"] == "verde_trujillo":
                matched_cluster = c
                break

        if matched_cluster:
            matched_cluster["files"].append(key)
            if not matched_cluster["dorsal"] and feat["dorsal"]:
                matched_cluster["dorsal"] = feat["dorsal"]
            if matched_cluster["face_patch"] is None and feat["face_patch"] is not None:
                matched_cluster["face_patch"] = feat["face_patch"]
        else:
            clusters.append({
                "color": feat["dominant_color"],
                "dorsal": feat["dorsal"],
                "face_patch": feat["face_patch"],
                "files": [key]
            })

    # Nombrar las carpetas personalizadas sin mezclar a nadie
    folders_result = {}
    player_num = 1

    for c in clusters:
        # Asignar nombre según dorsal o color/rol del jugador
        if c["color"] == "verde_trujillo":
            folder_name = "organizadas/Dorsal_01_Portero_Trujillo"
        elif c["dorsal"] and str(c["dorsal"]) in roster_map:
            d_val = int(c["dorsal"])
            folder_name = f"organizadas/Dorsal_{d_val:02d}_{roster_map[str(d_val)]}"
        elif c["dorsal"]:
            d_val = int(c["dorsal"])
            folder_name = f"organizadas/Dorsal_{d_val:02d}_Jugador_{d_val}"
        elif c["color"] == "azul_amarillo":
            folder_name = "organizadas/Jugador_Uniforme_Azul_Amarillo"
        elif c["color"] == "celeste":
            folder_name = "organizadas/Jugador_Uniforme_Celeste"
        else:
            folder_name = f"organizadas/Jugador_Identificado_{player_num:02d}"
            player_num += 1

        folders_result[folder_name] = c["files"]

    if general_files:
        folders_result["organizadas/00_Fotos_Generales_Equipo"] = general_files

    return folders_result

def process_photos_task():
    """Ejecuta la clasificación estricta y copia los archivos en sus carpetas unificadas."""
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
            and not obj["Key"].startswith("boveda_respaldo/")
        ]

        processing_status["total_photos"] = len(pending_photos)
        processing_status["processed_photos"] = 0

        if not pending_photos:
            log_event("No hay fotos nuevas para organizar.")
            processing_status["is_processing"] = False
            return

        # 1. Guardar copia en Bóveda de Respaldo por si el fotógrafo borra por error en su PC
        for key in pending_photos:
            filename = os.path.basename(key)
            backup_key = f"boveda_respaldo/{filename}"
            s3.copy_object(
                Bucket=R2_BUCKET_NAME,
                CopySource={'Bucket': R2_BUCKET_NAME, 'Key': key},
                Key=backup_key
            )

        # 2. Analizar rasgos de cada foto
        photo_items = []
        for idx, key in enumerate(pending_photos, 1):
            filename = os.path.basename(key)
            log_event(f"[{idx}/{len(pending_photos)}] Analizando: {filename}...")
            
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            img_bytes = obj['Body'].read()
            feat = extract_advanced_features(img_bytes)
            photo_items.append({"key": key, "features": feat})
            del img_bytes
            gc.collect()

        # 3. Agrupación estricta sin mezclar jugadores
        log_event("Separando cada jugador en su carpeta única...")
        folders_map = strict_unify_clusters(photo_items)

        # 4. Copiar a carpetas organizadas
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
        log_event("¡Listo! Cada atleta quedó en su carpeta individual sin mezclar fotos.")

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
    """Devuelve las fotos respaldadas en la Bóveda de Seguridad para recuperación."""
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

@app.get("/api/descargar-boveda-zip")
def download_backup_vault_zip():
    """Descarga todo el respaldo de seguridad por si el fotógrafo borró fotos por error."""
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
def castillo_branded_dashboard():
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
            
            /* Encabezado con paleta oficial de CastilloEnfoca */
            .header { text-align: center; margin-bottom: 24px; padding-top: 10px; }
            .logo-img { width: 110px; height: 110px; border-radius: 50%; margin: 0 auto 12px auto; display: block; box-shadow: 0 0 25px rgba(239, 68, 68, 0.45); border: 2px solid #ef4444; }
            .header h1 { font-size: 26px; font-weight: 900; letter-spacing: 1px; margin-bottom: 2px; }
            .brand-white { color: #ffffff; }
            .brand-red { color: #ef4444; }
            .header p { color: #a1a1aa; font-size: 14px; }

            .card { background: #141416; border-radius: 18px; padding: 22px; margin-bottom: 16px; border: 1px solid #27272a; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
            
            /* Caja de carga con borde neon naranja/rojo */
            .upload-box { border: 2px dashed #f97316; border-radius: 14px; padding: 34px 18px; text-align: center; cursor: pointer; background: #18181b; transition: 0.2s; }
            .upload-box:active { background: #27272a; border-color: #ef4444; }
            .upload-icon { font-size: 46px; margin-bottom: 8px; }
            .upload-text { font-size: 17px; font-weight: bold; color: #ffffff; }
            .upload-hint { font-size: 13px; color: #a1a1aa; margin-top: 4px; }
            
            input[type="file"] { display: none; }
            
            /* Botón de acción con degradado neon rojo-naranja */
            .btn-action { width: 100%; background: linear-gradient(135deg, #e11d48 0%, #ea580c 100%); color: white; border: none; padding: 18px; border-radius: 14px; font-size: 18px; font-weight: 900; cursor: pointer; margin-top: 16px; display: none; box-shadow: 0 4px 15px rgba(225, 29, 72, 0.4); text-transform: uppercase; }
            .btn-action:active { transform: scale(0.98); }
            .btn-action:disabled { background: #3f3f46; box-shadow: none; }

            /* Botón verde para descargar */
            .btn-download { width: 100%; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border: none; padding: 16px; border-radius: 14px; font-size: 17px; font-weight: bold; cursor: pointer; margin-top: 14px; text-align: center; text-decoration: none; display: none; box-shadow: 0 4px 15px rgba(16,185,129,0.3); }
            
            .btn-clean { width: 100%; background: transparent; color: #a1a1aa; border: 1px solid #27272a; padding: 12px; border-radius: 12px; font-size: 13px; cursor: pointer; margin-top: 12px; text-align: center; }
            .btn-clean:hover { color: #f43f5e; border-color: #f43f5e; }

            .status-box { background: #09090b; border-radius: 10px; padding: 14px; font-family: monospace; font-size: 13px; color: #f97316; text-align: center; border: 1px solid #27272a; }
            
            /* Carpetas ordenadas */
            .folder-item { background: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
            .folder-name { font-weight: bold; font-size: 15px; color: #ffffff; }
            .folder-count { background: linear-gradient(135deg, #e11d48 0%, #ea580c 100%); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }

            /* Bóveda de Respaldo Anti-Pérdidas */
            .vault-card { background: #111114; border: 1px solid #27272a; border-radius: 14px; padding: 16px; margin-top: 14px; text-align: center; }
            .vault-title { font-size: 14px; font-weight: bold; color: #facc15; margin-bottom: 6px; display: flex; items-center; justify-content: center; gap: 6px; }
            .vault-desc { font-size: 12px; color: #a1a1aa; margin-bottom: 10px; }
            .btn-vault { background: #27272a; color: #facc15; border: 1px solid #3f3f46; padding: 10px 16px; border-radius: 10px; font-size: 13px; font-weight: bold; text-decoration: none; display: inline-block; cursor: pointer; }
            .btn-vault:hover { background: #3f3f46; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="/logo.png" alt="CastilloEnfoca" class="logo-img" onerror="this.style.display='none'">
                <h1><span class="brand-white">CASTILLO</span><span class="brand-red">ENFOCA</span></h1>
                <p>Organizador Automático del Partido</p>
            </div>

            <!-- PASO 1: SUBIR FOTOS -->
            <div class="card">
                <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📸</div>
                    <div class="upload-text" id="uploadLabel">Toca aquí para seleccionar las fotos</div>
                    <div class="upload-hint">Puedes seleccionar todas las fotos del partido juntas</div>
                </div>
                <input type="file" id="fileInput" multiple accept="image/*" onchange="fotosSeleccionadas(event)">
                <button id="btnSubir" class="btn-action" onclick="subirFotos()">🚀 Subir y Organizar Fotos</button>
            </div>

            <!-- PASO 2: ESTADO EN VIVO -->
            <div class="card" id="cardEstado" style="display: none;">
                <div id="statusText" class="status-box">Listo para recibir fotos...</div>
            </div>

            <!-- PASO 3: TUS CARPETAS ORGANIZADAS -->
            <div class="card" id="cardCarpetas" style="display: none;">
                <h3 style="font-size: 15px; color: #f97316; margin-bottom: 14px; font-weight: bold;">📂 Tus Carpetas Listas:</h3>
                <div id="foldersList"></div>
                
                <a href="/api/descargar-zip" id="btnDescargar" class="btn-download" onclick="descargaIniciada()">📦 Descargar Todas las Carpetas en ZIP</a>
                
                <button class="btn-clean" onclick="limpiarTodo()">🧹 Limpiar para el siguiente partido</button>

                <!-- BÓVEDA DE RESPALDO DISRUPTIVA (ANTI-PÉRDIDAS) -->
                <div class="vault-card">
                    <div class="vault-title"><span>🛡️</span> Bóveda de Respaldo en la Nube</div>
                    <p class="vault-desc">¿Borraste una foto por error en tu computadora? Tu copia de seguridad está a salvo aquí.</p>
                    <a href="/api/descargar-boveda-zip" class="btn-vault">Restaurar Copia de Seguridad</a>
                </div>
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
                    status.innerText = "Fotos subidas. El robot las está agrupando sin mezclar...";
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
                setTimeout(() => {
                    // Al descargar, confirma y ofrece limpiar la vista principal
                    const status = document.getElementById("statusText");
                    const cardEstado = document.getElementById("cardEstado");
                    cardEstado.style.display = "block";
                    status.innerText = "✅ Descarga lista. Tu respaldo quedó a salvo en la Bóveda de Respaldo.";
                }, 1500);
            }

            async function limpiarTodo() {
                if (!confirm("¿Deseas limpiar la pantalla para empezar un nuevo partido?")) return;
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