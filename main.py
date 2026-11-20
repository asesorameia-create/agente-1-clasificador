import os
import io
import gc
import re
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse
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
    title="Agente 1: El Clasificador Inteligente de Fotos",
    description="Sistema autónomo de detección biométrica por rostros y dorsales deportivos",
    version="1.2.0"
)

# Variables de entorno para Cloudflare R2
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "agente-1-clasificador-fotos")

# Nómina de jugadores en memoria (Dorsal -> Nombre del Jugador)
roster_config = {
    "1": "Portero_Trujillo",
    "10": "Capitan_10"
}

def get_r2_client():
    """Retorna un cliente boto3 configurado para Cloudflare R2."""
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
    "folders_created": {},
    "logs": []
}

def log_event(message: str):
    logger.info(message)
    processing_status["logs"].append(message)
    if len(processing_status["logs"]) > 100:
        processing_status["logs"].pop(0)

def detect_player_and_dorsal(img_bytes: bytes, current_roster: Dict[str, str]) -> Dict[str, Any]:
    """Detecta rostro, dorsal numérico y texto deportivo en memoria de forma ultra-ligera."""
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
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

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

        # Zona central para espaldas o dorsales prominentes
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
            "player_name": player_name,
            "sharpness": round(float(sharpness), 2)
        }
    except Exception as e:
        logger.error(f"Error en detección: {e}")
        return {"has_faces": False, "dorsal": None, "player_name": None, "sharpness": 0}

def process_photos_task(custom_roster: Dict[str, str] = None):
    """Procesamiento con clasificación por carpetas nombradas (Dorsal + Nombre)."""
    global processing_status, roster_config
    if custom_roster:
        roster_config.update(custom_roster)

    processing_status["is_processing"] = True
    processing_status["logs"] = []
    log_event("Iniciando escaneo del baúl de Cloudflare R2...")

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
        log_event(f"Se encontraron {len(pending_photos)} fotos originales para clasificar.")

        if not pending_photos:
            log_event("No hay fotos pendientes de clasificar.")
            processing_status["is_processing"] = False
            return

        folders_map = {}
        # Historial de jugadores detectados en la sesión para asignar fotos de acción frontal
        last_identified_folder = None

        for idx, key in enumerate(pending_photos, 1):
            filename = os.path.basename(key)
            log_event(f"[{idx}/{len(pending_photos)}] Analizando: {filename}...")
            
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            img_bytes = obj['Body'].read()

            features = detect_player_and_dorsal(img_bytes, roster_config)
            del img_bytes
            gc.collect()

            # Lógica de carpetas personalizadas por dorsal y nombre:
            if features["dorsal"]:
                d_num = int(features["dorsal"])
                name_tag = features["player_name"] or roster_config.get(str(d_num), f"Jugador_{d_num}")
                folder_name = f"organizadas/Dorsal_{d_num:02d}_{name_tag}"
                last_identified_folder = folder_name
            elif features["has_faces"]:
                # Si tiene rostro pero el dorsal no es visible de frente, agrupar con el jugador en juego
                folder_name = last_identified_folder or f"organizadas/Dorsal_01_{roster_config.get('1', 'Portero_Trujillo')}"
            else:
                folder_name = "organizadas/00_Fotos_Generales_Equipo"

            destination_key = f"{folder_name}/{filename}"

            # Copiar en Cloudflare R2 directamente
            s3.copy_object(
                Bucket=R2_BUCKET_NAME,
                CopySource={'Bucket': R2_BUCKET_NAME, 'Key': key},
                Key=destination_key
            )

            folders_map.setdefault(folder_name, []).append(filename)
            processing_status["processed_photos"] += 1
            log_event(f"-> Foto '{filename}' guardada en carpeta: {folder_name}")

        processing_status["folders_created"] = {k: len(v) for k, v in folders_map.items()}
        log_event("¡Clasificación completada al 100%! Todas las carpetas por jugador están listas.")

    except Exception as e:
        log_event(f"Error durante el procesamiento: {str(e)}")
    finally:
        processing_status["is_processing"] = False
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok", "service": "Agente 1 - Clasificador con Dorsales y Nombres"}

@app.get("/api/estado")
def get_status():
    return processing_status

@app.post("/api/procesar")
def trigger_process(background_tasks: BackgroundTasks, nomina: str = Form(default="")):
    if processing_status["is_processing"]:
        return JSONResponse({"status": "busy", "message": "Ya hay un proceso en marcha."})
    
    parsed_roster = {}
    if nomina:
        for line in nomina.splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                parsed_roster[k.strip()] = v.strip().replace(" ", "_")
            elif ":" in line:
                k, v = line.split(":", 1)
                parsed_roster[k.strip()] = v.strip().replace(" ", "_")

    background_tasks.add_task(process_photos_task, parsed_roster)
    return {"status": "started", "message": "Proceso de clasificación iniciado con éxito."}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    has_keys = bool(R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_ENDPOINT_URL)
    status_badge = '<span style="color:#10b981;font-weight:bold;">Conectado a Cloudflare R2</span>' if has_keys else '<span style="color:#ef4444;font-weight:bold;">Pendiente configurar Llaves R2</span>'
    
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agente 1: Clasificador de Fotos por Dorsal y Nombre</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            .header {{ background: #1e293b; padding: 24px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }}
            h1 {{ margin: 0 0 8px 0; font-size: 24px; color: #38bdf8; }}
            p.subtitle {{ margin: 0; color: #94a3b8; font-size: 14px; }}
            .card {{ background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px; }}
            .btn {{ background: #0284c7; color: white; border: none; padding: 14px 28px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 16px; transition: 0.2s; }}
            .btn:hover {{ background: #0369a1; }}
            .btn:disabled {{ background: #475569; cursor: not-allowed; }}
            .log-box {{ background: #090d16; border: 1px solid #1e293b; padding: 15px; border-radius: 8px; height: 260px; overflow-y: auto; font-family: monospace; font-size: 13px; color: #38bdf8; }}
            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; background: #334155; }}
            textarea {{ width: 100%; box-sizing: border-box; background: #090d16; color: #38bdf8; border: 1px solid #334155; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 13px; margin: 10px 0; }}
            .folder-badge {{ background: #0369a1; color: white; padding: 6px 12px; border-radius: 6px; font-size: 13px; margin: 4px; display: inline-block; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Agente 1: Clasificador por Dorsal y Nombre</h1>
                <p class="subtitle">Organización autónoma de fotos en carpetas individuales por jugador</p>
                <div style="margin-top: 15px;">
                    <span class="badge">Estado: {status_badge}</span>
                    <span class="badge">Baúl R2: {R2_BUCKET_NAME}</span>
                </div>
            </div>

            <div class="card">
                <h2>Panel de Control del Fotógrafo</h2>
                <p style="color: #cbd5e1; font-size: 14px;">
                    Escribe o pega aquí la nómina del partido (ejemplo: <code>1 = Portero_Trujillo</code> o <code>10 = Juan_Perez</code>). Si no pones nada, el robot le pondrá el dorsal detectado automáticamente.
                </p>
                <textarea id="nominaInput" rows="3" placeholder="1 = Portero_Trujillo&#10;10 = Juan_Perez">1 = Portero_Trujillo&#10;10 = Delantero</textarea>
                <br>
                <button id="btnProcesar" class="btn" onclick="iniciarProcesamiento()">🚀 Iniciar Clasificación en Carpetas por Jugador</button>
            </div>

            <div class="card">
                <h3>Carpetas Creadas por el Robot:</h3>
                <div id="foldersBox" style="min-height: 40px; margin-bottom: 15px;">
                    <span style="color: #64748b; font-size: 14px;">Las carpetas aparecerán aquí tan pronto termine el análisis...</span>
                </div>
                <h3>Registro en Vivo de la IA</h3>
                <div id="logBox" class="log-box">Esperando inicio...</div>
            </div>
        </div>

        <script>
            async function iniciarProcesamiento() {{
                const btn = document.getElementById("btnProcesar");
                btn.disabled = true;
                btn.innerText = "⏳ Creando carpetas por jugador...";
                const nomina = document.getElementById("nominaInput").value;
                const formData = new FormData();
                formData.append("nomina", nomina);

                try {{
                    const res = await fetch("/api/procesar", {{ method: "POST", body: formData }});
                    const data = await res.json();
                }} catch(e) {{
                    alert("Error iniciando proceso: " + e);
                }}
            }}

            async function actualizarEstado() {{
                try {{
                    const res = await fetch("/api/estado");
                    const data = await res.json();
                    const logBox = document.getElementById("logBox");
                    const foldersBox = document.getElementById("foldersBox");

                    if (data.logs && data.logs.length > 0) {{
                        logBox.innerHTML = data.logs.map(l => "<div>> " + l + "</div>").join("");
                        logBox.scrollTop = logBox.scrollHeight;
                    }}

                    if (data.folders_created && Object.keys(data.folders_created).length > 0) {{
                        foldersBox.innerHTML = Object.entries(data.folders_created)
                            .map(([f, count]) => `<span class="folder-badge">📁 ${{f}} (${{count}} fotos)</span>`)
                            .join(" ");
                    }}

                    if (!data.is_processing) {{
                        const btn = document.getElementById("btnProcesar");
                        btn.disabled = false;
                        btn.innerText = "🚀 Iniciar Clasificación en Carpetas por Jugador";
                    }}
                }} catch(e) {{}}
            }}

            setInterval(actualizarEstado, 1500);
        </script>
    </body>
    </html>
    """
    return html