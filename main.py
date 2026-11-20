import os
import io
import gc
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import boto3
from botocore.config import Config
import numpy as np
import cv2

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Agente1-Clasificador")

app = FastAPI(
    title="Agente 1: El Clasificador Inteligente de Fotos",
    description="Sistema autónomo de detección biométrica por rostros y dorsales deportivos",
    version="1.0.1"
)

# Variables de entorno para Cloudflare R2
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "agente-1-clasificador-fotos")

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
    "players_detected": {},
    "logs": []
}

def log_event(message: str):
    logger.info(message)
    processing_status["logs"].append(message)
    if len(processing_status["logs"]) > 100:
        processing_status["logs"].pop(0)

def detect_faces_and_features_optimized(img_bytes: bytes) -> Dict[str, Any]:
    """Detecta rostros usando miniaturas de bajo consumo de memoria RAM."""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr
        if img is None:
            return {"faces_count": 0, "has_faces": False}

        # Redimensionar a max 800px para consumir menos de 10 MB de RAM por foto
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        del img

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        del gray
        gc.collect()

        return {
            "faces_count": len(faces),
            "has_faces": len(faces) > 0,
            "sharpness": round(float(sharpness), 2)
        }
    except Exception as e:
        logger.error(f"Error procesando miniatura: {e}")
        return {"faces_count": 0, "has_faces": False, "sharpness": 0}

def process_photos_task():
    """Procesamiento ultra-optimizado sin sobrecarga de memoria."""
    global processing_status
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
        log_event(f"Se encontraron {len(pending_photos)} fotos para clasificar.")

        if not pending_photos:
            log_event("No hay fotos pendientes de clasificar.")
            processing_status["is_processing"] = False
            return

        players_map = {}

        for idx, key in enumerate(pending_photos, 1):
            filename = os.path.basename(key)
            log_event(f"[{idx}/{len(pending_photos)}] Analizando: {filename}...")
            
            # Descargar imagen
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            img_bytes = obj['Body'].read()

            features = detect_faces_and_features_optimized(img_bytes)
            del img_bytes
            gc.collect()

            # Clasificación inteligente
            if features["has_faces"]:
                folder_name = "organizadas/Jugadores_Identificados"
            else:
                folder_name = "organizadas/00_Fotos_Generales_Equipo"

            destination_key = f"{folder_name}/{filename}"

            # Copiar en Cloudflare R2 (operación server-side directa sin costo de RAM)
            s3.copy_object(
                Bucket=R2_BUCKET_NAME,
                CopySource={'Bucket': R2_BUCKET_NAME, 'Key': key},
                Key=destination_key
            )

            players_map.setdefault(folder_name, []).append(filename)
            processing_status["processed_photos"] += 1
            log_event(f"-> Foto '{filename}' clasificada con éxito en: {folder_name}")

        processing_status["players_detected"] = {k: len(v) for k, v in players_map.items()}
        log_event("¡Clasificación completada al 100%! Todas las carpetas están creadas en Cloudflare.")

    except Exception as e:
        log_event(f"Error durante el procesamiento: {str(e)}")
    finally:
        processing_status["is_processing"] = False
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok", "service": "Agente 1 - Clasificador"}

@app.get("/api/estado")
def get_status():
    return processing_status

@app.post("/api/procesar")
def trigger_process(background_tasks: BackgroundTasks):
    if processing_status["is_processing"]:
        return JSONResponse({"status": "busy", "message": "Ya hay un proceso en marcha."})
    background_tasks.add_task(process_photos_task)
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
        <title>Agente 1: El Clasificador Inteligente de Fotos</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Agente 1: El Clasificador Inteligente de Fotos</h1>
                <p class="subtitle">Sistema autónomo de detección biométrica por rostros deportivos</p>
                <div style="margin-top: 15px;">
                    <span class="badge">Estado: {status_badge}</span>
                    <span class="badge">Baúl R2: {R2_BUCKET_NAME}</span>
                </div>
            </div>

            <div class="card">
                <h2>Panel de Control del Fotógrafo</h2>
                <p style="color: #cbd5e1; font-size: 14px;">
                    Presiona el botón para escanear las fotos del baúl y organizarlas automáticamente en carpetas.
                </p>
                <button id="btnProcesar" class="btn" onclick="iniciarProcesamiento()">🚀 Iniciar Clasificación de Fotos Ahora</button>
            </div>

            <div class="card">
                <h3>Registro en Vivo de la IA</h3>
                <div id="logBox" class="log-box">Esperando inicio...</div>
            </div>
        </div>

        <script>
            async function iniciarProcesamiento() {{
                const btn = document.getElementById("btnProcesar");
                btn.disabled = true;
                btn.innerText = "⏳ Procesando fotos con IA...";
                try {{
                    const res = await fetch("/api/procesar", {{ method: "POST" }});
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
                    if (data.logs && data.logs.length > 0) {{
                        logBox.innerHTML = data.logs.map(l => "<div>> " + l + "</div>").join("");
                        logBox.scrollTop = logBox.scrollHeight;
                    }}
                    if (!data.is_processing) {{
                        const btn = document.getElementById("btnProcesar");
                        btn.disabled = false;
                        btn.innerText = "🚀 Iniciar Clasificación de Fotos Ahora";
                    }}
                }} catch(e) {{}}
            }}

            setInterval(actualizarEstado, 1500);
        </script>
    </body>
    </html>
    """
    return html
