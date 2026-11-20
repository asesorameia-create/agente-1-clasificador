import os
import io
import re
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import boto3
from botocore.config import Config
from PIL import Image
import numpy as np
import cv2

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Agente1-Clasificador")

app = FastAPI(
    title="Agente 1: El Clasificador Inteligente de Fotos",
    description="Sistema de clasificación biométrica por rostro y dorsal para fotografía deportiva",
    version="1.0.0"
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

def detect_faces_and_features(img_bytes: bytes) -> Dict[str, Any]:
    """Detecta rostros y estima características visuales básicas."""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"faces_count": 0, "has_faces": False}

        # Detección facial rápida con clasificador en cascada
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        # Cálculo de nitidez básica (Varianza del Laplaciano)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        return {
            "faces_count": len(faces),
            "has_faces": len(faces) > 0,
            "sharpness": round(float(sharpness), 2),
            "dimensions": f"{img.shape[1]}x{img.shape[0]}"
        }
    except Exception as e:
        logger.error(f"Error procesando imagen con OpenCV: {e}")
        return {"faces_count": 0, "has_faces": False, "sharpness": 0}

def detect_jersey_number(img_bytes: bytes) -> str:
    """Intenta detectar números estampados en el área del torso/camiseta."""
    try:
        # Búsqueda simple basada en OCR o patrones numéricos si EasyOCR está activo
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            results = reader.readtext(img_bytes, allowlist='0123456789')
            for bbox, text, prob in results:
                text_clean = text.strip()
                if text_clean.isdigit() and 1 <= int(text_clean) <= 99 and prob > 0.4:
                    return f"Dorsal_{text_clean}"
        except Exception:
            pass
        return ""
    except Exception:
        return ""

def process_photos_task():
    """Tarea en segundo plano que procesa las fotos en Cloudflare R2."""
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
        # Listar objetos en el bucket
        response = s3.list_objects_v2(Bucket=R2_BUCKET_NAME)
        objects = response.get("Contents", [])
        
        # Filtrar fotos en la raíz o en carpeta inbox/
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        pending_photos = [
            obj["Key"] for obj in objects 
            if obj["Key"].lower().endswith(image_extensions) 
            and not obj["Key"].startswith("organizadas/")
        ]

        processing_status["total_photos"] = len(pending_photos)
        processing_status["processed_photos"] = 0
        log_event(f"Se encontraron {len(pending_photos)} fotos pendientes por clasificar.")

        if not pending_photos:
            log_event("No hay fotos pendientes en la bandeja de entrada.")
            processing_status["is_processing"] = False
            return

        players_map = {}

        for key in pending_photos:
            filename = os.path.basename(key)
            log_event(f"Analizando foto: {filename}...")
            
            # Descargar imagen en memoria
            obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=key)
            img_bytes = obj['Body'].read()

            features = detect_faces_and_features(img_bytes)
            jersey = detect_jersey_number(img_bytes)

            # Clasificación inteligente
            if jersey:
                folder_name = f"organizadas/Jugador_{jersey}"
            elif features["has_faces"]:
                folder_name = "organizadas/Jugadores_Por_Identificar"
            else:
                folder_name = "organizadas/00_Fotos_Generales_Equipo"

            destination_key = f"{folder_name}/{filename}"

            # Copiar foto a la carpeta correspondiente en R2
            s3.copy_object(
                Bucket=R2_BUCKET_NAME,
                CopySource={'Bucket': R2_BUCKET_NAME, 'Key': key},
                Key=destination_key
            )

            # Opcional: registrar en reporte
            players_map.setdefault(folder_name, []).append(filename)
            processing_status["processed_photos"] += 1
            log_event(f"-> Foto '{filename}' asignada a: {folder_name}")

        processing_status["players_detected"] = {k: len(v) for k, v in players_map.items()}
        log_event("¡Clasificación completada con éxito! Todas las fotos están en sus carpetas.")

    except Exception as e:
        log_event(f"Error durante el procesamiento: {str(e)}")
    finally:
        processing_status["is_processing"] = False

@app.get("/health")
def health():
    return {"status": "ok", "service": "Agente 1 - Clasificador"}

@app.get("/api/estado")
def get_status():
    return processing_status

@app.post("/api/procesar")
def trigger_process(background_tasks: BackgroundTasks):
    if processing_status["is_processing"]:
        return JSONResponse({"status": "busy", "message": "Ya hay un proceso de clasificación en marcha."})
    background_tasks.add_task(process_photos_task)
    return {"status": "started", "message": "Proceso de clasificación iniciado en segundo plano."}

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
            .btn {{ background: #0284c7; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 15px; transition: 0.2s; }}
            .btn:hover {{ background: #0369a1; }}
            .btn:disabled {{ background: #475569; cursor: not-allowed; }}
            .log-box {{ background: #090d16; border: 1px solid #1e293b; padding: 15px; border-radius: 8px; height: 220px; overflow-y: auto; font-family: monospace; font-size: 12px; color: #38bdf8; }}
            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; background: #334155; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Agente 1: El Clasificador Inteligente de Fotos</h1>
                <p class="subtitle">Sistema autónomo de detección biométrica por rostros y dorsales deportivos</p>
                <div style="margin-top: 15px;">
                    <span class="badge">Estado: {status_badge}</span>
                    <span class="badge">Baúl R2: {R2_BUCKET_NAME}</span>
                </div>
            </div>

            <div class="card">
                <h2>Panel de Control del Fotógrafo</h2>
                <p style="color: #cbd5e1; font-size: 14px;">
                    Sube tus fotos a tu baúl de Cloudflare R2 o usa este botón para iniciar la clasificación con Inteligencia Artificial.
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
                    alert(data.message);
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

            setInterval(actualizarEstado, 2000);
        </script>
    </body>
    </html>
    """
    return html
