"""
CastilloEnfoca - Agente 1 Clasificador (v4).

El fotografo sube las fotos desordenadas de un partido y el agente crea UNA carpeta por jugador:
  - Huella biometrica facial (YuNet + SFace) + dorsal (PP-OCR DB + CRNN) + equipo (texto y color de camiseta).
  - Nada esta escrito a mano para un partido concreto: dorsales 1-99, cualquier equipo, cualquier archivo.
  - Las fotos se analizan MIENTRAS se suben; al terminar la subida solo falta crear las carpetas.
  - Pantalla de revision: una ficha por jugador; corregir numero/nombre/equipo con un toque
    (poner el mismo numero y equipo de otra ficha las une).
"""
import os
import io
import gc
import re
import json
import time
import queue
import zipfile
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, Response, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from starlette.background import BackgroundTask

import motor_biometrico as motor
import identidad
from almacen import crear_almacen

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CastilloEnfoca")

app = FastAPI(
    title="CastilloEnfoca - Organizador biometrico por jugador",
    description="Una carpeta unica por jugador: huella facial + dorsal + equipo",
    version="4.0.0",
)

ALMACEN = crear_almacen()
EXT_IMG = (".jpg", ".jpeg", ".png", ".webp")
PREF_ORG, PREF_BOVEDA, PREF_MINI, PREF_SESION = "organizadas/", "boveda_respaldo/", "miniaturas/", "sesion/"
GENERAL = identidad.GENERAL


# ------------------------------------------------------------------ hilos de analisis
def _cpus_disponibles() -> int:
    """CPUs reales del contenedor (Render limita por cgroup aunque os.cpu_count() muestre las del host)."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:            # cgroup v2
            q, p = f.read().split()
            if q != "max":
                return max(1, int(int(q) / int(p)))
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:  # cgroup v1
            q, p = int(fq.read()), int(fp.read())
            if q > 0:
                return max(1, int(q / p))
    except Exception:
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


CPUS = _cpus_disponibles()
HILOS = max(1, min(int(os.getenv("ANALISIS_HILOS", "0")) or min(4, CPUS), 8))
if HILOS > 1:
    cv2.setNumThreads(max(1, CPUS // HILOS))
EJECUTOR = ThreadPoolExecutor(max_workers=HILOS, thread_name_prefix="analisis")
MAX_EN_MEMORIA = 3 * HILOS     # fotos esperando analisis con sus bytes en RAM (el resto se relee del almacen)
logger.info("CPUs=%s hilos de analisis=%s almacen=%s", CPUS, HILOS, type(ALMACEN).__name__ if ALMACEN else None)

_lock = threading.RLock()
_tareas: Dict[str, Future] = {}           # key -> analisis en curso / listo
_en_memoria = threading.Semaphore(MAX_EN_MEMORIA)

processing_status: Dict[str, Any] = {
    "is_processing": False,
    "last_run": None,
    "total_photos": 0,
    "processed_photos": 0,
    "current_action": "Listo para recibir fotos del partido",
    "folders_created": {},
    "logs": [],
    "fase": "espera",          # espera | subiendo | analizando | carpetas | listo | error
    "analizadas": 0,
    "eta_seg": None,
    "resumen": None,
    "inicio": None,
    "duracion_seg": None,
}


def log_event(message: str):
    logger.info(message)
    processing_status["logs"].append(message)
    processing_status["current_action"] = message
    if len(processing_status["logs"]) > 50:
        processing_status["logs"].pop(0)


def _analizar(key: str, datos: Optional[bytes]) -> Dict[str, Any]:
    try:
        if datos is None:
            datos = ALMACEN.leer(key)
        a = motor.analizar_foto(datos, os.path.basename(key))
    except Exception as e:
        logger.exception("No se pudo analizar %s: %s", key, e)
        a = {"nombre": os.path.basename(key), "hora": None, "jugadores": [], "firma": None, "ok": False}
    finally:
        datos = None
        gc.collect()
    a["key"] = key
    for j in a.get("jugadores", []):
        j.pop("torso", None)
    with _lock:
        processing_status["analizadas"] = sum(1 for f in _tareas.values() if f.done()) + 1
    return a


def _encolar(key: str, datos: Optional[bytes]):
    """Lanza el analisis de una foto. Si hay muchas en cola, no guarda sus bytes (se releen luego)."""
    with _lock:
        if key in _tareas:
            return
        if datos is not None and _en_memoria.acquire(blocking=False):
            def tarea(k=key, d=datos):
                try:
                    return _analizar(k, d)
                finally:
                    _en_memoria.release()
            _tareas[key] = EJECUTOR.submit(tarea)
        else:
            _tareas[key] = EJECUTOR.submit(_analizar, key, None)


# Compatibilidad con los nombres del pipeline anterior ---------------------------------
def detect_dorsal_and_face(img_bytes: bytes) -> Dict[str, Any]:
    """Huella de cada jugador visible en la foto (rostro SFace, dorsal 1-99, color/texto de camiseta)."""
    return motor.analizar_foto(img_bytes)


def classify_dynamically_by_dorsal_and_biometrics(photo_items: List[Dict[str, Any]], nomina: str = "") -> Dict[str, List[str]]:
    """Agrupa todo el lote en carpetas unicas por jugador (dorsal + biometria facial + rafagas)."""
    return identidad.clasificar_lote(photo_items, nomina)["carpetas"]


# ------------------------------------------------------------------ utilidades de sesion
def _fotos_sesion() -> List[str]:
    return [k for k, _ in ALMACEN.listar("") if "/" not in k and k.lower().endswith(EXT_IMG)]


def _nombre_seguro(nombre: str) -> str:
    base = os.path.basename((nombre or "foto.jpg").replace("\\", "/"))
    base = re.sub(r"[^\w\-. ]", "_", base, flags=re.UNICODE).strip() or "foto.jpg"
    if not base.lower().endswith(EXT_IMG):
        base += ".jpg"
    return base[:120]


def _leer_sesion() -> Dict[str, Any]:
    try:
        return json.loads(ALMACEN.leer(PREF_SESION + "resultado.json").decode("utf-8"))
    except Exception:
        return {}


def _guardar_sesion(d: Dict[str, Any]):
    ALMACEN.subir(PREF_SESION + "resultado.json", json.dumps(d, ensure_ascii=False).encode("utf-8"), "application/json")


def _limpiar_sesion(borrar_fotos: bool = True):
    """Borra carpetas organizadas, miniaturas y (si se pide) las fotos crudas de la sesion.
    Las fotos crudas ya estan respaldadas en la boveda desde que se subieron."""
    with _lock:
        for f in _tareas.values():
            f.cancel()
        _tareas.clear()
    if ALMACEN:
        for p in (PREF_ORG, PREF_MINI, PREF_SESION):
            ALMACEN.borrar_prefijo(p)
        if borrar_fotos:
            ALMACEN.borrar(_fotos_sesion())
    processing_status.update({
        "folders_created": {}, "logs": [], "current_action": "Listo para recibir fotos del partido",
        "fase": "espera", "analizadas": 0, "total_photos": 0, "processed_photos": 0, "eta_seg": None,
        "resumen": None, "inicio": None, "duracion_seg": None,
    })


# ------------------------------------------------------------------ pipeline principal
def process_photos_task(nomina: str = ""):
    """Espera los analisis (que empezaron durante la subida), agrupa por jugador y escribe las carpetas."""
    if not ALMACEN:
        log_event("ERROR: almacenamiento (Cloudflare R2) no conectado.")
        processing_status["fase"] = "error"
        return
    processing_status["is_processing"] = True
    processing_status["fase"] = "analizando"
    processing_status["inicio"] = processing_status.get("inicio") or time.time()
    try:
        if not motor.modelos_disponibles():
            raise RuntimeError(f"Faltan los modelos biometricos en {motor.MODELS_DIR}")
        keys = sorted(_fotos_sesion())
        processing_status["total_photos"] = len(keys)
        processing_status["processed_photos"] = 0
        if not keys:
            log_event("No hay fotos para organizar.")
            processing_status["fase"] = "espera"
            return
        for k in keys:
            _encolar(k, None)
        log_event(f"Analizando rostros y dorsales de {len(keys)} fotos...")
        t0, base = time.time(), sum(1 for k in keys if _tareas[k].done())
        while True:
            hechas = sum(1 for k in keys if _tareas[k].done())
            processing_status["processed_photos"] = hechas
            processing_status["analizadas"] = hechas
            if hechas > base:
                vel = (time.time() - t0) / (hechas - base)
                processing_status["eta_seg"] = int(vel * (len(keys) - hechas)) + 5
            processing_status["current_action"] = f"Analizando rostros y dorsales: {hechas}/{len(keys)} fotos"
            if hechas >= len(keys):
                break
            time.sleep(0.5)
        analisis = []
        for k in keys:
            try:
                analisis.append(_tareas[k].result())
            except Exception as e:
                logger.error("Analisis fallido %s: %s", k, e)
                analisis.append({"key": k, "jugadores": [], "hora": None, "firma": None})

        processing_status["fase"] = "carpetas"
        log_event("Uniendo huellas biometricas y creando una carpeta por jugador...")
        res = identidad.clasificar_lote(analisis, nomina)

        ALMACEN.borrar_prefijo(PREF_ORG)
        ALMACEN.borrar_prefijo(PREF_MINI)
        copias = [(k, f"{PREF_ORG}{carpeta}/{os.path.basename(k)}") for carpeta, ks in res["carpetas"].items() for k in ks]
        processing_status["total_photos"] = len(copias)
        processing_status["processed_photos"] = 0

        def copiar(par):
            ALMACEN.copiar(*par)
            processing_status["processed_photos"] += 1

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(copiar, copias))
        for carpeta, jpg in res["miniaturas"].items():
            ALMACEN.subir(f"{PREF_MINI}{carpeta}.jpg", jpg)

        processing_status["folders_created"] = {k: len(v) for k, v in res["carpetas"].items()}
        processing_status["resumen"] = res["resumen"]
        processing_status["duracion_seg"] = int(time.time() - processing_status["inicio"])
        _guardar_sesion({"resumen": res["resumen"], "detalle": res["detalle"], "dudas": res["dudas"], "nomina": nomina,
                         "fotos": len(keys), "fecha": time.strftime("%Y-%m-%d %H:%M")})
        processing_status["fase"] = "listo"
        processing_status["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        r = res["resumen"]
        log_event(f"¡Listo! {r['jugadores']} jugadores identificados en {r['fotos']} fotos.")
    except Exception as e:
        logger.exception("Error procesando fotos")
        processing_status["fase"] = "error"
        log_event(f"Error procesando fotos: {e}")
    finally:
        processing_status["is_processing"] = False
        processing_status["eta_seg"] = None
        gc.collect()


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/diagnostico")
def diagnostico():
    return {"almacen": type(ALMACEN).__name__ if ALMACEN else None, "modelos": motor.modelos_disponibles(),
            "cpus": CPUS, "hilos_analisis": HILOS, "opencv": cv2.__version__, "version": app.version}


@app.get("/logo.png")
def get_logo():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if os.path.exists(ruta):
        return FileResponse(ruta, media_type="image/png")
    return JSONResponse({"error": "No logo"}, status_code=404)


@app.get("/api/estado")
def get_status():
    return processing_status


@app.get("/api/carpetas")
def get_folders():
    if not ALMACEN:
        return {"folders": {}}
    try:
        result: Dict[str, List[str]] = {}
        for key, _ in ALMACEN.listar(PREF_ORG):
            parts = key.split("/")
            if len(parts) >= 3:
                result.setdefault(parts[1], []).append(parts[2])
        ses = _leer_sesion()
        dudas = [d for d in ses.get("dudas", []) if d.get("estado") == "pendiente"]
        return {"folders": result, "detalle": ses.get("detalle", {}), "resumen": ses.get("resumen"), "dudas": dudas}
    except Exception as e:
        logger.error("carpetas: %s", e)
        return {"folders": {}}


@app.get("/api/miniatura/{carpeta}")
def get_thumbnail(carpeta: str):
    try:
        return Response(ALMACEN.leer(f"{PREF_MINI}{carpeta}.jpg"), media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})
    except Exception:
        return Response(status_code=404)


@app.get("/api/miniatura-duda/{duda_id}")
def get_doubt_thumbnail(duda_id: str):
    try:
        return Response(ALMACEN.leer(f"{PREF_MINI}_dudas/{re.sub(r'[^a-z0-9]', '', duda_id)}.jpg"), media_type="image/jpeg")
    except Exception:
        return Response(status_code=404)


@app.post("/api/confirmar")
async def confirmar_duda(request: Request):
    """El fotografo responde una duda: si=True mete la foto en la carpeta del jugador; si=False la descarta."""
    d = await request.json()
    did, si = str(d.get("id", "")), bool(d.get("si"))

    def aplicar():
        ses = _leer_sesion()
        duda = next((x for x in ses.get("dudas", []) if x["id"] == did), None)
        if not duda:
            return {"error": "Duda no encontrada"}
        if duda["estado"] != "pendiente":
            return {"ok": True, "ya": duda["estado"]}
        duda["estado"] = "si" if si else "no"
        carpeta = duda["carpeta"]
        if si:
            # si la carpeta cambio de nombre en la revision, buscar su nombre actual por dorsal/equipo
            fotos = [duda["foto"]] + ([duda["foto_ref"]] if duda.get("foto_ref") else [])
            for k in fotos:
                dst = f"{PREF_ORG}{carpeta}/{os.path.basename(k)}"
                if not ALMACEN.existe(dst):
                    ALMACEN.copiar(k, dst)
                # ya tiene carpeta de jugador: sale de Generales
                ALMACEN.borrar([f"{PREF_ORG}{GENERAL}/{os.path.basename(k)}"])
            det = ses.setdefault("detalle", {}).setdefault(carpeta, {"equipo": duda.get("equipo"), "dorsal": duda.get("dorsal")})
            det["fotos"] = len(ALMACEN.listar(f"{PREF_ORG}{carpeta}/"))
            # las demas dudas que usaban esta misma foto de referencia ya no necesitan volver a copiarla
        _guardar_sesion(ses)
        pend = sum(1 for x in ses["dudas"] if x["estado"] == "pendiente")
        return {"ok": True, "pendientes": pend}

    r = await run_in_threadpool(aplicar)
    return JSONResponse(r, status_code=404 if "error" in r else 200)


_cache_vistas: Dict[str, bytes] = {}


@app.get("/api/vista/{carpeta}/{archivo}")
def get_preview(carpeta: str, archivo: str):
    """Vista previa liviana (360 px) de una foto de una carpeta, para revisar sin descargar.
    carpeta='_original' muestra la foto subida (para las dudas, que aun no estan en ninguna carpeta)."""
    key = _nombre_seguro(archivo) if carpeta == "_original" else f"{PREF_ORG}{carpeta}/{archivo}"
    if key in _cache_vistas:
        return Response(_cache_vistas[key], media_type="image/jpeg")
    try:
        datos = ALMACEN.leer(key)
        img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        h, w = img.shape[:2]
        s = 360 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
        out = buf.tobytes()
        if len(_cache_vistas) > 400:
            _cache_vistas.clear()
        _cache_vistas[key] = out
        return Response(out, media_type="image/jpeg")
    except Exception:
        return Response(status_code=404)


def _mover_carpeta(origen: str, destino: str):
    """Mueve (o une) las fotos y la miniatura de una carpeta de jugador a otra."""
    if origen == destino:
        return
    existentes = {os.path.basename(k) for k, _ in ALMACEN.listar(f"{PREF_ORG}{destino}/")}
    keys = [k for k, _ in ALMACEN.listar(f"{PREF_ORG}{origen}/")]
    for k in keys:
        if os.path.basename(k) not in existentes:
            ALMACEN.copiar(k, f"{PREF_ORG}{destino}/{os.path.basename(k)}")
    ALMACEN.borrar(keys)
    mo, md = f"{PREF_MINI}{origen}.jpg", f"{PREF_MINI}{destino}.jpg"
    if ALMACEN.existe(mo):
        if not ALMACEN.existe(md):
            ALMACEN.copiar(mo, md)
        ALMACEN.borrar([mo])


@app.post("/api/corregir")
async def corregir_carpeta(request: Request):
    """Corrige numero/nombre/equipo de una ficha. Si el nombre resultante ya existe, las fichas se UNEN."""
    if not ALMACEN:
        return JSONResponse({"error": "Almacen no conectado"}, status_code=500)
    d = await request.json()
    carpeta = str(d.get("carpeta", ""))
    dorsal = re.sub(r"\D", "", str(d.get("dorsal", "")))[:2]
    nombre = identidad.limpiar_nombre(str(d.get("nombre", "")))
    equipo = identidad.limpiar_nombre(str(d.get("equipo", ""))) or "Sin_Equipo"
    if not carpeta or carpeta == GENERAL:
        return JSONResponse({"error": "Carpeta invalida"}, status_code=400)
    if dorsal and int(dorsal) > 0:
        nueva = f"Dorsal_{int(dorsal):02d}_{equipo}" + (f"_{nombre}" if nombre else "")
    elif nombre:
        nueva = f"{nombre}_{equipo}"
    else:
        return JSONResponse({"error": "Escribe al menos el numero o el nombre"}, status_code=400)

    def mover():
        ses = _leer_sesion()
        detalle = ses.get("detalle", {})
        unida = False
        # si ya hay otra ficha con el mismo numero y equipo, es el mismo jugador: se unen en una sola
        if dorsal:
            pref = f"Dorsal_{int(dorsal):02d}_{equipo}"
            gemela = next((c for c in sorted({k.split("/")[1] for k, _ in ALMACEN.listar(PREF_ORG)})
                           if c != carpeta and (c == pref or c.startswith(pref + "_"))), None)
            if gemela:
                _mover_carpeta(carpeta, gemela)
                detalle.pop(carpeta, None)
                carpeta_actual, unida = gemela, True
            else:
                carpeta_actual = carpeta
        else:
            carpeta_actual = carpeta
        if carpeta_actual != nueva:
            unida = unida or bool(ALMACEN.listar(f"{PREF_ORG}{nueva}/"))
            _mover_carpeta(carpeta_actual, nueva)
            detalle.pop(carpeta_actual, None)
        detalle[nueva] = {"equipo": equipo, "dorsal": dorsal or None, "nombre": nombre or None, "revisada": True,
                          "fotos": len(ALMACEN.listar(f"{PREF_ORG}{nueva}/"))}
        ses["detalle"] = detalle
        for duda in ses.get("dudas", []):   # las dudas pendientes siguen a su jugador
            if duda["carpeta"] in (carpeta, carpeta_actual):
                duda["carpeta"], duda["dorsal"], duda["equipo"] = nueva, dorsal or None, equipo
        _guardar_sesion(ses)
        return {"carpeta": nueva, "unida": unida}

    try:
        return await run_in_threadpool(mover)
    except Exception as e:
        logger.exception("corregir")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/a-generales")
async def mover_a_generales(request: Request):
    """La ficha no es un jugador (publico, arbitro...): sus fotos pasan a Generales si no estan en otra carpeta."""
    d = await request.json()
    carpeta = str(d.get("carpeta", ""))
    if not carpeta or carpeta == GENERAL:
        return JSONResponse({"error": "Carpeta invalida"}, status_code=400)

    def mover():
        origen = [k for k, _ in ALMACEN.listar(f"{PREF_ORG}{carpeta}/")]
        en_otras = {os.path.basename(k) for k, _ in ALMACEN.listar(PREF_ORG) if not k.startswith(f"{PREF_ORG}{carpeta}/")}
        for k in origen:
            if os.path.basename(k) not in en_otras:
                ALMACEN.copiar(k, f"{PREF_ORG}{GENERAL}/{os.path.basename(k)}")
        ALMACEN.borrar(origen + [f"{PREF_MINI}{carpeta}.jpg"])
        ses = _leer_sesion()
        ses.get("detalle", {}).pop(carpeta, None)
        for duda in ses.get("dudas", []):
            if duda["carpeta"] == carpeta and duda["estado"] == "pendiente":
                duda["estado"] = "descartada"
        _guardar_sesion(ses)
        return {"ok": True}

    return await run_in_threadpool(mover)


@app.get("/api/boveda")
def get_backup_vault():
    if not ALMACEN:
        return {"backup_photos": [], "count": 0}
    try:
        files = [os.path.basename(k) for k, _ in ALMACEN.listar(PREF_BOVEDA)]
        return {"backup_photos": files, "count": len(files)}
    except Exception:
        return {"backup_photos": [], "count": 0}


@app.get("/api/boveda/foto/{filename}")
def get_backup_photo(filename: str):
    if not ALMACEN:
        return JSONResponse({"error": "No R2"}, status_code=500)
    try:
        datos = ALMACEN.leer(PREF_BOVEDA + _nombre_seguro(filename))
        return Response(datos, media_type="image/jpeg",
                        headers={"Content-Disposition": f'attachment; filename="{_nombre_seguro(filename)}"'})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.post("/api/procesar")
def trigger_process(background_tasks: BackgroundTasks, nomina: str = Form("")):
    if processing_status["is_processing"]:
        return JSONResponse({"status": "busy", "message": "Ya hay un proceso en marcha."})
    processing_status["is_processing"] = True   # evita doble clic mientras arranca
    background_tasks.add_task(process_photos_task, nomina)
    return {"status": "started", "message": "Organizacion biometrica por jugador iniciada."}


@app.post("/api/limpiar")
def clean_previous_session():
    """Deja todo listo para un partido nuevo (la boveda de respaldo NO se toca)."""
    if not ALMACEN:
        return JSONResponse({"error": "No R2"}, status_code=500)
    if processing_status["is_processing"]:
        return JSONResponse({"error": "Hay un proceso en marcha"}, status_code=409)
    try:
        _limpiar_sesion(borrar_fotos=True)
        return {"status": "ok", "message": "Sesión limpiada con éxito."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/subir-y-procesar")
async def upload_and_process(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...),
                             limpiar: int = 1, procesar: int = 1, nomina: str = Form("")):
    """Recibe fotos. La pantalla nueva las manda por tandas (limpiar=1 solo en la primera, procesar=0)
    y al final llama a /api/procesar. Llamado sin parametros se comporta como antes: limpia, sube y procesa."""
    if not ALMACEN:
        return JSONResponse({"error": "R2 no conectado."}, status_code=500)
    if limpiar:
        if processing_status["is_processing"]:
            return JSONResponse({"error": "Hay un proceso en marcha"}, status_code=409)
        await run_in_threadpool(_limpiar_sesion, True)
        processing_status["inicio"] = time.time()
    processing_status["fase"] = "subiendo"
    count = 0
    for file in files:
        if not file.filename:
            continue
        datos = await file.read()
        nombre = _nombre_seguro(file.filename)

        def guardar(nombre=nombre, datos=datos):
            key = nombre
            if ALMACEN.existe(key):   # mismo nombre de archivo de otra camara en este partido
                b, e = os.path.splitext(nombre)
                i = 2
                while ALMACEN.existe(f"{b}_{i}{e}"):
                    i += 1
                key = f"{b}_{i}{e}"
            ALMACEN.subir(key, datos)
            bkey = PREF_BOVEDA + key
            if ALMACEN.existe(bkey):
                bkey = PREF_BOVEDA + time.strftime("%Y%m%d-%H%M%S_") + key
            ALMACEN.copiar(key, bkey)
            return key

        key = await run_in_threadpool(guardar)
        _encolar(key, datos)          # el analisis empieza ya, mientras siguen llegando fotos
        datos = None
        count += 1
    processing_status["total_photos"] = len(_tareas)
    if procesar:
        if not processing_status["is_processing"]:
            processing_status["is_processing"] = True
            background_tasks.add_task(process_photos_task, nomina)
    return {"status": "ok", "uploaded": count, "message": f"{count} fotos subidas."}


def _armar_zip(prefijo: str, renombrar) -> str:
    fd, ruta = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(ruta, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:   # JPG ya va comprimido
        for key, _ in ALMACEN.listar(prefijo):
            with zf.open(renombrar(key), "w", force_zip64=True) as dst:
                ALMACEN.descargar_a(key, dst)
    return ruta


@app.get("/api/descargar-zip")
def download_all_zip():
    """ZIP con una carpeta por jugador. Se arma en disco (no en RAM) para aguantar partidos grandes.
    La sesion NO se borra al descargar: si la descarga falla se puede repetir; se limpia al empezar otro partido."""
    if not ALMACEN:
        return JSONResponse({"error": "No R2"}, status_code=500)
    ruta = _armar_zip(PREF_ORG, lambda k: k[len(PREF_ORG):])
    logger.info("ZIP generado: %s bytes", os.path.getsize(ruta))
    return FileResponse(ruta, media_type="application/zip", filename="Fotos_Organizadas_CastilloEnfoca.zip",
                        background=BackgroundTask(os.remove, ruta))


@app.get("/api/descargar-boveda-zip")
def download_backup_vault_zip():
    if not ALMACEN:
        return JSONResponse({"error": "No R2"}, status_code=500)
    ruta = _armar_zip(PREF_BOVEDA, lambda k: "Respaldo_Seguridad/" + k[len(PREF_BOVEDA):])
    return FileResponse(ruta, media_type="application/zip", filename="Boveda_Respaldo_CastilloEnfoca.zip",
                        background=BackgroundTask(os.remove, ruta))


@app.get("/")
def clean_dashboard():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pantalla.html"), encoding="utf-8") as f:
        html = f.read()
    response = HTMLResponse(content=html)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
