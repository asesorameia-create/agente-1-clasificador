"""
Motor biometrico de CastilloEnfoca (Agente 1 Clasificador).

Pipeline por foto (todo con OpenCV DNN, sin PyTorch, ~110 MB de modelos):
  1. YOLOX-s       -> detecta cada PERSONA en la foto.
  2. YuNet         -> detecta CARAS; SFace -> huella facial de 128 dimensiones.
  3. PP-OCRv3 DB   -> encuentra zonas de texto DENTRO de cada jugador (pecho, espalda, short).
     CRNN          -> lee esas zonas; para el dorsal se decodifica SOLO con digitos.
  4. Color de camiseta -> histograma de color del torso para separar EQUIPOS.

Luego, sobre todo el lote (clasificar_lote):
  - Equipos: agrupacion por color de camiseta + nombre leido en la camiseta (ej. "Trujillo").
  - Jugador = (equipo, dorsal). Su huella facial se aprende de las fotos donde se ve numero y cara.
  - Fotos sin numero visible: se asignan por huella facial (similitud coseno SFace).
  - Caras recurrentes sin numero nunca visto -> Jugador_XX_Equipo.
  - Rafagas (fotos seguidas en el tiempo): se propaga la identidad al vecino casi identico.
"""
from __future__ import annotations

import gc
import io
import os
import re
import logging
import difflib
import threading
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

logger = logging.getLogger("CastilloEnfoca.motor")

MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
MODEL_FILES = {
    "yolox": "object_detection_yolox_2022nov.onnx",
    "yunet": "face_detection_yunet_2023mar.onnx",
    "sface": "face_recognition_sface_2021dec.onnx",
    "db": "text_detection_en_ppocrv3_2023may.onnx",
    "crnn": "text_recognition_CRNN_EN_2021sep.onnx",
}
# Versiones cuantizadas int8 oficiales de OpenCV Zoo: ~4x menos memoria (necesario en Render de 512 MB).
# MODELOS_LIGEROS=1 activa las que se validaron sin perdida de precision.
MODEL_FILES_INT8 = {
    "yolox": "object_detection_yolox_2022nov_int8.onnx",
    "sface": "face_recognition_sface_2021dec_int8.onnx",
    "crnn": "text_recognition_CRNN_EN_2022oct_int8.onnx",
}
for _k in [k for k in os.getenv("MODELOS_LIGEROS_LISTA", "yolox,sface,crnn").split(",") if k]:
    if os.getenv("MODELOS_LIGEROS", "0") == "1" and _k in MODEL_FILES_INT8:
        MODEL_FILES[_k] = MODEL_FILES_INT8[_k]

# ---------------------------------------------------------------- parametros
LADO_TRABAJO = 2000          # lado mayor de la imagen de trabajo (memoria ~9 MB)
LADO_CARAS = 1280            # lado mayor para buscar caras
PERSONA_MIN_SCORE = 0.45
PERSONA_MIN_ALTURA = 0.16    # fraccion de la altura de la foto para considerar a un jugador
MAX_PERSONAS = int(os.getenv("MAX_PERSONAS", "16"))   # fotos de grupo: nadie se queda fuera
PERSONAS_SIEMPRE = 5         # las 5 mas grandes siempre; de la 6a en adelante, solo si se le ve la cara
ALTURA_RECORTE = 800         # el recorte de cada jugador se lleva a esta altura para leer el dorsal
YOLO_LADO = int(os.getenv("YOLO_LADO", "416"))  # entrada del detector de personas (multiplo de 32)
DORSAL_MIN_CONF = 0.86
CARA_MIN_PX = 34             # cara minima (en imagen de LADO_CARAS) para sacar huella
CARA_MIN_SCORE = 0.80
CARA_TTA_PX = 56             # caras nitidas: huella con cara + espejo (mas estable); pequenas: una pasada
PERSONAS_CON_TEXTO = 4       # se busca dorsal en los 4 protagonistas de cada foto

CRNN_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"

# Los modelos de OpenCV DNN no son seguros entre hilos: cada hilo de analisis carga los suyos.
_local = threading.local()


def modelos_disponibles() -> bool:
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in MODEL_FILES.values())


def _cargar_modelos() -> Dict[str, Any]:
    if getattr(_local, "modelos", None):
        return _local.modelos
    _modelos: Dict[str, Any] = {}
    _local.modelos = _modelos
    if True:
        p = lambda k: os.path.join(MODELS_DIR, MODEL_FILES[k])
        _modelos["yolox"] = cv2.dnn.readNet(p("yolox"))
        _modelos["yunet"] = cv2.FaceDetectorYN.create(p("yunet"), "", (320, 320), 0.6, 0.3, 5000)
        _modelos["sface"] = cv2.FaceRecognizerSF.create(p("sface"), "")
        db = cv2.dnn.TextDetectionModel_DB(p("db"))
        db.setBinaryThreshold(0.3).setPolygonThreshold(0.5).setMaxCandidates(60).setUnclipRatio(2.0)
        _modelos["db"] = db
        _modelos["crnn"] = cv2.dnn.readNet(p("crnn"))
        # grillas de YOLOX (YOLO_LADO x YOLO_LADO, strides 8/16/32)
        grids, strides = [], []
        for s in (8, 16, 32):
            n = YOLO_LADO // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        _modelos["grid"] = np.concatenate(grids, 0).astype(np.float32)
        _modelos["stride"] = np.concatenate(strides, 0).astype(np.float32)
        logger.info("Modelos biometricos cargados desde %s", MODELS_DIR)
        return _modelos


# ================================================================ utilidades
def _hora_exif(img_bytes: bytes) -> Optional[float]:
    """Hora de captura (segundos) desde EXIF, con subsegundos si existen."""
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
            dt = sub.get(36867) or exif.get(306)
            if not dt:
                return None
            import datetime as _dt
            t = _dt.datetime.strptime(str(dt).strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
            ss = sub.get(37521)
            if ss:
                try:
                    t += float("0." + str(ss).strip())
                except ValueError:
                    pass
            return t
    except Exception:
        return None


def _decodificar(img_bytes: bytes) -> Optional[np.ndarray]:
    """Decodifica reduciendo en el propio JPEG (ahorra RAM) y lleva el lado mayor a LADO_TRABAJO."""
    arr = np.frombuffer(img_bytes, np.uint8)
    flag = cv2.IMREAD_COLOR
    try:
        if Image is not None:
            with Image.open(io.BytesIO(img_bytes)) as im:
                lado = max(im.size)
            if lado >= LADO_TRABAJO * 4:
                flag = cv2.IMREAD_REDUCED_COLOR_4
            elif lado >= LADO_TRABAJO * 2:
                flag = cv2.IMREAD_REDUCED_COLOR_2
    except Exception:
        pass
    img = cv2.imdecode(arr, flag)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > LADO_TRABAJO:
        s = LADO_TRABAJO / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _detectar_personas(m, img: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    h, w = img.shape[:2]
    r = min(YOLO_LADO / h, YOLO_LADO / w)
    nh, nw = int(h * r), int(w * r)
    pad = np.full((YOLO_LADO, YOLO_LADO, 3), 114, dtype=np.uint8)
    pad[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(pad, 1.0, (YOLO_LADO, YOLO_LADO), (0, 0, 0), swapRB=True)
    m["yolox"].setInput(blob)
    out = m["yolox"].forward()[0]
    out[:, :2] = (out[:, :2] + m["grid"]) * m["stride"]
    out[:, 2:4] = np.exp(out[:, 2:4]) * m["stride"]
    score = out[:, 4] * out[:, 5]                     # objectness * clase "persona"
    keep = score > PERSONA_MIN_SCORE
    out, score = out[keep], score[keep]
    if len(out) == 0:
        return []
    cx, cy, bw, bh = out[:, 0] / r, out[:, 1] / r, out[:, 2] / r, out[:, 3] / r
    boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), PERSONA_MIN_SCORE, 0.45)
    res = []
    for i in np.array(idx).reshape(-1):
        x, y, ww, hh = boxes[i]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + ww)), min(h, int(y + hh))
        if x2 - x1 > 8 and y2 - y1 > 16:
            res.append((x1, y1, x2, y2, float(score[i])))
    return _sin_duplicados(res)


def _sin_duplicados(cajas):
    """Un recuadro casi entero dentro de otro de tamano parecido es la MISMA persona detectada dos veces
    (medido en vivo: el #10 rojo salio en dos carpetas desde la misma foto). Se queda el mas grande."""
    cajas = sorted(cajas, key=lambda c: -(c[2] - c[0]) * (c[3] - c[1]))
    quedan = []
    for c in cajas:
        dup = False
        for g in quedan:
            ix = max(0, min(c[2], g[2]) - max(c[0], g[0]))
            iy = max(0, min(c[3], g[3]) - max(c[1], g[1]))
            area_c = max(1, (c[2] - c[0]) * (c[3] - c[1]))
            if ix * iy / area_c >= 0.85 and (c[3] - c[1]) >= 0.6 * (g[3] - g[1]):
                dup = True
                break
        if not dup:
            quedan.append(c)
    return quedan


def _detectar_caras(m, img: np.ndarray):
    h, w = img.shape[:2]
    s = min(1.0, LADO_CARAS / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    m["yunet"].setInputSize((small.shape[1], small.shape[0]))
    _, faces = m["yunet"].detect(small)
    return small, s, ([] if faces is None else list(faces))


def _ctc(m, gray: np.ndarray, solo_digitos: bool) -> Tuple[str, float]:
    blob = cv2.dnn.blobFromImage(cv2.resize(gray, (100, 32), interpolation=cv2.INTER_AREA),
                                 1 / 127.5, (100, 32), 127.5)
    m["crnn"].setInput(blob)
    o = m["crnn"].forward()[:, 0, :]
    e = np.exp(o - o.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    if solo_digitos:
        p = p[:, :11]
        p = p / p.sum(1, keepdims=True)
    idx, conf = p.argmax(1), p.max(1)
    txt, cs, prev = "", [], 0
    for i, c in enumerate(idx):
        if c != 0 and c != prev:
            txt += CRNN_CHARS[c - 1]
            cs.append(conf[i])
        prev = c
    # la confianza incluye los "blancos": un texto con ruido entre caracteres baja la nota
    return txt, (float(min(np.mean(cs), np.mean(conf))) if cs else 0.0)


def _leer_digitos(m, canales: List[np.ndarray], rect) -> Tuple[str, float]:
    """Lee un dorsal probando recorte justo / leve margen, varios canales de color y la version invertida.
    (Medido en fotos reales: el recorte justo es el que mejor lee; ampliar mete letras y lo confunde.)"""
    H, W = canales[0].shape[:2]
    x, y, w, h = rect
    votos: Dict[str, List[float]] = defaultdict(list)
    for ex in (0.0, 0.1):
        if ex > 0 and votos and max(max(v) for v in votos.values()) >= 0.90:
            break  # ya hay una lectura clara: no gastar tiempo
        dx, dy = int(w * ex), int(h * (0.06 + ex))
        x1, y1 = max(0, x - dx), max(0, y - dy)
        x2, y2 = min(W, x + w + dx), min(H, y + h + dy)
        for ci, ch in enumerate(canales):
            if ci > 0 and votos and max(max(v) for v in votos.values()) >= 0.90:
                break  # el canal gris ya leyo claro
            crop = ch[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 8:
                continue
            for g in (crop, 255 - crop):
                t, c = _ctc(m, g, True)
                if re.fullmatch(r"[1-9][0-9]?", t):
                    votos[t].append(c)
    if not votos:
        return "", 0.0
    # gana la lectura con mejor confianza; si varias variantes coinciden, se refuerza
    t = max(votos, key=lambda k: (max(votos[k]) + 0.02 * (len(votos[k]) - 1)))
    return t, min(1.0, max(votos[t]) + 0.02 * (len(votos[t]) - 1))


def _agrupar_lineas(rects: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Une cajas vecinas de la misma linea (ej. '1' y '3' separados -> '13')."""
    rs = sorted(rects, key=lambda r: r[0])
    usados, res = set(), []
    for i, a in enumerate(rs):
        if i in usados:
            continue
        x1, y1, x2, y2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        for j in range(i + 1, len(rs)):
            if j in usados:
                continue
            b = rs[j]
            hb, ha = b[3], (y2 - y1)
            solape = min(y2, b[1] + hb) - max(y1, b[1])
            if solape > 0.6 * min(ha, hb) and 0.6 < hb / max(1, ha) < 1.6 and b[0] - x2 < 0.6 * max(ha, hb):
                x2 = max(x2, b[0] + b[2]); y1 = min(y1, b[1]); y2 = max(y2, b[1] + hb)
                usados.add(j)
        res.append((x1, y1, x2 - x1, y2 - y1))
    return res


def _ganancias_blanco(img: np.ndarray) -> np.ndarray:
    """Correccion de tinte por foto (sombra azulada, sol amarillo): 'white patch' robusto
    sobre los pixeles mas brillantes y poco saturados de la escena."""
    small = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    lum = small.mean(1)
    sat = small.max(1) - small.min(1)
    sel = small[(lum > np.percentile(lum, 85)) & (sat < np.percentile(sat, 60))]
    if len(sel) < 30:
        return np.ones(3, np.float32)
    ref = np.median(sel, axis=0)
    g = ref.mean() / np.maximum(ref, 1.0)
    return np.clip(g, 0.75, 1.33).astype(np.float32)


def _balance_blancos(bgr: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.clip(bgr.astype(np.float32) * g, 0, 255).astype(np.uint8)


# Firma de color de camiseta en Lab: 12 tonos x 2 niveles de croma + blanco, gris, negro.
def _histo_camiseta(bgr: np.ndarray) -> Optional[np.ndarray]:
    if bgr is None or bgr.size == 0:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0] * (100.0 / 255.0)
    a, b = lab[:, 1] - 128.0, lab[:, 2] - 128.0
    croma = np.hypot(a, b)
    hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    # descartar piel aproximada (tonos calidos de croma media y luminosidad media)
    piel = (hue > 20) & (hue < 75) & (croma > 10) & (croma < 40) & (L > 25) & (L < 85)
    L, croma, hue = L[~piel], croma[~piel], hue[~piel]
    if len(L) < 50:
        return None
    hist = np.zeros(27, np.float32)
    # blancos y negros reales toleran mas tinte (sombra azulada, reflejo del cesped)
    tolerancia = 12.0 + 0.45 * np.maximum(0.0, L - 60.0) + 0.6 * np.maximum(0.0, 32.0 - L)
    acrom = croma < tolerancia
    negro = acrom & (L < 32)
    blanco = acrom & (L > 66)
    gris = acrom & ~negro & ~blanco
    hist[24], hist[25], hist[26] = blanco.sum(), gris.sum(), negro.sum()
    cromo = ~acrom
    t = (hue[cromo] // 30).astype(np.int32).clip(0, 11)
    fuerte = (croma[cromo] >= 30).astype(np.int32)
    np.add.at(hist, t * 2 + fuerte, 1.0)
    tot = hist.sum()
    return hist / tot if tot > 0 else None


def _firma_numero(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Los 2 colores del dorsal (cifra y fondo) en Lab, ordenados por luminosidad.
    Mismo equipo = misma tipografia y colores del numero, en cualquier partido."""
    if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 6:
        return None
    px = cv2.cvtColor(cv2.resize(bgr, (24, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cen = cen[np.argsort(cen[:, 0])]
    cen[:, 0] *= 100.0 / 255.0
    cen[:, 1:] -= 128.0
    return cen.flatten()


def dist_firma_numero(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Diferencia media (Delta E aprox.) entre los colores de dos dorsales; 999 si falta alguno."""
    if a is None or b is None:
        return 999.0
    a, b = a.reshape(2, 3), b.reshape(2, 3)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _dist_color(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 1.0
    # distancia "suavizada": compara histogramas difuminados entre tonos vecinos
    return 1.0 - float(np.minimum(_suavizar(a), _suavizar(b)).sum())


def _suavizar(h: np.ndarray) -> np.ndarray:
    c = h[:24].reshape(12, 2).sum(1)
    c = 0.6 * c + 0.2 * np.roll(c, 1) + 0.2 * np.roll(c, -1)
    out = np.concatenate([c, h[24:]])
    return out / max(1e-6, out.sum())


# tono Lab en tramos de 30 grados
NOMBRES_COLOR = ["Rosado", "Rojo", "Naranja", "Amarillo", "Verde", "Verde", "Verde", "Celeste", "Azul", "Azul", "Morado", "Rosado"]


def _nombre_color(h: np.ndarray) -> str:
    ach = {"Blanco": h[24], "Gris": h[25], "Negro": h[26]}
    tonos = h[:24].reshape(12, 2).sum(1)
    top_t = int(tonos.argmax())
    cand = sorted(list(ach.items()) + [(NOMBRES_COLOR[top_t], tonos[top_t])], key=lambda x: -x[1])
    a = cand[0][0]
    b = cand[1][0] if len(cand) > 1 and cand[1][1] > 0.25 * cand[0][1] and cand[1][0] != a else ""
    return a + (b if b else "")


def _nitidez(jpg: bytes) -> float:
    """Nitidez de la cara (varianza del Laplaciano sobre la miniatura a 160 px). Medido en caras reales del
    cliente: desenfocadas de fondo 15-47, nitidas 81-2085."""
    try:
        g = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (160, 160), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _miniatura(img: np.ndarray, x1, y1, x2, y2, lado=160) -> bytes:
    h, w = img.shape[:2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * 0.75
    a, b = int(max(0, cx - s)), int(max(0, cy - s))
    c, d = int(min(w, cx + s)), int(min(h, cy + s))
    crop = img[b:d, a:c]
    if crop.size == 0:
        return b""
    crop = cv2.resize(crop, (lado, int(lado * crop.shape[0] / max(1, crop.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else b""


FLORENCE_MAX_PERSONAS = int(os.getenv("FLORENCE_MAX_PERSONAS", "3"))
FLORENCE_ALTURA_MIN = 0.22


def _votar_florence(jugadores: List[Dict[str, Any]], torsos: List[np.ndarray]) -> None:
    """Jurado de lectores: lector rapido (CRNN) + Florence-2. Reglas medidas con fotos reales:
    - coinciden -> numero seguro; desacuerdo -> ninguno;
    - un solo lector: se acepta solo si son 2 cifras vistas de espaldas (las lecturas parciales son de 1 cifra);
    - dos numeros o dos equipos en el mismo recorte (se colo otro jugador) -> no se asigna."""
    try:
        import lector_florence
        import identidad
    except Exception:
        return
    if not lector_florence.disponible():
        return
    for i, j in enumerate(jugadores[:FLORENCE_MAX_PERSONAS]):
        if i >= len(torsos) or j["altura"] < FLORENCE_ALTURA_MIN:
            continue
        local, conf = j.get("dorsal"), j.get("conf_dorsal", 0.0)
        if local and conf >= 0.97:
            continue                      # el lector rapido ya esta seguro
        r = lector_florence.leer(torsos[i])
        if r is None:
            continue
        nums = list(dict.fromkeys(r["numeros"]))
        equipos = identidad.equipos_en_texto(r["texto"])
        j["texto_florence"] = r["texto"][:80]
        if len(equipos) == 1:
            j["equipo_vision"] = next(iter(equipos))
        ambiguo = len(nums) > 1 or len(equipos) > 1
        if ambiguo or not nums:
            if local and conf < 0.97:     # nadie confirma la lectura dudosa: se descarta
                j["lecturas_debiles"] = [local] + nums
                j["dorsal"], j["conf_dorsal"] = None, 0.0
                j.pop("firma_numero", None)
            continue
        f = nums[0]
        if local == f:
            j["dorsal"], j["conf_dorsal"], j["fuente_dorsal"] = f, 0.99, "votacion"   # 2 votos
        elif local:
            j["lecturas_debiles"] = [local, f]                                         # desacuerdo
            j["dorsal"], j["conf_dorsal"] = None, 0.0
            j.pop("firma_numero", None)
        else:
            de_espaldas = j.get("huella") is None
            j["dorsal"], j["fuente_dorsal"] = f, "florence"
            # un solo lector nunca basta para nombrar la carpeta (medido: leyo "16" donde era 15):
            # queda como sugerencia que el fotografo confirma con un toque
            j["conf_dorsal"] = 0.93
            j["alto_dorsal"] = max(j.get("alto_dorsal", 0.0), 0.09)
    # un mismo numero no puede ser de dos jugadores de la misma foto: se queda el de mas votos
    por_num: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for j in jugadores:
        if j.get("dorsal"):
            por_num[j["dorsal"]].append(j)
    for js in por_num.values():
        if len(js) > 1:
            js.sort(key=lambda x: -x.get("conf_dorsal", 0.0))
            for x in js[1:]:
                x["lecturas_debiles"] = [x["dorsal"]]
                x["dorsal"], x["conf_dorsal"] = None, 0.0
                x.pop("firma_numero", None)


SMOL_MAX_PERSONAS = int(os.getenv("SMOL_MAX_PERSONAS", "3"))
SMOL_FUERTE_SI, SMOL_FUERTE_CONF = 0.90, 0.95   # solo con esta seguridad SmolVLM nombra la carpeta sin otro voto


def _votar_smol(jugadores: List[Dict[str, Any]], torsos: List[np.ndarray]) -> None:
    """Tercer voto: SmolVLM (se le pregunta el numero). Lee los numeros grandes decorados que los lectores
    de texto no ven. Coincide con otro lector -> seguro; solo y muy seguro -> seguro; si no, sugerencia."""
    try:
        import lector_smol
    except Exception:
        return
    if not lector_smol.disponible():
        return
    for i, j in enumerate(jugadores[:SMOL_MAX_PERSONAS]):
        if i >= len(torsos) or j["altura"] < FLORENCE_ALTURA_MIN:
            continue
        local, conf = j.get("dorsal"), j.get("conf_dorsal", 0.0)
        if local and conf >= 0.97:
            continue
        r = lector_smol.leer(torsos[i])
        if not r or not r["numero"]:
            continue
        n = r["numero"]
        j["lectura_smol"] = n
        debiles = [x for x in (j.get("lecturas_debiles") or []) if x]
        if local == n or n in debiles:
            j["dorsal"], j["conf_dorsal"], j["fuente_dorsal"] = n, 0.99, "votacion"     # 2 lectores coinciden
            j["alto_dorsal"] = max(j.get("alto_dorsal", 0.0), 0.09)
        elif local:
            j["lecturas_debiles"] = [local, n]                                          # desacuerdo: nadie
            j["dorsal"], j["conf_dorsal"] = None, 0.0
            j.pop("firma_numero", None)
        elif r["si"] >= SMOL_FUERTE_SI and r["conf"] >= SMOL_FUERTE_CONF:
            j["dorsal"], j["conf_dorsal"], j["fuente_dorsal"] = n, 0.97, "smol"
            j["alto_dorsal"] = max(j.get("alto_dorsal", 0.0), 0.09)
        else:
            j["lecturas_debiles"] = debiles + [n]
    por_num: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for j in jugadores:
        if j.get("dorsal"):
            por_num[j["dorsal"]].append(j)
    for js in por_num.values():
        if len(js) > 1:
            js.sort(key=lambda x: -x.get("conf_dorsal", 0.0))
            for x in js[1:]:
                x["lecturas_debiles"] = [x["dorsal"]]
                x["dorsal"], x["conf_dorsal"] = None, 0.0
                x.pop("firma_numero", None)


JURADO_MAX_PERSONAS = int(os.getenv("JURADO_MAX_PERSONAS", "4"))


def _jurado(jugadores: List[Dict[str, Any]], torsos_vision: List[np.ndarray], torsos_smol: List[np.ndarray]) -> None:
    """JURADO ESCALONADO de lectores de dorsal (cada uno mira la foto a su manera):
      1) siempre: lector rapido (CRNN, ya leyo) + SmolVLM (se le PREGUNTA el numero);
      2) si coinciden -> numero confirmado, no se molesta al tercero;
      3) si discrepan o solo uno vio numero -> Florence-2 desempata: gana la mayoria (2 de 3);
      4) sin mayoria -> no se nombra (queda como sugerencia). Nunca se inventa.
    El lector lento solo trabaja cuando hay desacuerdo: varios puntos de vista sin perder velocidad."""
    try:
        import lector_smol
        hay_smol = lector_smol.disponible()
    except Exception:
        hay_smol = False
    try:
        import lector_florence
        import identidad
        hay_florence = lector_florence.disponible()
    except Exception:
        hay_florence = False
    for i, j in enumerate(jugadores[:JURADO_MAX_PERSONAS]):
        if j["altura"] < FLORENCE_ALTURA_MIN:
            continue
        local, conf = j.get("dorsal"), j.get("conf_dorsal", 0.0)
        if local and conf >= 0.97 and len(local) == 2:
            continue                                        # dos cifras leidas con seguridad: no hace falta jurado
        # UNA cifra siempre va a votacion: es la que mas se confunde (logo "R5" -> 5, un "15" tapado -> 5)
        votos: List[str] = [local] if local else []
        smol = None
        if hay_smol and i < len(torsos_smol):
            r = lector_smol.leer(torsos_smol[i])
            if r and r["numero"]:
                smol = r
                votos.append(r["numero"])
                j["lectura_smol"] = r["numero"]
        flor = None
        necesita_juez = len(votos) == 1 or (len(votos) == 2 and votos[0] != votos[1])
        if necesita_juez and hay_florence and i < len(torsos_vision):
            r = lector_florence.leer(torsos_vision[i])
            if r is not None:
                j["texto_florence"] = r["texto"][:80]
                equipos = identidad.equipos_en_texto(r["texto"])
                if len(equipos) == 1:
                    j["equipo_vision"] = next(iter(equipos))
                nums = list(dict.fromkeys(r["numeros"]))
                if len(nums) == 1 and len(equipos) <= 1:   # dos numeros/equipos en el recorte: se colo otro
                    flor = nums[0]
                    votos.append(flor)
        cuenta = Counter(votos)
        gana, n = cuenta.most_common(1)[0] if cuenta else (None, 0)
        if n >= 2:
            j["dorsal"], j["conf_dorsal"], j["fuente_dorsal"] = gana, 0.99, "votacion"
            j["alto_dorsal"] = max(j.get("alto_dorsal", 0.0), 0.09)
        elif smol and not local and flor in (None, smol["numero"]) and len(smol["numero"]) == 2                 and smol["si"] >= SMOL_FUERTE_SI and smol["conf"] >= SMOL_FUERTE_CONF:
            j["dorsal"], j["conf_dorsal"], j["fuente_dorsal"] = smol["numero"], 0.97, "smol"
            j["alto_dorsal"] = max(j.get("alto_dorsal", 0.0), 0.09)
        elif votos:
            j["lecturas_debiles"] = list(dict.fromkeys(votos))
            j["dorsal"], j["conf_dorsal"] = None, 0.0
            j.pop("firma_numero", None)
    # un mismo numero no puede ser de dos jugadores de la misma foto: se queda el de mas votos
    por_num: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for j in jugadores:
        if j.get("dorsal"):
            por_num[j["dorsal"]].append(j)
    for js in por_num.values():
        if len(js) > 1:
            js.sort(key=lambda x: -x.get("conf_dorsal", 0.0))
            for x in js[1:]:
                x["lecturas_debiles"] = [x["dorsal"]]
                x["dorsal"], x["conf_dorsal"] = None, 0.0
                x.pop("firma_numero", None)


def _fusionar_vision(jugadores: List[Dict[str, Any]], torsos: List[np.ndarray]) -> None:
    """Combina el lector local con la lectura de Claude (si esta disponible) sobre los protagonistas."""
    try:
        import lector_vision
    except Exception:
        return
    if not lector_vision.disponible() or not jugadores:
        return
    lect = lector_vision.leer_torsos(torsos[:lector_vision.MAX_TORSOS])
    for i, r in lect.items():
        if i >= len(jugadores):
            continue
        j = jugadores[i]
        j["vision"] = True
        if r.get("equipo"):
            j["equipo_vision"] = str(r["equipo"])[:40]
        if not r["camiseta_puesta"] and r["numero"]:
            j["numero_en_mano"] = True            # p. ej. camiseta entregada en una ceremonia
        if r["numero_seguro"] and r["camiseta_puesta"]:
            local = j.get("dorsal")
            j["dorsal"] = str(r["numero"])
            j["conf_dorsal"] = 0.99
            if local != j["dorsal"] or not j.get("alto_dorsal"):
                j["alto_dorsal"] = max(j.get("alto_dorsal", 0) if local == j["dorsal"] else 0, 0.09)
            j["fuente_dorsal"] = "vision"
        elif j.get("dorsal") and j.get("conf_dorsal", 0) < 0.97:
            # Claude no ve un numero legible: la lectura local dudosa se descarta
            j["dorsal"], j["conf_dorsal"] = None, 0.0
            j.pop("firma_numero", None)


# ================================================================ analisis por foto
def analizar_foto(img_bytes: bytes, nombre: str = "") -> Dict[str, Any]:
    """Extrae la 'huella' de cada jugador visible en una foto."""
    m = _cargar_modelos()
    res: Dict[str, Any] = {"nombre": nombre, "hora": _hora_exif(img_bytes), "jugadores": [], "firma": None, "ok": False}
    img = _decodificar(img_bytes)
    if img is None:
        return res
    H, W = img.shape[:2]
    try:
        # firma global de la foto (para rafagas)
        tiny = cv2.resize(img, (48, 64) if H > W else (64, 48), interpolation=cv2.INTER_AREA)
        res["firma"] = cv2.calcHist([cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        res["firma"] /= max(1e-6, res["firma"].sum())

        personas = _detectar_personas(m, img)
        small, s_c, caras = _detectar_caras(m, img)

        # jugadores relevantes: los de mayor tamano
        personas.sort(key=lambda p: -(p[3] - p[1]))
        alt_max = (personas[0][3] - personas[0][1]) if personas else 0
        relevantes = [p for p in personas if (p[3] - p[1]) >= PERSONA_MIN_ALTURA * H or (p[3] - p[1]) >= 0.6 * alt_max][:MAX_PERSONAS]

        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ganancias = _ganancias_blanco(img)
        caras_usadas = set()
        torsos_smol: List[np.ndarray] = []
        torsos_vision: List[np.ndarray] = []   # solo del cuello hacia abajo: las caras no salen del servidor
        for n_p, (x1, y1, x2, y2, sc) in enumerate(relevantes):
            ph, pw = y2 - y1, x2 - x1
            jug: Dict[str, Any] = {
                "caja": (x1 / W, y1 / H, x2 / W, y2 / H), "altura": ph / H,
                "dorsal": None, "conf_dorsal": 0.0, "palabras": [], "huella": None,
                "color": None, "mini": b"", "mini_cara": False,
            }
            # ---- cara dentro de la persona (tercio superior)
            mejor_c, mejor_a = None, 0
            for ci, f in enumerate(caras):
                if ci in caras_usadas:
                    continue
                fx, fy, fw, fh = f[0] / s_c, f[1] / s_c, f[2] / s_c, f[3] / s_c
                cxf, cyf = fx + fw / 2, fy + fh / 2
                if x1 <= cxf <= x2 and y1 - 0.05 * ph <= cyf <= y1 + 0.38 * ph and fw * fh > mejor_a:
                    mejor_c, mejor_a = ci, fw * fh
            fbox = None
            if mejor_c is not None:
                f = caras[mejor_c]
                caras_usadas.add(mejor_c)
                fbox = (f[0] / s_c, f[1] / s_c, f[2] / s_c, f[3] / s_c)
                if f[2] >= CARA_MIN_PX and f[-1] >= CARA_MIN_SCORE:
                    try:
                        al = m["sface"].alignCrop(small, f)
                        # huella = promedio de la cara y su espejo (TTA): mas estable ante poses
                        v = m["sface"].feature(al).flatten().astype(np.float32)
                        v /= max(1e-6, np.linalg.norm(v))
                        if f[2] >= CARA_TTA_PX:
                            v2 = m["sface"].feature(cv2.flip(al, 1)).flatten().astype(np.float32)
                            v = v + v2 / max(1e-6, np.linalg.norm(v2))
                        jug["huella"] = v / max(1e-6, np.linalg.norm(v))
                        jug["calidad_cara"] = float(f[-1]) * min(1.0, f[2] / 80.0)
                        fx, fy, fw, fh = f[0] / s_c, f[1] / s_c, f[2] / s_c, f[3] / s_c
                        jug["mini"] = _miniatura(img, fx, fy, fx + fw, fy + fh)
                        jug["mini_cara"] = True
                        jug["nitidez"] = _nitidez(jug["mini"])
                    except cv2.error:
                        pass

            if n_p >= PERSONAS_SIEMPRE and jug["huella"] is None:
                continue          # persona pequena de fondo sin cara: no aporta y gasta tiempo
            # ---- color de camiseta: justo debajo de la cara (si se ve) o centro del torso
            if fbox is not None:
                fx, fy, fw, fh = fbox
                ty1, ty2 = int(fy + 1.4 * fh), int(fy + 3.4 * fh)
                tx1, tx2 = int(fx + fw / 2 - 1.0 * fw), int(fx + fw / 2 + 1.0 * fw)
                ty1, ty2 = max(y1, ty1), min(y2, ty2)
                tx1, tx2 = max(x1, tx1), min(x2, tx2)
            else:
                ty1, ty2 = y1 + int(0.22 * ph), y1 + int(0.45 * ph)
                tx1, tx2 = x1 + int(0.32 * pw), x2 - int(0.32 * pw)
            torso = img[ty1:ty2, tx1:tx2] if (ty2 - ty1 > 6 and tx2 - tx1 > 6) else img[0:0]
            if torso.size:
                torso = cv2.resize(torso, (40, max(8, int(40 * torso.shape[0] / max(1, torso.shape[1])))), interpolation=cv2.INTER_AREA)
                torso = _balance_blancos(torso, ganancias)
                jug["torso"] = torso
                jug["color"] = _histo_camiseta(torso)

            # ---- dorsal y palabras dentro del jugador
            pad = int(0.04 * pw)
            cx1, cx2 = max(0, x1 - pad), min(W, x2 + pad)
            crop = img[y1:y2, cx1:cx2]
            g = gray_full[y1:y2, cx1:cx2]
            canales = [g, crop[:, :, 2]]   # gris + canal rojo (numeros rojos sobre fondo oscuro)
            esc = min(1.6, ALTURA_RECORTE / max(1, ph))
            tw = max(64, int(round(crop.shape[1] * esc / 32)) * 32)
            th = max(64, int(round(crop.shape[0] * esc / 32)) * 32)
            m["db"].setInputParams(1.0 / 255.0, (tw, th), (122.67891434, 116.66876762, 104.00698793), False)
            quads = []
            if n_p < PERSONAS_CON_TEXTO:
                try:
                    quads, _ = m["db"].detect(crop)
                except cv2.error:
                    quads = []
            rects = []
            for q in quads:
                q = np.array(q)
                rx, ry, rw, rh = cv2.boundingRect(q.astype(np.int32))
                rx, ry = max(0, rx), max(0, ry)
                cyr = (ry + rh / 2) / ph
                if rh < 0.035 * ph or not (0.12 <= cyr <= 0.95):
                    continue
                rects.append((rx, ry, rw, rh))
            rects = sorted(rects, key=lambda r: -r[3])[:6]      # los textos mas grandes
            cand = []
            pw_c = crop.shape[1]
            vistos = set()
            for r_ in _agrupar_lineas(rects) + rects:
                if r_ in vistos:
                    continue
                vistos.add(r_)
                rx, ry, rw, rh = r_
                cxr = (rx + rw / 2) / max(1, pw_c)
                # el dorsal va en el centro del cuerpo; los bordes suelen ser brazos o fondo
                if rw / max(1, rh) <= 2.6 and 0.15 <= cxr <= 0.85:
                    t, c = _leer_digitos(m, canales, r_)
                    if t in ("1", "7") and (rw / max(1, rh) < 0.22 or c < 0.93):
                        continue  # trazos finos (brazos, rayas) se confunden con 1 o 7
                    if t and c >= DORSAL_MIN_CONF:
                        # se prefiere el numero mas grande (espalda/pecho) y mas seguro
                        cand.append((c * (rh / ph) ** 0.5 + 0.02 * len(t), t, c, r_))
                if rw / max(1, rh) >= 2.2 and rw > 0.25 * pw:
                    crop_w = g[ry:ry + rh, rx:rx + rw]
                    if crop_w.size:
                        best = ("", 0.0)
                        for gg in (crop_w, 255 - crop_w):
                            t, c = _ctc(m, gg, False)
                            if c > best[1]:
                                best = (t, c)
                        if len(best[0]) >= 5 and best[1] >= 0.55 and best[0].isalpha():
                            jug["palabras"].append(best[0])
            if cand:
                cand.sort(key=lambda z: -z[0])
                _, t, c, (rx, ry, rw, rh) = cand[0]
                jug["dorsal"], jug["conf_dorsal"] = str(int(t)), round(c, 3)
                jug["alto_dorsal"] = rh / ph   # numero grande (espalda/pecho) vs pequeno (short)
                jug["firma_numero"] = _firma_numero(_balance_blancos(crop[ry:ry + rh, rx:rx + rw], ganancias))
                # posicion absoluta del numero y que tan centrado esta en el cuerpo (para desempatar solapes)
                jug["_num_xy"] = ((cx1 + rx + rw / 2) / W, (y1 + ry + rh / 2) / H, rh / H)
                jug["_num_centrado"] = 1.0 - abs((cx1 + rx + rw / 2 - (x1 + x2) / 2) / max(1, pw))
                if not jug["mini"]:
                    bx1, by1 = cx1 + rx, y1 + ry
                    jug["mini"] = _miniatura(img, bx1, by1, bx1 + rw, by1 + rh)
            if not jug["mini"]:
                jug["mini"] = _miniatura(img, x1, y1, x2, y1 + int(0.45 * ph))
            # torso para el lector de vision (sin cabeza)
            top = int(fbox[1] + 1.15 * fbox[3]) if fbox is not None else y1 + int(0.18 * ph)
            top = max(y1, min(top, y1 + int(0.4 * ph)))
            torsos_vision.append(img[top:max(top + 1, y2 - int(0.08 * ph)), max(0, x1 - pad):min(W, x2 + pad)].copy())
            # pecho/espalda sin cabeza para SmolVLM (el recorte medido: 12 %-55 % de la altura de la persona)
            torsos_smol.append(img[y1 + int(0.12 * ph):y1 + int(0.55 * ph), x1:x2].copy())
            res["jugadores"].append(jug)
        _jurado(res["jugadores"], torsos_vision, torsos_smol)
        _fusionar_vision(res["jugadores"], torsos_vision)
        # un mismo numero fisico no puede pertenecer a dos personas solapadas de la foto
        js = [j for j in res["jugadores"] if j.get("_num_xy")]
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                ja, jb = js[a], js[b]
                if not (ja["dorsal"] and jb["dorsal"] and ja["dorsal"] == jb["dorsal"]):
                    continue
                (xa, ya, ha), (xb, yb, hb) = ja["_num_xy"], jb["_num_xy"]
                if abs(xa - xb) + abs(ya - yb) < 0.6 * max(ha, hb):
                    perdedor = ja if ja["_num_centrado"] < jb["_num_centrado"] else jb
                    perdedor["dorsal"], perdedor["conf_dorsal"] = None, 0.0
                    perdedor.pop("firma_numero", None)
        res["ok"] = True
        del gray_full, small
    except Exception as e:  # una foto rara no debe tumbar el lote
        logger.exception("Error analizando %s: %s", nombre, e)
    finally:
        del img
        gc.collect()
    return res
