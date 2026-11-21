"""
Lector de camisetas con Claude (vision): dorsal y nombre del equipo, incluso con tipografias raras.

- Privacidad: solo se envia el TORSO de cada jugador (del cuello hacia abajo). Las caras nunca salen
  del servidor; la huella facial sigue siendo 100% local (SFace).
- Opcional: si no hay ANTHROPIC_API_KEY o la API falla, el agente sigue con el lector local (CRNN).
- Costo aproximado: una llamada por foto con hasta 5 torsos pequenos (~0,01 USD por foto).
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("CastilloEnfoca.vision")

MODELO = os.getenv("VISION_MODELO", "claude-opus-5-5")
MAX_TORSOS = 5
ALTO_TORSO = 384     # px: suficiente para leer un numero, barato en tokens

_cliente = None


def disponible() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("VISION_DESACTIVADA", "0") != "1"


def _client():
    global _cliente
    if _cliente is None:
        import anthropic
        _cliente = anthropic.Anthropic(timeout=60.0, max_retries=2)
    return _cliente


ESQUEMA = {
    "type": "object",
    "properties": {
        "jugadores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "imagen": {"type": "integer"},
                    "numero": {"type": ["integer", "null"]},
                    "numero_seguro": {"type": "boolean"},
                    "camiseta_puesta": {"type": "boolean"},
                    "equipo": {"type": ["string", "null"]},
                },
                "required": ["imagen", "numero", "numero_seguro", "camiseta_puesta", "equipo"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["jugadores"],
    "additionalProperties": False,
}

INSTRUCCION = (
    "Each image is the torso of one player in an amateur football (soccer) match photo. For every image, read:\n"
    "- numero: the shirt or shorts number (1-99) printed on THIS player's kit, or null if no number is legible. "
    "Numbers may be stylized, outlined, two-tone, partly folded or seen at an angle; read them like a careful "
    "human would. If you can only guess, return null.\n"
    "- numero_seguro: true only if you can clearly read every digit.\n"
    "- camiseta_puesta: false if the numbered shirt is held in the hands or not worn by the person (e.g. a "
    "jersey presentation); true otherwise.\n"
    "- equipo: the team, club, city or state name printed on the shirt (e.g. 'Trujillo', 'Carabobo', "
    "'La Guaira'), or null. Ignore sponsors, brands and the word 'Venezuela'.\n"
    "Return one entry per image, using the image index given before it."
)


def _jpeg_b64(img: np.ndarray) -> Optional[str]:
    if img is None or img.size == 0 or min(img.shape[:2]) < 12:
        return None
    h, w = img.shape[:2]
    s = min(1.0, ALTO_TORSO / h)
    if s < 1.0:
        img = cv2.resize(img, (max(12, int(w * s)), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.standard_b64encode(buf.tobytes()).decode("ascii") if ok else None


def leer_torsos(torsos: List[np.ndarray]) -> Dict[int, Dict[str, Any]]:
    """Devuelve {indice: {"numero", "numero_seguro", "camiseta_puesta", "equipo"}}; {} si no se pudo."""
    if not disponible() or not torsos:
        return {}
    contenido: List[Dict[str, Any]] = []
    indices = []
    for i, t in enumerate(torsos[:MAX_TORSOS]):
        b64 = _jpeg_b64(t)
        if not b64:
            continue
        contenido.append({"type": "text", "text": f"Image {i}:"})
        contenido.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
        indices.append(i)
    if not indices:
        return {}
    contenido.append({"type": "text", "text": INSTRUCCION})
    try:
        import anthropic
        resp = _client().beta.messages.create(
            model=MODELO,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": ESQUEMA}},
            messages=[{"role": "user", "content": contenido}],
        )
        if resp.stop_reason == "refusal":
            logger.warning("Lectura de camisetas rechazada por el modelo; se usa el lector local")
            return {}
        texto = next((b.text for b in resp.content if b.type == "text"), "")
        import json
        datos = json.loads(texto)
    except anthropic.AuthenticationError:
        logger.error("ANTHROPIC_API_KEY invalida: lectura de camisetas con Claude desactivada")
        os.environ["VISION_DESACTIVADA"] = "1"
        return {}
    except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
        logger.warning("Lectura de camisetas con Claude no disponible (%s); se usa el lector local", e)
        return {}
    except Exception as e:   # JSON inesperado u otro fallo: nunca tumbar el analisis
        logger.warning("Respuesta de vision no utilizable: %s", e)
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for j in datos.get("jugadores", []):
        i = j.get("imagen")
        if isinstance(i, int) and i in indices:
            n = j.get("numero")
            if not (isinstance(n, int) and 1 <= n <= 99):
                n = None
            out[i] = {"numero": n, "numero_seguro": bool(j.get("numero_seguro")) and n is not None,
                      "camiseta_puesta": bool(j.get("camiseta_puesta", True)), "equipo": j.get("equipo") or None}
    return out
