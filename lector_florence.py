"""
Segundo lector de camisetas: Florence-2 (Microsoft, licencia MIT), local y gratis por foto.

Medido con las 20 fotos del cliente (25 numeros visibles): CRNN solo 28 %, Florence-2 solo 52 %,
ambos con votacion 56 % y 0 numeros equivocados aceptados. Ademas lee nombres de equipo
(Caracas, La Guaira, Trujillo, Carabobo).

- Corre dentro del servidor (CPU); ~10 s por torso, asi que solo se usa en los protagonistas sin
  lectura segura del lector rapido.
- Version nativa de 'transformers' (florence-community/Florence-2-base): no ejecuta codigo descargado.
- Si no esta instalado o falla, el agente sigue con el lector rapido.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("CastilloEnfoca.florence")

REPO = os.getenv("FLORENCE_REPO", "florence-community/Florence-2-base")
REVISION = os.getenv("FLORENCE_REVISION", "00921df66db728a9ceb750f5eca43e5c203a2051")   # la version probada
_lock = threading.Lock()
_modelo = None
_fallo = False


def disponible() -> bool:
    if _fallo or os.getenv("LECTOR_FLORENCE", "1") == "0":
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _cargar():
    global _modelo, _fallo
    if _modelo is not None or _fallo:
        return _modelo
    try:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        torch.set_num_threads(max(1, int(os.getenv("FLORENCE_HILOS", str(os.cpu_count() or 1)))))
        proc = AutoProcessor.from_pretrained(REPO, revision=REVISION)
        model = Florence2ForConditionalGeneration.from_pretrained(REPO, revision=REVISION, torch_dtype=torch.float32).eval()
        _modelo = (proc, model)
        logger.info("Florence-2 cargado (%s)", REPO)
    except Exception as e:
        logger.error("No se pudo cargar Florence-2 (%s); se sigue con el lector rapido", e)
        _fallo = True
    return _modelo


def leer(bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    """OCR del torso. Devuelve {"numeros": [...], "texto": str} o None si no se pudo."""
    if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 16 or not disponible():
        return None
    with _lock:   # un solo modelo compartido: las llamadas van en fila
        pm = _cargar()
        if pm is None:
            return None
        proc, model = pm
        try:
            import torch
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inp = proc(text="<OCR>", images=img, return_tensors="pt")
            with torch.inference_mode():
                out = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                     max_new_tokens=24, num_beams=1, do_sample=False)
            texto = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        except Exception as e:
            logger.warning("Florence-2 fallo en un torso: %s", e)
            return None
    numeros = re.findall(r"(?<!\d)([1-9]\d?)(?!\d)", texto)
    return {"numeros": numeros, "texto": texto}
