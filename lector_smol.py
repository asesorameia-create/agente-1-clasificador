"""
Tercer lector de dorsales: SmolVLM-256M-Instruct (Hugging Face, licencia Apache 2.0, uso comercial).

A diferencia de los lectores de texto (CRNN, Florence-2), a este modelo se le PREGUNTA:
  1) "¿Hay un numero grande en esta camiseta?"  -> probabilidad de "si"
  2) "¿Que numero tiene la camiseta?"            -> numero + seguridad de cada cifra

Medido (21 nov 2026) con recortes reales del cliente:
- Numeros grandes y decorados que CRNN, Florence-2 y PP-OCR no leian (4 Caracas, 6 Apure, 8 La Guaira,
  3 / 6 / 10 Trujillo): 6 de 6 correctos.
- Personas sin numero (publico, suplentes con peto): 0 de 18 aceptados con el filtro de abajo
  (sin filtro inventaba "1", "10", "11"). El mayor "si" de una persona sin numero fue 0,71.
- ~3,5 s por pregunta en CPU sin dividir la imagen.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger("CastilloEnfoca.smol")

REPO = os.getenv("SMOL_REPO", "HuggingFaceTB/SmolVLM-256M-Instruct")
REVISION = os.getenv("SMOL_REVISION", "d039ab0d4b9311b617e8ffec1d06c8464364aeb3")   # la version probada
P_HAY_NUMERO = 0.85     # bajo esto no se pregunta el numero (personas sin numero: maximo medido 0,71)
P_CIFRAS = 0.90         # seguridad minima de cada cifra leida
Q_HAY = "Is there a large number printed on this shirt? Answer yes or no."
Q_NUM = "What is the jersey number of the player? Answer only with the number, or none."

_lock = threading.Lock()
_modelo = None
_fallo = False


def disponible() -> bool:
    if _fallo or os.getenv("LECTOR_SMOL", "1") == "0":
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
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(REPO, revision=REVISION)
        model = AutoModelForImageTextToText.from_pretrained(REPO, revision=REVISION, torch_dtype=torch.float32).eval()
        tok = proc.tokenizer
        si = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes", " yes")]
        no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no", " no")]
        _modelo = (proc, model, si, no)
        logger.info("SmolVLM cargado (%s)", REPO)
    except Exception as e:
        logger.error("No se pudo cargar SmolVLM (%s); se sigue sin el", e)
        _fallo = True
    return _modelo


def _preguntar(proc, model, img, q: str, n: int):
    import torch
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    inp = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True), images=[img],
               return_tensors="pt", do_image_splitting=False)
    with torch.inference_mode():
        o = model.generate(**inp, max_new_tokens=n, do_sample=False, output_scores=True, return_dict_in_generate=True)
    gen = o.sequences[0, inp["input_ids"].shape[1]:]
    probs = [float(torch.softmax(s[0], -1)[t]) for s, t in zip(o.scores, gen)]
    return proc.tokenizer.decode(gen, skip_special_tokens=True).strip(), probs, o.scores[0][0]


def leer(bgr: np.ndarray) -> Optional[Dict[str, Any]]:
    """Recorte del pecho/espalda (sin cabeza). Devuelve {"numero", "conf", "si"}; numero None si no hay."""
    if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 24 or not disponible():
        return None
    with _lock:   # un solo modelo compartido: las llamadas van en fila
        pm = _cargar()
        if pm is None:
            return None
        proc, model, ids_si, ids_no = pm
        try:
            import torch
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            img.thumbnail((512, 512))
            _, _, s0 = _preguntar(proc, model, img, Q_HAY, 1)
            sm = torch.softmax(s0, -1)
            ps, pn = float(sm[ids_si].sum()), float(sm[ids_no].sum())
            si = ps / (ps + pn + 1e-9)
            if si < P_HAY_NUMERO:
                return {"numero": None, "conf": 0.0, "si": si}
            txt, probs, _ = _preguntar(proc, model, img, Q_NUM, 4)
        except Exception as e:
            logger.warning("SmolVLM fallo en un recorte: %s", e)
            return None
    m = re.match(r"\s*([1-9]\d?)(?!\d)", txt)
    if not m:
        return {"numero": None, "conf": 0.0, "si": si}
    conf = min(probs[:len(m.group(1))]) if probs else 0.0
    return {"numero": m.group(1) if conf >= P_CIFRAS else None, "conf": conf, "si": si}
