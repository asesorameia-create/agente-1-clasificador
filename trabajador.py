"""
Trabajador aislado para el analisis de fotos (la parte pesada: modelos de IA).

Por que existe (medido en la prueba en vivo del 21 nov 2026): Render se cayo varias veces con
"Ran out of memory (used over 4GB)" mientras analizaba fotos. Si el analisis corre dentro del mismo
proceso que la pagina, un pico de memoria tumba TODO (pagina, subidas, descargas).

Aqui el analisis corre en un proceso aparte que el servidor vigila:
- si pasa de TRABAJADOR_LIMITE_MB, se cuelga o muere -> se reinicia y la foto se reintenta en modo
  liviano (sin Florence-2). La pagina nunca se cae y ninguna foto se pierde.
- se renueva cada TRABAJADOR_RECICLAR fotos para soltar la memoria acumulada.
- si aun asi hubiera un pico, el sistema mata al proceso mas grande: el trabajador, no el servidor.

Protocolo: mensajes pickle con prefijo de longitud por stdin/stdout del proceso hijo.
"""
from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("CastilloEnfoca.trabajador")

LIMITE_MB = int(os.getenv("TRABAJADOR_LIMITE_MB", "3000"))
RECICLAR = int(os.getenv("TRABAJADOR_RECICLAR", "15"))
TIEMPO_MAX = int(os.getenv("TRABAJADOR_TIEMPO_MAX", "240"))
CONTENEDOR_LIMITE_MB = int(os.getenv("CONTENEDOR_LIMITE_MB", "3600"))   # Render corta a 4096 MB


def memoria_contenedor_mb() -> int:
    """Lo que Render mide de verdad: la memoria de TODO el contenedor (cgroup), no solo de un proceso."""
    for ruta in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(ruta) as f:
                return int(f.read().strip()) // 2**20
        except Exception:
            continue
    return 0


def _escribir(f, obj) -> None:
    b = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
    f.flush()


def _leer(f):
    cab = f.read(8)
    if len(cab) < 8:
        raise EOFError("trabajador cerrado")
    n = struct.unpack("<Q", cab)[0]
    datos = f.read(n)
    if len(datos) < n:
        raise EOFError("mensaje incompleto")
    return pickle.loads(datos)


# ------------------------------------------------------------------ lado hijo
def _bucle() -> None:
    entrada = os.fdopen(os.dup(0), "rb")
    salida = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                      # cualquier print de librerias va a stderr, no rompe el canal
    import motor_biometrico as motor
    florence_original = os.environ.get("LECTOR_FLORENCE")
    while True:
        try:
            msg = _leer(entrada)
        except EOFError:
            break
        if msg is None:
            break
        datos, nombre, opciones = msg
        if opciones.get("liviano"):
            os.environ["LECTOR_FLORENCE"] = "0"
            os.environ["LECTOR_SMOL"] = "0"
        elif florence_original is None:
            os.environ.pop("LECTOR_FLORENCE", None)
        else:
            os.environ["LECTOR_FLORENCE"] = florence_original
        try:
            r = motor.analizar_foto(datos, nombre)
            for j in r.get("jugadores", []):
                j.pop("torso", None)
        except Exception as e:          # noqa: BLE001
            r = {"nombre": nombre, "hora": None, "jugadores": [], "firma": None, "ok": False, "error": str(e)}
        _escribir(salida, r)


# ------------------------------------------------------------------ lado servidor
class Trabajador:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.hechas = 0
        self.lock = threading.Lock()
        self.reinicios = 0

    def _arrancar(self) -> None:
        self.proc = subprocess.Popen([sys.executable, "-c", "import trabajador; trabajador._bucle()"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     cwd=os.path.dirname(os.path.abspath(__file__)))
        self.hechas = 0
        logger.info("Trabajador de analisis iniciado (pid %s)", self.proc.pid)

    def _parar(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.kill()
            p.wait(timeout=10)
        except Exception:
            pass

    def rss_mb(self) -> int:
        if not self.proc:
            return 0
        try:
            with open(f"/proc/{self.proc.pid}/status") as f:
                for linea in f:
                    if linea.startswith("VmRSS:"):
                        return int(linea.split()[1]) // 1024
        except Exception:
            pass
        return 0

    def analizar(self, datos: bytes, nombre: str) -> Dict[str, Any]:
        with self.lock:
            for intento, opciones in enumerate(({}, {"liviano": True})):
                if self.proc is None or self.proc.poll() is not None or self.hechas >= RECICLAR:
                    self._parar()
                    self._arrancar()
                resultado: Dict[str, Any] = {}
                error: list = []

                def esperar():
                    try:
                        resultado["r"] = _leer(self.proc.stdout)
                    except Exception as e:   # noqa: BLE001
                        error.append(e)

                try:
                    _escribir(self.proc.stdin, (datos, nombre, opciones))
                except Exception as e:       # noqa: BLE001
                    motivo = f"no acepto la foto ({e})"
                else:
                    hilo = threading.Thread(target=esperar, daemon=True)
                    hilo.start()
                    t0, motivo, pico = time.time(), None, 0
                    while hilo.is_alive():
                        hilo.join(0.2)
                        m = self.rss_mb()
                        pico = max(pico, m)
                        if m > LIMITE_MB:
                            motivo = f"memoria {m} MB > {LIMITE_MB} MB"
                            break
                        mc = memoria_contenedor_mb()
                        if mc > CONTENEDOR_LIMITE_MB:
                            motivo = f"memoria del servidor {mc} MB > {CONTENEDOR_LIMITE_MB} MB"
                            break
                        if time.time() - t0 > TIEMPO_MAX:
                            motivo = f"tardo mas de {TIEMPO_MAX} s"
                            break
                    if motivo is None and "r" in resultado:
                        self.hechas += 1
                        r = resultado["r"]
                        r["memoria_trabajador_mb"] = pico
                        if intento:
                            r["modo_liviano"] = True
                        return r
                    if motivo is None:
                        motivo = f"el trabajador termino ({error[0] if error else 'sin respuesta'})"
                self.reinicios += 1
                logger.warning("Foto %s: %s. Se reinicia el trabajador%s", nombre, motivo,
                               " y se reintenta en modo liviano" if intento == 0 else "")
                self._parar()
            return {"nombre": nombre, "hora": None, "jugadores": [], "firma": None, "ok": False}


TRABAJADOR = Trabajador()
