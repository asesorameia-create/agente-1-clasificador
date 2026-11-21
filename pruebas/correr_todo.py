"""Examen automatico antes de cada publicacion: cada fallo real que tuvo el agente es una prueba aqui.
Uso: python pruebas/correr_todo.py   (si algo falla, NO publicar)"""
import subprocess, sys, os
AQUI = os.path.dirname(os.path.abspath(__file__))
fallos = 0
for t in ("test_identidad.py", "test_jurado.py"):
    r = subprocess.run([sys.executable, os.path.join(AQUI, t)], capture_output=True, text=True)
    ultima = (r.stdout.strip().splitlines() or [""])[-1]
    print(("OK   " if r.returncode == 0 else "FALLA"), t, "|", ultima if r.returncode == 0 else r.stderr.strip().splitlines()[-1])
    fallos += r.returncode != 0
sys.exit(1 if fallos else 0)
