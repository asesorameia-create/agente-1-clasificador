import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
os.environ.setdefault("MODELS_DIR", os.path.join(os.path.dirname(AQUI), "models"))
import numpy as np, cv2
"""Prueba de humo SIN fotos reales: partido inventado con las trampas acordadas."""
import motor_biometrico as mb, identidad
rng = np.random.default_rng(1)
def color(bgr):
    t = np.zeros((40, 40, 3), np.uint8); t[:] = bgr
    t = (t.astype(int) + rng.integers(-8, 8, t.shape)).clip(0, 255).astype(np.uint8)
    return mb._histo_camiseta(t)
COMUN = rng.normal(size=128); COMUN /= np.linalg.norm(COMUN)   # las caras reales comparten estructura
def cara(base=None):
    v = (COMUN * 0.55 + rng.normal(size=128) / 11.3 * 0.84) if base is None else base + rng.normal(scale=0.35, size=128) * np.linalg.norm(base) / 11
    return (v / np.linalg.norm(v)).astype(np.float32)
ROJO, BLANCO, VERDE = (40, 40, 220), (235, 235, 235), (60, 160, 60)
fA, fB, fS = cara(), cara(), cara()
grupo = [cara() for _ in range(5)]
def J(c, face=None, d=None, x=0.3, alt=0.6, q=0.8):
    return {"caja": (x, 0.2, x + 0.2, 0.2 + alt), "altura": alt, "dorsal": d, "conf_dorsal": 0.99 if d else 0.0,
            "palabras": [], "huella": face, "color": color(c), "mini": b"x", "mini_cara": face is not None,
            "calidad_cara": q if face is not None else 0.0, "alto_dorsal": (0.06 if face is not None else 0.12) if d else 0.0}
fotos = [
    [J(ROJO, None, "8")],                     # 0 A de espaldas
    [J(ROJO, cara(fA), "8")],                 # 1 A de frente con numero
    [J(ROJO, cara(fA))],                      # 2 A sin numero
    [J(BLANCO, fB, "8")],                     # 3 B (otro equipo) #8
    [J(BLANCO, None, "8")],                   # 4 B de espaldas
    [J(ROJO, cara(fA), x=0.0, alt=0.45, q=0.6)] + [J(ROJO, g, x=0.15 * (k + 1), alt=0.45, q=0.6) for k, g in enumerate(grupo)],  # 5 grupo
    [J(VERDE, fS, x=0.1, alt=0.3, q=0.7), J(ROJO, cara(grupo[0]), alt=0.6)],   # 6 espectador + jugador del grupo
    [J(BLANCO, cara(fB), x=0.5)],             # 7 B de frente sin numero
    [J((150, 190, 215), cara(), "5", x=0.6)],  # 8 caso real: beige con logo "R5" leido como 5
    [J((230, 190, 120), None, "5", x=0.8)],    # 9 caso real: celeste de espaldas, "15" tapado leido como 5
]
analisis = [{"key": f"f{i}.jpg", "hora": None, "firma": None, "jugadores": js, "ok": True} for i, js in enumerate(fotos)]
r = identidad.clasificar_lote(analisis, "")
print("RESUMEN", r["resumen"])
print("parecido_cara de las caras inventadas:", sorted(round(j.get("parecido_cara", -1), 2) for a in analisis for j in a["jugadores"] if j["huella"] is not None)[:4], "...")
for c, ks in sorted(r["carpetas"].items()):
    print(f"  {c:28s} grupo={r['detalle'].get(c, {}).get('grupo', '-'):10s} {ks}")
print("DUDAS", [(d['tipo'], d.get('carpeta')) for d in r["dudas"]])
print("dist rojo-blanco", round(identidad._dist_color(fotos[0][0]["color"], fotos[3][0]["color"]), 2),
      "rojo-verde", round(identidad._dist_color(fotos[0][0]["color"], fotos[6][0]["color"]), 2))

c5 = [c for c, ks in r["carpetas"].items() if "f8.jpg" in ks and "f9.jpg" in ks]
print("CASO R5/15 (no deben mezclarse):", "MAL, MEZCLADOS en " + str(c5) if c5 else "OK, separados")

grupo = [c for c, ks in r["carpetas"].items() if "f5.jpg" in ks]
a_ = [c for c, ks in r["carpetas"].items() if {"f0.jpg", "f1.jpg", "f2.jpg"} <= set(ks)]
assert not c5, "REGRESION: dos personas con un 5 falso quedaron mezcladas"
assert len(grupo) >= 6, f"REGRESION: en la foto de grupo faltan jugadores ({len(grupo)})"
assert a_, "REGRESION: el mismo jugador (#8 rojo) quedo partido en varias carpetas"
assert not [c for c, ks in r["carpetas"].items() if "f6.jpg" in ks and len(ks) == 1], "REGRESION: el espectador recibio carpeta"
print("TEST IDENTIDAD: OK")
