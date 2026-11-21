import os, sys
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
os.environ.setdefault("MODELS_DIR", os.path.join(os.path.dirname(AQUI), "models"))
import numpy as np, cv2
import motor_biometrico as mb, lector_smol, lector_florence
lector_smol.disponible = lambda: True; lector_florence.disponible = lambda: True
casos = [  # (rapido, conf, smol(num,si,conf), florence numeros, esperado)
    ("8", 0.90, ("8", 0.95, 0.99), None, "8"),          # coinciden: no hace falta juez
    ("6", 0.90, ("8", 0.95, 0.99), ["8"], "8"),         # discrepan: Florence desempata a favor de SmolVLM
    ("6", 0.90, ("8", 0.95, 0.99), ["6"], "6"),         # discrepan: Florence desempata a favor del rapido
    ("6", 0.90, ("8", 0.95, 0.99), ["3"], None),        # tres distintos: nadie gana, no se inventa
    (None, 0, ("10", 0.99, 0.99), [], "10"),            # solo SmolVLM muy seguro y Florence no contradice
    (None, 0, ("10", 0.99, 0.99), ["7"], None),         # SmolVLM vs Florence distintos: no se nombra
    (None, 0, ("1", 0.86, 0.91), [], None),             # SmolVLM poco seguro solo: sugerencia
    (None, 0, None, ["5"], None),                       # solo Florence: no basta
    ("12", 0.99, ("3", 0.95, 0.99), None, "12"),        # rapido seguro: ni se pregunta
    ("5", 0.99, None, [], None),                        # caso real "R5": una cifra con un solo lector NO basta
    ("5", 0.99, ("5", 0.95, 0.99), None, "5"),          # una cifra con dos lectores de acuerdo: si
    (None, 0, ("5", 0.99, 0.99), [], None),             # una cifra solo SmolVLM: no basta
]
llamadas = {"f": 0}
ok = 0
for rap, conf, sm, fl, esperado in casos:
    j = {"altura": 0.6, "dorsal": rap, "conf_dorsal": conf}
    lector_smol.leer = lambda b, sm=sm: ({"numero": sm[0], "si": sm[1], "conf": sm[2]} if sm else {"numero": None, "si": 0.1, "conf": 0})
    def fl_leer(b, fl=fl):
        llamadas["f"] += 1
        return {"numeros": fl or [], "texto": " ".join(fl or [])}
    lector_florence.leer = fl_leer
    t = np.zeros((40, 40, 3), np.uint8)
    mb._jurado([j], [t], [t])
    bien = j.get("dorsal") == esperado
    ok += bien
    print(("OK " if bien else "MAL"), f"rapido={rap} smol={sm and sm[0]} florence={fl} -> {j.get('dorsal')} (esperado {esperado})")
print(f"{ok}/{len(casos)} casos bien; Florence se llamo {llamadas['f']} veces de {len(casos)}")

assert ok == len(casos), "REGRESION en el jurado de lectores"
import motor_biometrico as mb2
assert len(mb2._sin_duplicados([(100, 100, 300, 600, .9), (110, 120, 290, 560, .8)])) == 1, "REGRESION: recuadro duplicado"
assert len(mb2._sin_duplicados([(100, 100, 300, 600, .9), (250, 110, 450, 610, .8)])) == 2, "REGRESION: dos jugadores juntos"
print("TEST JURADO + DUPLICADOS: OK")
