"""
Fusion de identidades de CastilloEnfoca: convierte las huellas de cada foto en
UNA carpeta unica por jugador.

Cada aparicion de un jugador en una foto es una "instancia" con:
  huella facial SFace (128D), dorsal leido (si se vio), equipo (color + nombre en camiseta),
  posicion en la foto y hora de captura.

Algoritmo (aglomerativo con candados, de la evidencia mas fuerte a la mas debil):
  1. Mismo equipo + mismo dorsal bien leido          -> misma persona.
  2. Rostros muy parecidos (coseno SFace alto)       -> misma persona.
  3. Rafagas: foto vecina casi identica, misma zona  -> misma persona.
  Candados: nunca se unen dos personas de la MISMA foto, ni dos dorsales
  distintos bien leidos, ni equipos distintos.
Resultado: Dorsal_06_Trujillo, Dorsal_08_La_Guaira, Jugador_01_Carabobo (rostro recurrente sin numero),
y 00_Fotos_Generales_Equipo para lo que no tiene un protagonista identificable.
"""
from __future__ import annotations

import difflib
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from motor_biometrico import _dist_color, _nombre_color, dist_firma_numero

GENERAL = "00_Fotos_Generales_Equipo"

# Umbrales SFace (coseno). Calibracion publicada sobre LFW: FAR 1e-4 -> 0.421, FAR 1e-5 -> 0.472.
SIM_UNIR_FUERTE = 0.50      # dos caras sueltas: misma persona
SIM_UNIR_A_DORSAL = 0.43    # cara sin numero contra el conjunto de caras de un dorsal conocido
DORSAL_FIRME = 0.90         # lectura de dorsal que actua como candado
DIST_EQUIPO = 0.30          # mismo dorsal sin firma de numero: camisetas casi iguales
DIST_FIRMA_NUMERO = 12.0    # Delta E medio entre colores de cifra/fondo de dos dorsales iguales
DIST_UNION_COLOR = 0.70     # por encima, dos grupos nunca se unen (ej. camiseta negra vs blanca)
DIST_HEREDAR_NOMBRE = 0.15  # camiseta casi identica a la de un jugador cuyo equipo se leyo
RAFAGA_SEG = 2.0

# Medidor de certeza: solo lo que supera estos umbrales entra solo a la carpeta; lo demas se pregunta.
SIM_SEGURA = 0.56           # coseno SFace "misma persona sin duda" (margen sobre FAR 1e-5 = 0.472 en LFW)
DORSAL_SEGURO = 0.97        # lectura de dorsal practicamente sin ambiguedad
RAFAGA_SEGURA = 0.88        # foto vecina casi identica
NUMERO_GRANDE = 0.10        # numero >10% de la altura de la persona = numero de espalda


def _numero_sospechoso(j: Dict[str, Any]) -> bool:
    """Numero grande (de espalda) y a la vez cara de frente: tipico de una camiseta sostenida en la mano
    (entrega, celebracion) o de una lectura que no pertenece a esa persona."""
    return bool(j.get("dorsal")) and j.get("alto_dorsal", 0) >= NUMERO_GRANDE and j.get("huella") is not None \
        and j.get("calidad_cara", 0) >= 0.6

# Equipos de los Juegos Deportivos Nacionales / estados y ciudades de Venezuela,
# para corregir la ortografia de lo que el OCR lee en la camiseta.
NOMBRES_EQUIPO = [
    "Amazonas", "Anzoategui", "Apure", "Aragua", "Barinas", "Bolivar", "Carabobo", "Cojedes",
    "Delta Amacuro", "Falcon", "Guarico", "Lara", "Merida", "Miranda", "Monagas", "Nueva Esparta",
    "Portuguesa", "Sucre", "Tachira", "Trujillo", "La Guaira", "Vargas", "Yaracuy", "Zulia",
    "Caracas", "Distrito Capital", "Maracaibo", "Valencia", "Barquisimeto", "Maracay", "Maturin",
    "Barcelona", "Cumana", "Guayana", "Puerto Cabello", "San Cristobal",
]
PALABRAS_IGNORADAS = {
    "venezuela", "deporte", "deportes", "mascerca", "cerca", "the", "fundadeporte", "encantta",
    "nike", "adidas", "puma", "umbro", "joma", "kelme", "sport", "sports", "club", "futbol",
}


def _sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _canon(p: str) -> str:
    return re.sub(r"[^a-z]", "", _sin_acentos(p).lower())


def limpiar_nombre(s: str) -> str:
    s = _sin_acentos(s)
    s = re.sub(r"[^\w\- ]", "", s, flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", s)[:40]


def _num_archivo(nombre: str) -> Tuple[str, Optional[int]]:
    base = os.path.splitext(os.path.basename(nombre))[0]
    mm = re.search(r"(\d{3,})(?!.*\d)", base)
    if not mm:
        return base, None
    return base[:mm.start()], int(mm.group(1))


def parsear_nomina(texto: str) -> List[Tuple[Optional[str], str, str]]:
    """Lineas como '8 Juan Perez', 'Juan Perez 8' o 'Trujillo 8 Juan Perez' -> (equipo, dorsal, nombre)."""
    out = []
    for linea in (texto or "").splitlines():
        linea = linea.strip().strip("-•*").strip()
        mm = re.search(r"(?<!\d)(\d{1,2})(?!\d)", linea)
        if not linea or not mm:
            continue
        antes = linea[:mm.start()].strip(" -=:#,.")
        despues = linea[mm.end():].strip(" -=:#,.")
        num = str(int(mm.group(1)))
        if despues:
            out.append((antes or None, num, despues))
        elif antes:
            out.append((None, num, antes))
    return out


def _nombre_equipo(palabras: List[str]) -> Optional[str]:
    """Nombre de equipo a partir de lecturas OCR ruidosas ('triuillo', 'truillo' -> 'Trujillo')."""
    votos: Counter = Counter()
    libres: List[str] = []
    for p in palabras:
        c = _canon(p)
        if len(c) < 4 or c in PALABRAS_IGNORADAS:
            continue
        if any(difflib.SequenceMatcher(None, c, ig).ratio() > 0.8 for ig in PALABRAS_IGNORADAS):
            continue
        mejor, r = None, 0.0
        for n in NOMBRES_EQUIPO:
            cn = _canon(n)
            rr = difflib.SequenceMatcher(None, c, cn).ratio()
            if cn in c or (len(c) >= 5 and c in cn):
                rr = max(rr, 0.85)
            if rr > r:
                mejor, r = n, rr
        if mejor and r >= 0.72:
            votos[mejor] += 1
        else:
            libres.append(c)
    if votos:
        return votos.most_common(1)[0][0]
    # palabra desconocida que se repite (ej. nombre de un club): la forma mas central
    if len(libres) >= 2:
        best, bs = None, -1.0
        for a in libres:
            s = sum(difflib.SequenceMatcher(None, a, b).ratio() for b in libres)
            if s > bs:
                best, bs = a, s
        apoyo = sum(1 for b in libres if difflib.SequenceMatcher(None, best, b).ratio() >= 0.7)
        if apoyo >= 2:
            return best.capitalize()
    return None


class _Grupos:
    """Union-find con candados (fotos, dorsales firmes, color de camiseta)."""

    def __init__(self, inst):
        self.padre = list(range(len(inst)))
        self.fotos = [{i[0]} for i in inst]
        self.dorsales = [Counter({i[2]["dorsal"]: i[2]["conf_dorsal"]}) if i[2]["dorsal"] else Counter() for i in inst]
        self.colores = [[i[2]["color"]] if i[2]["color"] is not None else [] for i in inst]
        self.huellas = [[i[2]["huella"]] if i[2]["huella"] is not None else [] for i in inst]

    def color(self, r) -> Optional[np.ndarray]:
        cs = self.colores[r]
        return np.median(np.array(cs), axis=0) if cs else None

    def raiz(self, a):
        while self.padre[a] != a:
            self.padre[a] = self.padre[self.padre[a]]
            a = self.padre[a]
        return a

    def dorsal_firme(self, r) -> Optional[str]:
        d = self.dorsales[r]
        if not d:
            return None
        k, v = d.most_common(1)[0]
        return k if v >= DORSAL_FIRME else None

    def compatibles(self, a, b) -> bool:
        ra, rb = self.raiz(a), self.raiz(b)
        if ra == rb:
            return False
        if self.fotos[ra] & self.fotos[rb]:
            return False  # dos personas de la misma foto
        ca, cb = self.color(ra), self.color(rb)
        if ca is not None and cb is not None and _dist_color(ca, cb) > DIST_UNION_COLOR:
            return False  # camisetas claramente distintas: otro equipo
        da, db = self.dorsal_firme(ra), self.dorsal_firme(rb)
        if da and db and da != db:
            return False
        return True

    def unir(self, a, b) -> bool:
        if not self.compatibles(a, b):
            return False
        ra, rb = self.raiz(a), self.raiz(b)
        if len(self.fotos[ra]) < len(self.fotos[rb]):
            ra, rb = rb, ra
        self.padre[rb] = ra
        self.fotos[ra] |= self.fotos[rb]
        self.dorsales[ra] += self.dorsales[rb]
        self.colores[ra] += self.colores[rb]
        self.huellas[ra] += self.huellas[rb]
        return True

    def sim(self, a, b) -> float:
        """Similitud entre los conjuntos de huellas de dos grupos (media de los 2 mejores pares)."""
        ha, hb = self.huellas[self.raiz(a)], self.huellas[self.raiz(b)]
        if not ha or not hb:
            return 0.0
        s = np.array(ha) @ np.array(hb).T
        top = np.sort(s.ravel())[::-1][:2]
        return float(top.mean())


def clasificar_lote(analisis: List[Dict[str, Any]], nomina: str = "") -> Dict[str, Any]:
    """
    analisis: resultados de motor_biometrico.analizar_foto() con clave 'key' (ruta en R2).
    Devuelve {"carpetas": {carpeta: [keys]}, "miniaturas": {carpeta: bytes}, "resumen": {...}, "detalle": {...}}
    """
    inst = [(fi, ji, j) for fi, a in enumerate(analisis) for ji, j in enumerate(a.get("jugadores", []))]
    G = _Grupos(inst)

    # ---------------- 1. evidencia fuerte: mismo dorsal + camiseta parecida
    con_dorsal = sorted([i for i, x in enumerate(inst) if x[2]["dorsal"]], key=lambda i: -inst[i][2]["conf_dorsal"])
    for p, a in enumerate(con_dorsal):
        for b in con_dorsal[p + 1:]:
            ja, jb = inst[a][2], inst[b][2]
            if ja["dorsal"] != jb["dorsal"]:
                continue
            dn = dist_firma_numero(ja.get("firma_numero"), jb.get("firma_numero"))
            # mismo numero con los mismos colores de cifra/fondo (misma camiseta), o camisetas casi identicas
            if dn <= DIST_FIRMA_NUMERO or (dn >= 999 and _dist_color(ja["color"], jb["color"]) <= DIST_EQUIPO):
                G.unir(a, b)

    # ---------------- 2. biometria facial: pares de mayor a menor similitud
    con_cara = [i for i, x in enumerate(inst) if x[2]["huella"] is not None]
    if con_cara:
        Hm = np.array([inst[i][2]["huella"] for i in con_cara])
        S = Hm @ Hm.T
        pares = [(float(S[a, b]), con_cara[a], con_cara[b]) for a in range(len(con_cara))
                 for b in range(a + 1, len(con_cara)) if S[a, b] >= SIM_UNIR_A_DORSAL]
        pares.sort(reverse=True)
        for s, a, b in pares:
            ra, rb = G.raiz(a), G.raiz(b)
            if ra == rb:
                continue
            umbral = SIM_UNIR_A_DORSAL if (G.dorsales[ra] or G.dorsales[rb]) else SIM_UNIR_FUERTE
            # se confirma contra el conjunto de huellas del grupo, no un solo par
            if s >= umbral and G.sim(a, b) >= umbral - 0.03:
                G.unir(a, b)

    # ---------------- 3. rafagas: misma escena en fotos seguidas
    orden = sorted(range(len(analisis)), key=lambda i: (
        analisis[i].get("hora") is None, analisis[i].get("hora") or 0, _num_archivo(analisis[i].get("key", ""))))
    idx_de = {(x[0], x[1]): i for i, x in enumerate(inst)}
    enlaces_rafaga: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for p in range(len(orden) - 1):
        fa, fb = orden[p], orden[p + 1]
        a, b = analisis[fa], analisis[fb]
        ha, hb = a.get("hora"), b.get("hora")
        if ha is not None and hb is not None:
            if abs(ha - hb) > RAFAGA_SEG:
                continue
        else:
            pa, na = _num_archivo(a.get("key", ""))
            pb, nb = _num_archivo(b.get("key", ""))
            if na is None or nb is None or pa != pb or abs(na - nb) > 2:
                continue
        if a.get("firma") is None or b.get("firma") is None:
            continue
        parecido = float(np.minimum(a["firma"], b["firma"]).sum())
        if parecido < 0.80:
            continue
        for ji, j in enumerate(a.get("jugadores", [])):
            for jv, jj in enumerate(b.get("jugadores", [])):
                ca, cb = j["caja"], jj["caja"]
                dc = abs((ca[0] + ca[2]) / 2 - (cb[0] + cb[2]) / 2) + abs((ca[1] + ca[3]) / 2 - (cb[1] + cb[3]) / 2)
                if dc < 0.12 and abs(j["altura"] - jj["altura"]) < 0.3 * max(j["altura"], jj["altura"]) \
                        and _dist_color(j["color"], jj["color"]) <= DIST_EQUIPO:
                    ia, ib = idx_de[(fa, ji)], idx_de[(fb, jv)]
                    if G.unir(ia, ib):
                        enlaces_rafaga[ia].append((ib, parecido))
                        enlaces_rafaga[ib].append((ia, parecido))

    # ---------------- 4. grupos -> jugadores (solo los que tienen identidad clara)
    grupos: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(inst)):
        grupos[G.raiz(i)].append(i)
    jugadores = []
    for r, miembros in grupos.items():
        fotos_todas = G.fotos[r]
        dorsal = None
        if G.dorsales[r]:
            k, v = G.dorsales[r].most_common(1)[0]
            vistas = sum(1 for i in miembros if inst[i][2]["dorsal"] == k)
            grande = max((inst[i][2].get("alto_dorsal", 0) for i in miembros if inst[i][2]["dorsal"] == k), default=0)
            con_cara = any(inst[i][2]["huella"] is not None for i in miembros)
            # visto en 2+ fotos, o una sola vez pero claro (numero de espalda/pecho, o con rostro visible)
            if (vistas >= 2 and v >= 2 * 0.86) or (v >= 0.97 and k != "1" and (grande >= 0.08 or (con_cara and grande >= 0.05))):
                dorsal = k
        if dorsal is None and (len(fotos_todas) < 2 or not G.huellas[r]):
            continue  # vista una sola vez y sin numero confiable: no merece carpeta propia

        # ----- medidor de certeza: que fotos del grupo tienen prueba SUFICIENTE y cuales son dudosas
        relevante = lambda i: bool(inst[i][2]["dorsal"]) or inst[i][2]["huella"] is not None or inst[i][2]["altura"] >= 0.3
        seguros: set = set()
        if dorsal:
            seguros = {i for i in miembros if inst[i][2]["dorsal"] == dorsal and inst[i][2]["conf_dorsal"] >= DORSAL_SEGURO
                       and not _numero_sospechoso(inst[i][2])}
            if not seguros:   # varias lecturas medianas que coinciden en fotos distintas tambien son prueba
                lect = [i for i in miembros if inst[i][2]["dorsal"] == dorsal and not _numero_sospechoso(inst[i][2])]
                if len({inst[i][0] for i in lect}) >= 2:
                    seguros = set(lect)
        else:
            caras_g = [i for i in miembros if inst[i][2]["huella"] is not None]
            mejor = max(((float(np.dot(inst[a][2]["huella"], inst[b][2]["huella"])), a, b)
                         for x, a in enumerate(caras_g) for b in caras_g[x + 1:] if inst[a][0] != inst[b][0]), default=None)
            if mejor and mejor[0] >= SIM_SEGURA:
                seguros = {mejor[1], mejor[2]}
        cambio = True
        while cambio:   # la certeza se contagia solo por evidencia fuerte (rostro muy parecido o rafaga casi identica)
            cambio = False
            for i in miembros:
                if i in seguros:
                    continue
                hi = inst[i][2]["huella"]
                fuerte = hi is not None and any(
                    inst[s][0] != inst[i][0] and inst[s][2]["huella"] is not None
                    and float(np.dot(hi, inst[s][2]["huella"])) >= SIM_SEGURA for s in seguros)
                fuerte = fuerte or any(s in seguros and p >= RAFAGA_SEGURA for s, p in enlaces_rafaga.get(i, []))
                if fuerte:
                    seguros.add(i)
                    cambio = True
        fotos_ok = sorted({inst[i][0] for i in seguros if relevante(i)})
        dudas = []
        for i in miembros:
            if i in seguros or not relevante(i) or inst[i][0] in fotos_ok:
                continue
            j = inst[i][2]
            if _numero_sospechoso(j):
                motivo = "Número grande visto de frente (¿camiseta en la mano?)"
            elif j["dorsal"] and j["dorsal"] == dorsal:
                motivo = "El número se ve poco nítido"
            elif j["huella"] is not None:
                motivo = "La cara se parece, pero no es idéntica"
            else:
                motivo = "Foto de la misma ráfaga"
            dudas.append({"foto": inst[i][0], "mini": j["mini"], "motivo": motivo})
        if not fotos_ok and not dudas:
            continue
        mini, mejor_q = b"", -1.0
        palabras: List[str] = []
        for i in miembros:
            j = inst[i][2]
            palabras += j["palabras"]
            q = (2.0 if j.get("mini_cara") else 0.0) + j.get("calidad_cara", 0.0) + j["altura"] + (5.0 if i in seguros else 0.0)
            if j["mini"] and q > mejor_q:
                mini, mejor_q = j["mini"], q
        # grupo sin ninguna foto segura: la primera duda queda como "referencia" y se pregunta por las demas
        if not fotos_ok and not dorsal and len(dudas) >= 2:
            ref = max(dudas, key=lambda d: len(d["mini"]))
            dudas.remove(ref)
            for d in dudas:
                d["foto_ref"] = ref["foto"]
            mini = ref["mini"] or mini
        jugadores.append({"dorsal": dorsal, "fotos": fotos_ok, "mini": mini, "color": G.color(r),
                          "palabras": palabras, "n": len(fotos_ok), "dudas": dudas})

    # ---------------- 5. equipo: lo leido en la camiseta del jugador; si no, se hereda solo de un
    #                     companero con camiseta casi identica; si no, el color (nunca un nombre inventado)
    portadores = []
    for P in jugadores:
        P["equipo"] = _nombre_equipo(P["palabras"])
        if P["equipo"] and P["color"] is not None:
            portadores.append((P["equipo"], P["color"]))
    for P in jugadores:
        if P["equipo"]:
            continue
        mejor, md = None, 9.0
        if P["color"] is not None:
            for n, c in portadores:
                d = _dist_color(P["color"], c)
                if d < md:
                    mejor, md = n, d
        if mejor and md <= DIST_HEREDAR_NOMBRE:
            P["equipo"] = mejor
        else:
            P["equipo"] = _nombre_color(P["color"]) if P["color"] is not None else "Sin_Equipo"

    # ---------------- 6. nombres de carpeta (unicos), con nomina opcional
    nom = parsear_nomina(nomina)

    def nombre_de(eq_nombre: str, dorsal: str) -> Optional[str]:
        sin_eq = None
        for e_, d_, n_ in nom:
            if d_ != dorsal:
                continue
            if e_ and difflib.SequenceMatcher(None, _canon(e_), _canon(eq_nombre)).ratio() >= 0.6:
                return n_
            if not e_ and sin_eq is None:
                sin_eq = n_
        return sin_eq

    carpetas: Dict[str, List[str]] = defaultdict(list)
    miniaturas: Dict[str, bytes] = {}
    detalle: Dict[str, Dict[str, Any]] = {}
    n_anon: Dict[str, int] = defaultdict(int)
    usadas = set()
    dudas_salida: List[Dict[str, Any]] = []
    for P in sorted(jugadores, key=lambda x: (x["equipo"], x["dorsal"] is None, int(x["dorsal"] or 0), -x["n"])):
        nombre = None
        if P["dorsal"]:
            nombre = nombre_de(P["equipo"], P["dorsal"])
            c = f"Dorsal_{int(P['dorsal']):02d}_{limpiar_nombre(P['equipo'])}" + (f"_{limpiar_nombre(nombre)}" if nombre else "")
        else:
            n_anon[P["equipo"]] += 1
            c = f"Jugador_{n_anon[P['equipo']]:02d}_{limpiar_nombre(P['equipo'])}"
        base, k = c, 2
        while c in usadas:
            c = f"{base}_{k}"
            k += 1
        usadas.add(c)
        if P["fotos"]:
            carpetas[c] = [analisis[fi]["key"] for fi in P["fotos"]]
        if P["mini"]:
            miniaturas[c] = P["mini"]
        detalle[c] = {"equipo": P["equipo"], "dorsal": P["dorsal"], "nombre": nombre, "fotos": len(P["fotos"])}
        for d in P["dudas"]:
            did = f"d{len(dudas_salida) + 1:04d}"
            dudas_salida.append({
                "id": did, "carpeta": c, "foto": analisis[d["foto"]]["key"], "motivo": d["motivo"],
                "foto_ref": analisis[d["foto_ref"]]["key"] if "foto_ref" in d else None,
                "dorsal": P["dorsal"], "equipo": P["equipo"], "estado": "pendiente",
            })
            if d["mini"]:
                miniaturas[f"_dudas/{did}"] = d["mini"]

    con_jugador = {k for ks in carpetas.values() for k in ks}
    for a in analisis:
        if a["key"] not in con_jugador:
            carpetas[GENERAL].append(a["key"])

    resumen = {
        "fotos": len(analisis),
        "jugadores": len([c for c in carpetas if c != GENERAL]),
        "con_dorsal": len([c for c in carpetas if c.startswith("Dorsal_")]),
        "equipos": sorted({P["equipo"] for P in jugadores}),
        "generales": len(carpetas.get(GENERAL, [])),
        "por_confirmar": len(dudas_salida),
    }
    return {"carpetas": dict(carpetas), "miniaturas": miniaturas, "resumen": resumen, "detalle": detalle,
            "dudas": dudas_salida}
