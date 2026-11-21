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
DIST_MISMA_FOTO = 0.35      # companero de la misma foto (misma luz): basta una camiseta parecida
RAFAGA_SEG = 2.0

# Medidor de certeza: solo lo que supera estos umbrales entra solo a la carpeta; lo demas se pregunta.
DIST_CAMISETA_NUMERO = 0.35 # mismo numero + camiseta parecida = mismo jugador (si no coincide nada mas)
DIST_FIRMA_DISTINTA = 25.0  # cifra claramente de otro color (Delta E): mismo numero, otra camiseta
SIM_CARAS_DISTINTAS = 0.30  # por debajo, dos rostros son de personas distintas (SFace)
NITIDEZ_SUELTO = 55.0       # medido: caras desenfocadas de fondo 15-47, nitidas 81+
CARA_SUELTO = 0.40          # calidad minima de cara para dar carpeta a quien sale una sola vez
DIST_EQUIPO_GRUPO = 0.40    # camisetas del mismo equipo dentro del partido
OTROS = "Otros_jugadores"    # arqueros, sueltos y equipos con menos de 3 jugadores: una sola carpeta
SIM_SEGURA = 0.56           # coseno SFace "misma persona sin duda" (margen sobre FAR 1e-5 = 0.472 en LFW)
DORSAL_SEGURO = 0.97        # lectura de dorsal practicamente sin ambiguedad
RAFAGA_SEGURA = 0.88        # foto vecina casi identica
NUMERO_GRANDE = 0.10        # numero >10% de la altura de la persona = numero de espalda
DIST_COLOR_SEGURA = 0.60    # una union "segura" por cara exige camisetas compatibles
CARA_PROTAGONISTA = 0.55    # calidad de cara minima para abrir ficha con una sola foto
ALTURA_PROTAGONISTA = 0.30  # el protagonista ocupa al menos 30% de la altura de la foto
SIM_PREGUNTAR_UNION = 0.40  # dos fichas con caras parecidas (sin llegar a "seguro"): se pregunta si unirlas
# "¿Quien es?": sugerencias para protagonistas sin identificar (el fotografo elige con un toque)
SIM_SUGERIR = 0.40
DIST_SUGERIR_COLOR = 0.50
MAX_CANDIDATOS = 3


def _es_protagonista(x, analisis) -> bool:
    """Persona principal de su foto, con cara nitida: suficiente para abrirle ficha aunque salga una vez."""
    fi, _, j = x[0], x[1], x[2]
    if j["huella"] is None or j.get("calidad_cara", 0) < CARA_PROTAGONISTA or j["altura"] < ALTURA_PROTAGONISTA:
        return False
    alturas = sorted((o["altura"] for o in analisis[fi].get("jugadores", [])), reverse=True)
    return j["altura"] >= 0.6 * alturas[0] and (len(alturas) < 3 or j["altura"] >= alturas[1] * 0.999)


def _numero_sospechoso(j: Dict[str, Any]) -> bool:
    """Numero grande (de espalda) y a la vez cara de frente: tipico de una camiseta sostenida en la mano
    (entrega, celebracion) o de una lectura que no pertenece a esa persona."""
    if j.get("numero_en_mano"):
        return True
    if j.get("fuente_dorsal") == "vision":
        return False      # Claude confirmo que la camiseta esta puesta
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


def equipos_en_texto(texto: str) -> set:
    """Equipos conocidos presentes en un texto OCR pegado (ej. 'DEPORTETRUJILLODEPORTE', 'CARABOBO11')."""
    t = _canon(texto)
    out = set()
    for n in NOMBRES_EQUIPO:
        cn = _canon(n)
        if len(cn) < 4 or len(t) < 4:
            continue
        L = len(cn)
        if L <= 6:
            if cn in t:          # nombres cortos (Lara, Apure, Sucre...): solo coincidencia completa
                out.add(n)
            continue
        mejor = max((difflib.SequenceMatcher(None, cn, t[k:k + L]).ratio() for k in range(0, max(1, len(t) - L + 1))),
                    default=0.0)
        if mejor >= 0.82:
            out.add(n)
    return out


def _vistas_distintas(fotos: List[int], analisis: List[Dict[str, Any]]) -> int:
    """Cuantas fotos REALMENTE distintas: copias o fotos casi identicas cuentan como una sola
    (una lectura equivocada repetida en una copia no debe 'confirmarse' a si misma)."""
    unicas: List[int] = []
    for fi in dict.fromkeys(fotos):
        fa = analisis[fi].get("firma")
        if fa is not None and any(analisis[u].get("firma") is not None
                                  and float(np.minimum(fa, analisis[u]["firma"]).sum()) >= 0.97 for u in unicas):
            continue
        unicas.append(fi)
    return len(unicas)


def _equipo_por_vision(nombres: List[str]) -> Optional[str]:
    """Nombre de equipo leido por Claude: se normaliza contra la lista de estados/ciudades si coincide."""
    if not nombres:
        return None
    normal = [(_nombre_equipo([n, n]) or n.strip().title()) for n in nombres if n and n.strip()]
    return Counter(normal).most_common(1)[0][0] if normal else None


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



def _agrupar_equipos(jugadores: List[Dict[str, Any]]) -> None:
    """Agrupa a los jugadores del partido por camiseta (dentro del partido, misma luz). Cada grupo toma el
    nombre de equipo leido por sus miembros; si nadie lo leyo, 'Equipo_A', 'Equipo_B'... Los 'sueltos'
    (una foto, sin numero) solo se quedan si su camiseta es de un equipo con 2+ jugadores."""
    fijos = sorted([P for P in jugadores if not P.get("suelto") and P["color"] is not None], key=lambda P: -P["n"])
    centros: List[List[Any]] = []          # [color_medio, miembros]
    for P in fijos:
        mejor, md = None, DIST_EQUIPO_GRUPO
        for c in centros:
            d = _dist_color(P["color"], c[0])
            if d <= md:
                mejor, md = c, d
        if mejor is None:
            centros.append([np.array(P["color"], dtype=np.float32), [P]])
        else:
            mejor[1].append(P)
            mejor[0] = np.mean([np.asarray(x["color"], dtype=np.float32) for x in mejor[1]], axis=0)
    centros.sort(key=lambda c: -len(c[1]))
    # MENOS ES MAS: un "equipo" de 1-2 personas (arqueros, arbitro, sueltos) no merece carpeta propia
    for c in [c for c in centros if len(c[1]) < 3]:
        for P in c[1]:
            P["grupo"] = OTROS
    centros = [c for c in centros if len(c[1]) >= 3]
    usados, letra = set(), 0
    for c in centros:
        nombres = Counter(P["equipo"] for P in c[1] if P["equipo"])
        nombre = next((n for n, _ in nombres.most_common() if n not in usados), None)
        if not nombre:
            nombre = f"Equipo_{chr(65 + letra)}"
            letra += 1
        usados.add(nombre)
        c.append(nombre)
        for P in c[1]:
            P["grupo"] = nombre
    for P in jugadores:
        if P.get("grupo") or P["color"] is None or not centros:
            continue
        d, c = min(((_dist_color(P["color"], c[0]), c) for c in centros), key=lambda x: x[0])
        if P.get("suelto"):
            # quien sale una sola vez solo es jugador si en ESA foto aparece junto a jugadores de su equipo
            # (foto de grupo, jugada). Gradas, publico y tecnicos solos no reciben carpeta.
            fotos_eq = {f for x in c[1] for f in x["fotos"]}
            if d <= DIST_EQUIPO_GRUPO - 0.05 and set(P["fotos"]) & fotos_eq:
                P["grupo"] = c[2]
        elif d <= DIST_EQUIPO_GRUPO:
            P["grupo"] = c[2]
        else:
            P["grupo"] = OTROS


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
            # REGLA: mismo numero dentro del partido = el mismo jugador (en un equipo no se repiten numeros),
            # SALVO prueba de que son personas distintas: caras distintas, cifra de otro color, otro equipo escrito.
            dn = dist_firma_numero(ja.get("firma_numero"), jb.get("firma_numero"))
            if dn < 999 and dn > DIST_FIRMA_DISTINTA:
                continue                                    # cifra de otro color/estilo: otra camiseta
            ea, eb = _nombre_equipo(ja["palabras"]), _nombre_equipo(jb["palabras"])
            if ea and eb and ea != eb:
                continue                                    # equipos escritos distintos
            ra, rb = G.raiz(a), G.raiz(b)
            ha, hb = G.huellas[ra], G.huellas[rb]
            sim = float((np.array(ha) @ np.array(hb).T).max()) if ha and hb else None
            if sim is not None and sim < SIM_UNIR_A_DORSAL:
                continue                                    # las dos caras se ven y no se parecen: dos personas
                                                            # (auditoria en vivo: 3 mezclas de 12 por numeros iguales)
            # el numero solo NO basta: tiene que coincidir algo mas (cara, color de la cifra o camiseta).
            # Medido en vivo: un logo "R5" y un "15" tapado dieron dos "5" falsos y mezclaron a dos personas.
            misma_cifra = dn <= DIST_FIRMA_NUMERO
            misma_cara = sim is not None and sim >= SIM_UNIR_A_DORSAL
            misma_camiseta = _dist_color(ja["color"], jb["color"]) <= DIST_CAMISETA_NUMERO
            if not (misma_cifra or misma_cara or misma_camiseta):
                continue
            G.unir(a, b)                                    # (unir() ademas bloquea: misma foto, camiseta muy distinta)

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
            vistas = _vistas_distintas([inst[i][0] for i in miembros if inst[i][2]["dorsal"] == k], analisis)
            grande = max((inst[i][2].get("alto_dorsal", 0) for i in miembros if inst[i][2]["dorsal"] == k), default=0)
            con_cara = any(inst[i][2]["huella"] is not None for i in miembros)
            # visto en 2+ fotos, o una sola vez pero claro (numero de espalda/pecho, o con rostro visible)
            por_vision = any(inst[i][2].get("fuente_dorsal") == "vision" and inst[i][2]["dorsal"] == k for i in miembros)
            mejor_conf = max(inst[i][2]["conf_dorsal"] for i in miembros if inst[i][2]["dorsal"] == k)
            # la confianza es la MEJOR lectura individual (nunca la suma: las copias no se confirman a si mismas)
            if (vistas >= 2 and v >= 2 * 0.86) or (mejor_conf >= 0.97 and (k != "1" or por_vision)
                                                    and (grande >= 0.08 or (con_cara and grande >= 0.05))):
                dorsal = k
        protagonista_solo = None
        suelto = False
        if dorsal is None and len(fotos_todas) < 2:
            # partidos de 40-50 fotos: un protagonista con cara nitida merece su ficha aunque salga una sola vez
            cand = [i for i in miembros if _es_protagonista(inst[i], analisis)]
            if not cand:
                # NADIE FUERA: con cara visible tambien recibe carpeta (fotos de grupo, segundo plano);
                # mas abajo se descarta si su camiseta no es de ningun equipo del partido (publico, tecnicos)
                cand = [i for i in miembros if inst[i][2]["huella"] is not None
                        and inst[i][2].get("calidad_cara", 0) >= CARA_SUELTO
                        and inst[i][2].get("nitidez", NITIDEZ_SUELTO) >= NITIDEZ_SUELTO]   # caras de fondo desenfocadas: Generales
                if not cand:
                    continue
                suelto = True
            protagonista_solo = max(cand, key=lambda i: inst[i][2]["calidad_cara"] if suelto else inst[i][2]["altura"])
        elif dorsal is None and not G.huellas[r]:
            continue

        # ----- medidor de certeza: que fotos del grupo tienen prueba SUFICIENTE y cuales son dudosas
        relevante = lambda i: bool(inst[i][2]["dorsal"]) or inst[i][2]["huella"] is not None or inst[i][2]["altura"] >= 0.3
        seguros: set = set()
        if protagonista_solo is not None:
            seguros = {protagonista_solo}     # es su propia foto: no hay nada que confundir
        elif dorsal:
            seguros = {i for i in miembros if inst[i][2]["dorsal"] == dorsal and inst[i][2]["conf_dorsal"] >= DORSAL_SEGURO
                       and not _numero_sospechoso(inst[i][2])}
            if not seguros:   # varias lecturas medianas que coinciden en fotos distintas tambien son prueba
                lect = [i for i in miembros if inst[i][2]["dorsal"] == dorsal and not _numero_sospechoso(inst[i][2])]
                if _vistas_distintas([inst[i][0] for i in lect], analisis) >= 2:
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
                # cara muy parecida Y camiseta compatible (candado extra contra gemelos de otro equipo)
                fuerte = hi is not None and any(
                    inst[s][0] != inst[i][0] and inst[s][2]["huella"] is not None
                    and float(np.dot(hi, inst[s][2]["huella"])) >= SIM_SEGURA
                    and _dist_color(inst[i][2]["color"], inst[s][2]["color"]) <= DIST_COLOR_SEGURA for s in seguros)
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
        equipos_v: List[str] = []
        for i in miembros:
            j = inst[i][2]
            palabras += j["palabras"]
            if j.get("equipo_vision"):
                equipos_v.append(j["equipo_vision"])
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
                          "palabras": palabras, "equipos_v": equipos_v, "n": len(fotos_ok), "dudas": dudas,
                          "huellas": [inst[i][2]["huella"] for i in seguros if inst[i][2]["huella"] is not None],
                          "lecturas": {inst[i][2]["dorsal"] for i in miembros if inst[i][2]["dorsal"]},
                          "miembros": set(miembros), "suelto": suelto})

    # ---------------- 5. equipo: lo leido en la camiseta del jugador; si no, se hereda de un companero
    #                     DE LA MISMA FOTO con la misma camiseta (misma luz, mismo partido); si no, de un
    #                     jugador con camiseta casi identica; si no, se deja vacio. Nunca un color como nombre.
    def votos_equipo(miembros_):
        v: Counter = Counter()
        for i in miembros_:
            j = inst[i][2]
            if j.get("equipo_vision"):
                n = _equipo_por_vision([j["equipo_vision"]])
                if n:
                    v[n] += 1
            n = _nombre_equipo(j["palabras"])
            if n:
                v[n] += 1
        return v

    portadores = []
    nombres_foto: Dict[int, List[Tuple[np.ndarray, str]]] = defaultdict(list)
    for P in jugadores:
        v = votos_equipo(P["miembros"])
        top = v.most_common(2)
        # el equipo se escribe solo con 2 senales que coinciden (dos lecturas o dos fotos) y sin empate
        P["equipo"] = top[0][0] if top and top[0][1] >= 2 and (len(top) < 2 or top[0][1] > top[1][1]) else None
        if P["equipo"] and P["color"] is not None:
            portadores.append((P["equipo"], P["color"]))
            for fi in P["fotos"]:
                nombres_foto[fi].append((P["color"], P["equipo"]))
    for P in jugadores:
        if P["equipo"]:
            continue
        mejor, md = None, 9.0
        if P["color"] is not None:
            for fi in P["fotos"]:                       # 1) companeros de la misma foto
                for c, n in nombres_foto.get(fi, []):
                    d = _dist_color(P["color"], c)
                    if d <= DIST_MISMA_FOTO and d < md:
                        mejor, md = n, d
            if mejor is None:                          # 2) camiseta casi identica en otra foto
                for n, c in portadores:
                    d = _dist_color(P["color"], c)
                    if d <= DIST_HEREDAR_NOMBRE and d < md:
                        mejor, md = n, d
        P["equipo"] = mejor or ""

    # ---------------- 5c. equipos del partido por camiseta: MENOS ES MAS (el ZIP se ordena Equipo > Jugador)
    _agrupar_equipos(jugadores)
    jugadores = [P for P in jugadores if not (P.get("suelto") and not P.get("grupo"))]

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
        etiqueta = P["equipo"] or (P.get("grupo") if P.get("grupo") != OTROS else "") or ""
        eq = f"_{limpiar_nombre(etiqueta)}" if etiqueta else ""    # el nombre dice el equipo: nunca dos iguales
        if P["dorsal"]:
            nombre = nombre_de(P["equipo"], P["dorsal"])
            c = f"Dorsal_{int(P['dorsal']):02d}{eq}" + (f"_{limpiar_nombre(nombre)}" if nombre else "")
        else:
            n_anon[eq] += 1
            c = f"Jugador_{n_anon[eq]:02d}{eq}"
        base, k = c, 2
        g_ = P.get("grupo") or ""
        if c in usadas and g_ and g_ != OTROS and not P["equipo"] and limpiar_nombre(g_) not in base:
            c = f"{base}_{limpiar_nombre(g_)}"             # dos #8 de equipos distintos: lo dice el nombre
        while c in usadas:
            c = f"{base}_{k}"
            k += 1
        usadas.add(c)
        P["carpeta"] = c
        if P["fotos"]:
            carpetas[c] = [analisis[fi]["key"] for fi in P["fotos"]]
        if P["mini"]:
            miniaturas[c] = P["mini"]
        sugerido = None if P["dorsal"] else (sorted(P.get("lecturas") or [])[:1] or [None])[0]
        detalle[c] = {"equipo": P["equipo"], "dorsal": P["dorsal"], "nombre": nombre, "fotos": len(P["fotos"]),
                      "sugerido": sugerido, "grupo": P.get("grupo") or ""}
        for d in P["dudas"]:
            did = f"d{len(dudas_salida) + 1:04d}"
            dudas_salida.append({
                "id": did, "tipo": "mismo", "carpeta": c, "foto": analisis[d["foto"]]["key"], "motivo": d["motivo"],
                "foto_ref": analisis[d["foto_ref"]]["key"] if "foto_ref" in d else None,
                "dorsal": P["dorsal"], "equipo": P["equipo"], "estado": "pendiente",
            })
            if d["mini"]:
                miniaturas[f"_dudas/{did}"] = d["mini"]

    con_jugador = {k for ks in carpetas.values() for k in ks}
    for a in analisis:
        if a["key"] not in con_jugador:
            carpetas[GENERAL].append(a["key"])

    # ---------------- 6b. "¿Son la misma persona?": dos fichas con caras parecidas (sin llegar a 'seguro')
    todas = [P for P in jugadores if P.get("carpeta") and P["fotos"]]
    fichas = [P for P in todas if P["huellas"]]
    pares_u = []
    # pista de numero: ficha sin numero confirmado cuya lectura coincide con un jugador confirmado
    for B in todas:
        if B["dorsal"]:
            continue
        for A in todas:
            if A is B or not A["dorsal"] or A["dorsal"] not in B["lecturas"] or set(A["fotos"]) & set(B["fotos"]):
                continue
            if A["color"] is None or B["color"] is None or _dist_color(A["color"], B["color"]) <= DIST_UNION_COLOR:
                pares_u.append((1.5, A, B))
    # mismo numero en dos fichas (sin prueba para unirlas solas): se pregunta
    for x, A in enumerate(todas):
        for B in todas[x + 1:]:
            if A["dorsal"] and A["dorsal"] == B["dorsal"] and not (set(A["fotos"]) & set(B["fotos"])):
                if A.get("grupo") and B.get("grupo") and A["grupo"] != B["grupo"]:
                    continue        # mismo numero en equipos distintos: no hay nada que preguntar
                pares_u.append((1.6, A, B))
    for x, A in enumerate(fichas):
        for B in fichas[x + 1:]:
            if A["dorsal"] and B["dorsal"] and A["dorsal"] != B["dorsal"]:
                continue   # dos numeros distintos leidos con seguridad: personas distintas
            if set(A["fotos"]) & set(B["fotos"]):
                continue   # aparecen juntos en una foto
            if A["color"] is not None and B["color"] is not None and _dist_color(A["color"], B["color"]) > DIST_SUGERIR_COLOR:
                continue
            s = max(float(np.dot(h1, h2)) for h1 in A["huellas"] for h2 in B["huellas"])
            if s >= SIM_PREGUNTAR_UNION:
                pares_u.append((s, A, B))
    usadas_u = set()
    for s, A, B in sorted(pares_u, key=lambda t: -t[0]):
        if A["carpeta"] in usadas_u or B["carpeta"] in usadas_u:
            continue
        if (B["dorsal"] and not A["dorsal"]) or (len(B["fotos"]) > len(A["fotos"]) and not A["dorsal"]):
            A, B = B, A     # se conserva la ficha con numero (o con mas fotos)
        usadas_u.update({A["carpeta"], B["carpeta"]})
        dudas_salida.append({
            "id": f"u{len(dudas_salida) + 1:04d}", "tipo": "unir", "carpeta": A["carpeta"], "carpeta_b": B["carpeta"],
            "foto": analisis[B["fotos"][0]]["key"], "foto_ref": None, "dorsal": A["dorsal"], "equipo": A["equipo"],
            "estado": "pendiente",
            "motivo": (f"Las dos fichas tienen el número {A['dorsal']}" if s >= 1.6 else
                       f"En esta ficha se leyó el número {A['dorsal']}" if s >= 1.5
                       else "Dos fichas con caras parecidas: si son el mismo jugador, se unen"),
        })

    # ---------------- 7. "¿Quien es?": foto sin carpeta con un protagonista claro -> 3 candidatos para elegir
    en_duda = {d["foto"] for d in dudas_salida} | {d["foto_ref"] for d in dudas_salida if d.get("foto_ref")}
    con_carpeta = [P for P in jugadores if P.get("carpeta") and P["fotos"]]
    generales = set(carpetas.get(GENERAL, []))
    for fi, a in enumerate(analisis):
        if a["key"] not in generales or a["key"] in en_duda or not con_carpeta:
            continue
        for ji, j in sorted(enumerate(a.get("jugadores", [])), key=lambda t: -t[1]["altura"])[:2]:
            if j["altura"] < 0.28 and j["huella"] is None:
                continue
            cands = []
            for P in con_carpeta:
                d = _dist_color(j["color"], P["color"]) if (j["color"] is not None and P["color"] is not None) else 0.5
                mismo_num = bool(j["dorsal"]) and j["dorsal"] == P["dorsal"]
                # mismo numero: se tolera camiseta vista de otro lado; solo cara: exige camiseta cercana
                if d > (DIST_UNION_COLOR if mismo_num else DIST_SUGERIR_COLOR):
                    continue
                s = 0.0
                if j["huella"] is not None and P["huellas"]:
                    sim = max(float(np.dot(j["huella"], h)) for h in P["huellas"])
                    if sim >= SIM_SUGERIR:
                        s += 2.0 * sim
                if j["dorsal"] and j["dorsal"] == P["dorsal"]:
                    s += 1.0
                s += 0.4 * max(0.0, 1.0 - d / 0.5)
                if s >= 0.8:   # exige cara parecida (>=0.40) o mismo numero: el color solo no basta
                    cands.append((s, P["carpeta"]))
            if not cands:
                continue
            cands.sort(reverse=True)
            did = f"q{len(dudas_salida) + 1:04d}"
            dudas_salida.append({
                "id": did, "tipo": "quien", "carpeta": cands[0][1], "candidatos": [c for _, c in cands[:MAX_CANDIDATOS]],
                "foto": a["key"], "foto_ref": None, "dorsal": None, "equipo": None, "estado": "pendiente",
                "motivo": "Hay un protagonista, pero sin número ni cara seguros: elige quién es",
            })
            if j["mini"]:
                miniaturas[f"_dudas/{did}"] = j["mini"]
            break   # una sola pregunta por foto

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
