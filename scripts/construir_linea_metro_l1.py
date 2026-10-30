"""Construye ``models/metro_l1/linea_config.json``: geometría, oferta y vehículo de la Línea 1.

Uso (desde la raíz del backend)::

    python scripts/construir_linea_metro_l1.py

Entradas: ``data/geo/puntos_metro_l1.geojson`` y ``data/geo/trazado_metro_l1.geojson``
(shapefile oficial ATU). La calibración del modelo O-D (beta) la agrega después
``scripts/evaluar_od_metro_l1.py``.
"""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
GEO = RAIZ / "data" / "geo"
SALIDA = RAIZ / "models" / "metro_l1" / "linea_config.json"

FUENTES = {
    "horario_2026": "https://larepublica.pe/sociedad/2026/09/04/linea-1-del-metro-de-lima-oficializa-horarios-de-atencion-para-septiembre-de-2026-207920",
    "horario_dom_feriados": "https://www.infobae.com/peru/2026/10/18/horarios-del-tren-de-lima-frecuencia-de-paso-en-hora-punta-fines-de-semana-y-claves-para-organizar-el-viaje/",
    "ositran_2024": "https://www.ositran.gob.pe/anterior/wp-content/uploads/2018/04/id-2024-linea1.pdf",
    "ositran_2022": "https://www.ositran.gob.pe/anterior/wp-content/uploads/2023/07/id-l1-2022.pdf",
    "capacidad_1200": "https://gestion.pe/economia/metro-lima-incorpora-primer-tren-6-coches-capacidad-1-200-pasajeros-245920-noticia/",
    "metropolis_9000": "https://es.wikipedia.org/wiki/Alstom_Metropolis_9000",
    "linea1_wikipedia": "https://es.wikipedia.org/wiki/L%C3%ADnea_1_del_Metro_de_Lima_y_Callao",
    "tcqsm_rail": "https://onlinepubs.trb.org/onlinepubs/tcrp/docs/tcrp100/Part5.pdf",
    "crowding_review": "https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=1065&context=jpt",
}

# Horario oficial (septiembre 2026, publicado por la Línea 1): (desde, hasta, intervalo_min)
HORARIO = {
    "LAB": [("05:00", "05:50", 10), ("05:50", "06:18", 4), ("06:18", "09:27", 3),
            ("09:27", "15:37", 5), ("15:37", "16:33", 4), ("16:33", "20:12", 3),
            ("20:12", "21:12", 4), ("21:12", "22:00", 6)],
    "SAB": [("05:00", "06:00", 10), ("06:00", "07:05", 5), ("07:05", "08:44", 3.7),
            ("08:44", "10:04", 4), ("10:04", "11:04", 4.3), ("11:04", "12:03", 4.5),
            ("12:03", "13:07", 4), ("13:07", "15:02", 3.5), ("15:02", "19:04", 3.7),
            ("19:04", "20:04", 4), ("20:04", "21:47", 4.7), ("21:47", "22:00", 6.3)],
    # Domingos; feriados con el mismo esquema (Infobae, oct. 2026)
    "DOM": [("05:30", "06:06", 12), ("06:06", "07:12", 11), ("07:12", "08:08", 9.3),
            ("08:08", "09:04", 8), ("09:04", "11:04", 7.5), ("11:04", "17:04", 6.7),
            ("17:04", "18:58", 6), ("18:58", "19:50", 6.5), ("19:50", "21:49", 8.5),
            ("21:49", "22:00", 11)],
}
TIEMPO_EXTREMO_MIN = 54.0   # Villa El Salvador - Bayóvar (Wikipedia / Línea 1)

# Vehículo: tren Alstom Metropolis 9000 de 6 coches (102 m x 2,71 m)
CAPACIDAD = 1200            # pasajeros por tren (AATE / Ositrán)
ASIENTOS = round(156 * 6 / 5)   # ficha genérica 5 coches = 156 sentados -> 6 coches (supuesto)
LARGO_M, ANCHO_M = 102.0, 2.71
AREA_ASIENTO_M2 = 0.45      # huella por asiento (supuesto de diseño)
FRACCION_UTIL = 0.90        # descuenta cabinas, puertas y equipos (supuesto)
DENSIDAD_SATURADO = 4.0     # pie/m2: límite de confort (TCQSM nivel E desde 2,86; Tirachini 4-5)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def minutos(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def trenes_por_hora(bandas) -> list[float]:
    """Salidas por hora = sum(minutos de la banda dentro de la hora / intervalo)."""
    salidas = [0.0] * 24
    for ini, fin, hw in bandas:
        a, b = minutos(ini), minutos(fin)
        for h in range(24):
            sol = max(0, min(b, (h + 1) * 60) - max(a, h * 60))
            salidas[h] += sol / hw
    # Los trenes que salen antes de las 22:00 recorren la línea hasta ~23:00 (Ositrán: llegada
    # del último tren a las 23:00): en la hora 22 las estaciones intermedias siguen servidas.
    salidas[22] = max(salidas[22], 60.0 / bandas[-1][2])
    return [round(x, 3) for x in salidas]


def haversine_km(p, q) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (p[0], p[1], q[0], q[1]))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def longitud_trazado_km(trazado) -> float:
    total = 0.0
    for f in trazado["features"]:
        c = f["geometry"]["coordinates"]
        total += sum(haversine_km(p, q) for p, q in zip(c, c[1:]))
    return total


def posiciones_km(estaciones, trazado) -> list[float]:
    """Km de cada estación: distancias rectas entre estaciones consecutivas, reescaladas para
    que el total coincida con la longitud del trazado oficial (corrige las curvas)."""
    pos = [0.0]
    for a, b in zip(estaciones, estaciones[1:]):
        pos.append(pos[-1] + haversine_km(a, b))
    escala = longitud_trazado_km(trazado) / pos[-1]
    return [round(x * escala, 3) for x in pos]


def main() -> None:
    puntos = json.loads((GEO / "puntos_metro_l1.geojson").read_text(encoding="utf-8"))
    trazado = json.loads((GEO / "trazado_metro_l1.geojson").read_text(encoding="utf-8"))
    card = json.loads((RAIZ / "models" / "metro_l1" / "model_card.json").read_text(encoding="utf-8"))
    serie_por_norm = {norm(s): s for s in card["series"]}

    feats = puntos["features"]   # el GeoJSON viene ordenado de Villa El Salvador a Bayóvar
    coords = [f["geometry"]["coordinates"] for f in feats]
    nombres = [f["properties"]["estacion"] for f in feats]
    orden = [serie_por_norm[norm(n)] for n in nombres]
    assert sorted(orden) == sorted(card["series"]), "estaciones sin serie en el modelo"
    assert norm(orden[0]) == "villa el salvador" and norm(orden[-1]) == "bayovar"

    km = posiciones_km(coords, trazado)
    area_pie = LARGO_M * ANCHO_M * FRACCION_UTIL - ASIENTOS * AREA_ASIENTO_M2
    umbral_saturado = ASIENTOS + DENSIDAD_SATURADO * area_pie

    cfg = {
        "sistema": "metro_l1",
        "version": "2.0.0",
        "estaciones": orden,
        "nombres_geojson": nombres,
        "km": km,
        "longitud_km": km[-1],
        "sentidos": {"asc": "hacia Bayóvar", "desc": "hacia Villa El Salvador"},
        "tiempo_extremo_a_extremo_min": TIEMPO_EXTREMO_MIN,
        "velocidad_comercial_kmh": round(km[-1] / (TIEMPO_EXTREMO_MIN / 60), 2),
        "oferta": {
            "fuente": FUENTES["horario_2026"],
            "nota": "feriados = esquema DOM; hora 22 servida por los últimos trenes en ruta",
            "horario": {k: [list(b) for b in v] for k, v in HORARIO.items()},
            "trenes_hora": {k: trenes_por_hora(v) for k, v in HORARIO.items()},
        },
        "vehiculo": {
            "modelo": "Alstom Metropolis 9000, 6 coches",
            "capacidad": CAPACIDAD,
            "asientos": ASIENTOS,
            "asientos_estado": "estimado: 156 sentados por tren de 5 coches (ficha genérica) x 6/5",
            "area_de_pie_m2": round(area_pie, 1),
            "densidad_implicita_capacidad_pax_m2": round((CAPACIDAD - ASIENTOS) / area_pie, 2),
            "umbral_saturado_pasajeros": round(umbral_saturado),
            "densidad_saturado_pax_m2": DENSIDAD_SATURADO,
        },
        "od": {
            "metodo": "gravedad doblemente acotada (Furness) por hora; bajadas por viaje de retorno",
            "beta": 0.0,
            "n": 0.0,
            "beta_criterio": "ME2 (Ortúzar §12.4.4): sin distribución de longitudes observada, la "
                             "matriz más probable es la de máxima entropía; además beta = 0 minimiza "
                             "la asimetría de la matriz diaria (ver od_evaluacion.json)",
            "bloques": {"am": list(range(5, 10)), "md": list(range(10, 16)),
                        "pm": list(range(16, 22)), "noche": [22, 23, 0, 1, 2, 3, 4]},
            "espejo": {"am": "pm", "md": "md", "pm": "am", "noche": "dia"},
        },
        "clases": {
            "Asientos": "carga del tren <= asientos",
            "De_pie": "asientos < carga <= asientos + 4 pasajeros/m2 de pie",
            "Saturado": "más de 4 pasajeros/m2 de pie (TCQSM: incomodidad desde 2,86; aplastamiento > 5)",
        },
        "fuentes": FUENTES,
    }
    SALIDA.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{SALIDA.relative_to(RAIZ)}: {len(orden)} estaciones, {km[-1]:.1f} km, "
          f"área de pie {area_pie:.0f} m2, densidad a capacidad "
          f"{cfg['vehiculo']['densidad_implicita_capacidad_pax_m2']} pax/m2, "
          f"saturado > {umbral_saturado:.0f} pasajeros")
    for k, v in cfg["oferta"]["trenes_hora"].items():
        print(k, v)


if __name__ == "__main__":
    main()
