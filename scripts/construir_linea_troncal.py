"""Construye ``models/troncal/linea_config.json``: el Metropolitano troncal como una línea.

Uso (desde la raíz del backend)::

    python scripts/construir_linea_troncal.py

Topología (proyección de las estaciones sobre la relación OSM "Regular C" y coordenadas ATU):

- Eje sur (Vía Expresa): Matellini -> Estación Central, posición = proyección sobre el trazado.
  Andrés Reyes (no está en el GeoJSON) va entre Canaval y Moreyra y Javier Prado, a una cuadra
  de Canaval y Moreyra (inaugurada en 2020).
- Centro: el ramal Colmena - Jr. de la Unión - Tacna - Ramón Castilla y el eje España - Quilca -
  2 de Mayo son caminos paralelos entre Estación Central y Caquetá; se modelan como un solo
  corredor (posición = proyección sobre la recta Central-Caquetá). Simplificación declarada.
- Eje norte y ampliación: Caquetá -> Naranjal -> Chimpu Ocllo, en orden de latitud.
- "Naranjal Alimentador" comparte ubicación con Naranjal (no hay viajes entre ambas).
- Bulevar (Barranco, entre Estadio Unión y Balta): coordenada corregida en el GeoJSON (antes
  apuntaba al paradero de alimentador homónimo en Comas).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
GEO = RAIZ / "data" / "geo"
SALIDA = RAIZ / "models" / "troncal" / "linea_config.json"

SUR = ["Matellini", "Rosario de Villa", "Terán", "Escuela Militar", "Union", "Bulevar", "Balta",
       "Plaza de Flores", "28 de Julio", "Benavides", "Ricardo Palma", "Angamos", "Domingo Orué",
       "Aramburu", "Canaval y Moreyra", "Andrés Reyes", "Javier Prado", "Canadá", "México",
       "Estadio Nacional", "Estacion Central"]
CENTRO = ["Colmena", "Jiron de la Unión", "Tacna", "Ramon Castilla", "España", "Quilca", "2 de Mayo"]
NORTE = ["Caqueta", "Parque del Trabajo", "Uni", "Honorio Delgado", "El Milagro", "Tomas Valle",
         "Los jazmines", "Independencia", "Pacifico", "Izaguirre", "Naranjal", "Naranjal Alimentador",
         "Universidad", "Las Vegas", "22 De Agosto", "Andres Belaunde", "Los Incas", "Chimpu Ocllo"]

FUENTES = {
    "bus_164": "https://www.gob.pe/institucion/atu/noticias/895166-atu-nuevo-bus-articulado-del-metropolitano-podra-transportar-a-164-pasajeros",
    "brtdata_lima": "https://brtdata.org/location/latin_america/peru/lima/?lang=es",
    "horarios_2026": "https://www.infobae.com/peru/2026/10/17/horarios-del-metropolitano-apertura-de-estaciones-y-salidas-de-buses-en-lima/",
    "andres_reyes": "https://gestion.pe/peru/metropolitano-nueva-estacion-andres-reyes-san-isidro-estaria-lista-ano-273267-noticia/",
    "osm_regular_c": "https://www.openstreetmap.org/relation/1963019",
    "tcqsm": "https://onlinepubs.trb.org/onlinepubs/tcrp/docs/tcrp100/Part5.pdf",
}

# Bus articulado 18,5 x 2,56 m: 164 pasajeros, 47 sentados y 117 de pie (ATU, 2024)
CAPACIDAD, ASIENTOS, LARGO_M, ANCHO_M = 164, 47, 18.5, 2.56
AREA_ASIENTO_M2, FRACCION_UTIL, DENSIDAD_SATURADO = 0.45, 0.85, 4.0
# Oferta: 117 buses/h en hora punta (Protransporte, vía BRTData, 2013) y perfil del día según los
# intervalos publicados (Infobae oct. 2026: punta 2-4 min, valle 5-7, extremos 8-12; domingos y
# feriados solo A, B y C)
BUSES_PUNTA = 117.0
RATIO_HORA = {h: 1.0 for h in (6, 7, 8, 17, 18, 19)} | {h: 0.5 for h in (*range(9, 17), 20, 21)} | {5: 0.3, 22: 0.3}
RATIO_DIA = {"LAB": 1.0, "SAB": 0.6, "DOM": 0.3}
VELOCIDAD_KMH = 25.5   # Protransporte, vía BRTData
CV_INTERVALO = 0.3     # supuesto: carril segregado, regularidad intermedia entre metro (0) y bus (0,5)


def intervalo_servicio() -> dict:
    """Intervalo (min) del servicio que toma el usuario: punta 3, valle 6, extremos 10."""
    base = {"LAB": (3.0, 6.0, 10.0), "SAB": (4.0, 6.0, 10.0), "DOM": (6.0, 8.0, 12.0)}
    out = {}
    for b, (punta, valle, extremo) in base.items():
        out[b] = [punta if RATIO_HORA.get(h) == 1.0 else valle if RATIO_HORA.get(h) == 0.5
                  else extremo if h in RATIO_HORA else 0.0 for h in range(24)]
    return out


def haversine_km(p, q) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (p[0], p[1], q[0], q[1]))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def main() -> None:
    from shapely.geometry import LineString, Point, shape

    card = json.loads((RAIZ / "models" / "troncal" / "model_card.json").read_text(encoding="utf-8"))
    pts = json.loads((GEO / "puntos_troncal.geojson").read_text(encoding="utf-8"))["features"]
    coord = {f["properties"]["unidad"]: tuple(f["geometry"]["coordinates"]) for f in pts}
    tr = shape(json.loads((GEO / "trazado_troncal.geojson").read_text(encoding="utf-8"))["features"][0]["geometry"])
    km_grado = 111.32 * math.cos(math.radians(12.05))   # km por grado de longitud ~ latitud

    km = {}
    # eje sur: distancia a lo largo del trazado OSM
    base = tr.project(Point(coord["Matellini"]))
    for e in SUR:
        if e in coord:
            km[e] = (tr.project(Point(coord[e])) - base) * (km_grado + 111.32) / 2
    km["Andrés Reyes"] = km["Canaval y Moreyra"] + 0.15
    # centro: proyección sobre la recta Estación Central - Caquetá
    c0, c1 = coord["Estacion Central"], coord["Caqueta"]
    recta = LineString([c0, c1])
    largo_centro = haversine_km(c0, c1)
    for e in CENTRO:
        km[e] = km["Estacion Central"] + recta.project(Point(coord[e]), normalized=True) * largo_centro
    km["Caqueta"] = km["Estacion Central"] + largo_centro
    # norte: distancias rectas encadenadas
    for a, b in zip(NORTE, NORTE[1:]):
        prev = "Naranjal" if a == "Naranjal Alimentador" else a
        km[b] = km[prev] if b == "Naranjal Alimentador" else km[prev] + haversine_km(coord[prev], coord[b])
    orden = SUR + CENTRO + NORTE
    estaciones = [e for e in orden if e in card["series"]]
    faltan = set(card["series"]) - set(estaciones)
    assert not faltan, faltan
    estaciones.sort(key=lambda e: km[e])
    area_pie = LARGO_M * ANCHO_M * FRACCION_UTIL - ASIENTOS * AREA_ASIENTO_M2
    oferta = {b: [round(BUSES_PUNTA * RATIO_DIA[b] * RATIO_HORA.get(h, 0.0), 2) for h in range(24)]
              for b in ("LAB", "SAB", "DOM")}
    cfg = {
        "sistema": "troncal", "version": "2.0.0",
        "estaciones": estaciones,
        "km": [round(km[e], 3) for e in estaciones],
        "longitud_km": round(max(km.values()), 2),
        "sentidos": {"asc": "hacia el norte (Naranjal / Chimpu Ocllo)", "desc": "hacia el sur (Matellini)"},
        "velocidad_comercial_kmh": VELOCIDAD_KMH,
        "oferta": {"trenes_hora": oferta, "cv_intervalo": CV_INTERVALO,
                   "intervalo_servicio_min": intervalo_servicio(),
                   "fuente_intervalo_servicio": "Infobae oct. 2026: punta 2-4 min, valle 5-7, "
                                                "extremos 8-12 por servicio (sáb./dom. supuesto)",
                   "fuente": "117 buses/h en punta (Protransporte 2013, BRTData) x perfil horario "
                             "(Infobae oct. 2026). Dato de 2013, previo a la ampliación norte: en la "
                             "ampliación pasan menos buses que en el eje central",
                   "ratio_hora": RATIO_HORA, "ratio_dia": RATIO_DIA},
        "vehiculo": {
            "modelo": "bus articulado 18,5 m", "capacidad": CAPACIDAD, "asientos": ASIENTOS,
            "fuente": FUENTES["bus_164"], "area_de_pie_m2": round(area_pie, 1),
            "densidad_implicita_capacidad_pax_m2": round((CAPACIDAD - ASIENTOS) / area_pie, 2),
            "umbral_saturado_pasajeros": round(ASIENTOS + DENSIDAD_SATURADO * area_pie, 1),
        },
        "od": {
            "metodo": "gravedad doblemente acotada (Furness) por hora; bajadas por viaje de retorno",
            "beta": 0.0, "n": 0.0,
            "bloques": {"am": list(range(5, 10)), "md": list(range(10, 16)),
                        "pm": list(range(16, 22)), "noche": [22, 23, 0, 1, 2, 3, 4]},
            "espejo": {"am": "pm", "md": "md", "pm": "am", "noche": "dia"},
        },
        "simplificaciones": [
            "servicios regulares y expresos se agregan en un solo flujo de buses por tramo",
            "el ramal del centro y el eje España-Quilca-2 de Mayo se modelan como un solo corredor",
            "la frecuencia es la del eje central (117 buses/h en punta) en toda la línea",
        ],
        "fuentes": FUENTES,
    }
    SALIDA.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(estaciones)} estaciones, {cfg['longitud_km']} km, densidad a capacidad "
          f"{cfg['vehiculo']['densidad_implicita_capacidad_pax_m2']} pax/m2, "
          f"saturado > {cfg['vehiculo']['umbral_saturado_pasajeros']}")
    print([(e, round(km[e], 1)) for e in estaciones])


if __name__ == "__main__":
    main()
