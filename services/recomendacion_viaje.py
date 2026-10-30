"""Elige el paradero de subida de un viaje por costo generalizado, no solo por caminata.

``/api/geo/match-trip`` encuentra las rutas directas entre el origen y el destino del usuario
(paradero de subida y de bajada a menos de un radio). Antes se ordenaban solo por metros de
caminata. Ahora cada candidata se valora en minutos equivalentes (``model/recomendacion.py``,
Ortúzar ec. 10.18):

    2 · caminata + 2 · espera + multiplicador(hacinamiento) · tiempo a bordo

con la ocupación pronosticada para la hora actual en el sentido del viaje. Si la mejor opción no
es la más cercana y ahorra al menos 3 minutos equivalentes, se le dice al usuario que le conviene
caminar a otro paradero (el diseño de recomendación de paraderos del README de datos).
"""
from __future__ import annotations

import json
import math

from model import recomendacion as R
from services import calendario
from services import model_service as MS
from services.mapeo_estaciones import serie_de_estacion

MAX_CANDIDATAS = 12
FACTOR_RODEO = 1.3          # distancia en vía / distancia recta (supuesto)
VEL_BUS_DEFECTO_KMH = 20.0


def _haversine_m(a: dict, b: dict) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    x = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371000.0 * math.asin(math.sqrt(x))


def _viaje_min(sistema: str, serie: str, est_b: dict, est_l: dict) -> float:
    """Tiempo a bordo entre el paradero de subida y el de bajada."""
    capa = MS.modelo(sistema).linea
    if hasattr(capa, "estaciones"):                         # Metro L1 / troncal
        sb, sl = serie_de_estacion(est_b)[1], serie_de_estacion(est_l)[1]
        km = dict(zip(capa.estaciones, capa.cfg["km"]))
        if sb in km and sl in km:
            return abs(km[sb] - km[sl]) / float(capa.cfg["velocidad_comercial_kmh"]) * 60.0
    if hasattr(capa, "rutas") and serie in capa.rutas:      # corredores: km a lo largo de la ruta
        cb = (est_b.get("raw_properties") or {}).get("cod_paradero")
        cl = (est_l.get("raw_properties") or {}).get("cod_paradero")
        for sen in capa.rutas[serie]["sentidos"].values():
            if cb in sen["paradas"] and cl in sen["paradas"]:
                i, j = sen["paradas"].index(cb), sen["paradas"].index(cl)
                if j > i:
                    return (sen["km"][j] - sen["km"][i]) / capa.velocidad_kmh() * 60.0
    vel = capa.velocidad_kmh() if hasattr(capa, "velocidad_kmh") else VEL_BUS_DEFECTO_KMH
    return _haversine_m(est_b, est_l) / 1000.0 * FACTOR_RODEO / vel * 60.0


def _sentido_valido(sistema: str, serie: str, est_b: dict, est_l: dict) -> bool:
    """Corredores: el paradero de subida debe estar en el sentido que llega al de bajada (no en
    la vereda del frente). Los demás sistemas definen el sentido con el destino."""
    capa = MS.modelo(sistema).linea
    if not (hasattr(capa, "rutas") and serie in capa.rutas):
        return True
    cb = (est_b.get("raw_properties") or {}).get("cod_paradero")
    cl = (est_l.get("raw_properties") or {}).get("cod_paradero")
    for sen in capa.rutas[serie]["sentidos"].values():
        if cb in sen["paradas"] and cl in sen["paradas"]:
            return sen["paradas"].index(cl) > sen["paradas"].index(cb)
    return False


def _evaluar(ruta: dict, hora: int) -> dict:
    from routes.predictions import _buscar_estacion, _item   # evita import circular
    est_b = _buscar_estacion(ruta["boarding_id"], ruta.get("route_code"))
    est_l = _buscar_estacion(ruta["leaving_id"], ruta.get("route_code"))
    caminata = float(ruta.get("origin_walk_m") or 0) + float(ruta.get("dest_walk_m") or 0)
    sistema, serie = serie_de_estacion(est_b) if est_b else (None, None)
    base = {"caminata_min": round(caminata / R.VELOCIDAD_PIE_M_MIN, 1)}
    if serie is None or est_l is None:
        return {**base, "sin_prediccion": True, "costo_generalizado_min": None}
    if not _sentido_valido(sistema, serie, est_b, est_l):
        return {**base, "sentido_contrario": True, "costo_generalizado_min": None}
    hacia = serie_de_estacion(est_l)[1] if sistema in ("metro_l1", "troncal") else None
    paradero = (est_b.get("raw_properties") or {}).get("cod_paradero") if sistema == "corredor" else None
    res = MS.prediccion_serie(sistema, serie, None, hacia=hacia, paradero=paradero)
    fila = next(f for f in res["filas"] if f["fecha_hora"].hour == hora)
    it = _item(fila, res["sentido"], res["sentidos"])
    viaje = _viaje_min(sistema, serie, est_b, est_l)
    if it["eta_min"] is None:                                # fuera de horario
        return {**base, "sin_servicio": True, "costo_generalizado_min": None,
                "aforo": {"clase": it["class"]}}
    mult = it.get("multiplicador_hacinamiento") or 1.0
    gc = float(R.costo(it["eta_min"], viaje, mult, caminata))
    return {**base,
            "espera_min": it["eta_min"], "viaje_min": round(viaje, 1),
            "costo_generalizado_min": round(gc, 1),
            "recomendacion_paradero": it["recommendation"],
            "probabilidad_no_subir": it.get("probabilidad_no_subir"),
            "aforo": {"clase": it["class"], "ocupacion": it["crowding"],
                      "pasajeros_por_vehiculo": it.get("pasajeros_por_tren"), "sentido": it.get("sentido")}}


def ordenar_rutas(rutas: list[dict], hora: int | None = None) -> tuple[list[dict], dict | None]:
    """Enriquece y reordena las rutas de match-trip; devuelve (rutas, recomendación del viaje)."""
    if not rutas:
        return rutas, None
    hora = calendario.ahora_lima().hour if hora is None else hora
    # se evalúan las candidatas en orden de caminata hasta tener MAX_CANDIDATAS en el sentido correcto
    evaluadas, validas_n = [], 0
    for r in rutas:
        if validas_n >= MAX_CANDIDATAS:
            break
        try:
            r.update(_evaluar(r, hora))
        except Exception as exc:  # noqa: BLE001 - una candidata sin datos no rompe el viaje
            r.update({"sin_prediccion": True, "costo_generalizado_min": None, "detalle_error": str(exc)})
        evaluadas.append(r)
        validas_n += not r.get("sentido_contrario")
    clave = lambda r: (r.get("costo_generalizado_min") is None,  # noqa: E731
                       r.get("costo_generalizado_min") or 0.0,
                       r["origin_walk_m"] + r["dest_walk_m"])
    validas = [r for r in evaluadas if r.get("costo_generalizado_min") is not None] or evaluadas
    mas_cercana = min(validas, key=lambda r: r["origin_walk_m"] + r["dest_walk_m"])
    evaluadas = sorted(evaluadas, key=clave)
    mejor = evaluadas[0]
    resumen = {"criterio": "costo generalizado = 2·caminata + 2·espera + hacinamiento·tiempo a bordo (min equivalentes)",
               "hora": hora, "mejor": {"boarding_station": mejor["boarding_station"], "route_code": mejor["route_code"]}}
    if (mejor is not mas_cercana and mejor.get("costo_generalizado_min") is not None
            and mas_cercana.get("costo_generalizado_min") is not None):
        ahorro = mas_cercana["costo_generalizado_min"] - mejor["costo_generalizado_min"]
        if ahorro >= R.AHORRO_MINIMO_MIN:
            extra_m = round(mejor["origin_walk_m"] - mas_cercana["origin_walk_m"])
            clase = {"Asientos": "con asientos", "De_pie": "de pie", "Saturado": "saturado"}.get(
                (mejor.get("aforo") or {}).get("clase"), "")
            if mejor["boarding_id"] != mas_cercana["boarding_id"]:
                que = (f"subir en {mejor['boarding_station']} ({mejor['route_code']}) en vez de "
                       f"{mas_cercana['boarding_station']}" + (f": caminas {extra_m} m más" if extra_m > 0 else ""))
                resumen["cambiar_paradero"] = True
            elif mejor["route_code"] != mas_cercana["route_code"]:
                que = f"tomar la {mejor['route_code']} en vez de la {mas_cercana['route_code']}"
            else:
                que = f"bajar en {mejor['leaving_station']} en vez de {mas_cercana['leaving_station']}"
            resumen.update({"ahorro_min": round(ahorro, 1),
                            "mensaje": f"Te conviene {que}; el vehículo va {clase} y ahorras ~{round(ahorro)} min equivalentes de viaje."})
    return evaluadas + rutas[len(evaluadas):], resumen
