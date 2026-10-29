"""Endpoints de predicción de demanda, ocupación a bordo, espera y recomendación.

Las horas y fechas son de Lima (UTC-5). Para días sin datos ATU propios (posteriores a 2025) el
pronóstico se calcula con el día homólogo de 2025 y la respuesta lo declara en ``referencia``.
"""
import math

import pandas as pd
from flask import Blueprint, jsonify, request

from routes.stations import load_all_stations_into_memory
from services import calendario
from services import model_service as MS
from services.mapeo_estaciones import serie_de_estacion

predictions_bp = Blueprint("predictions", __name__)


def _num(x, nd=None):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd) if nd is not None else x


def _item(fila: dict, sentido: str | None, sentidos: dict | None) -> dict:
    """Una hora de pronóstico; con ``sentido`` usa la ocupación de ese sentido."""
    sfx = f"_{sentido}" if sentido and f"ocupacion_{sentido}" in fila else ""
    ocup = fila.get(f"ocupacion{sfx}") if sfx else fila.get("crowding_index")
    clase = fila.get(f"aforo{sfx}") if sfx else fila.get("aforo")
    eta = _num(fila.get(f"eta{sfx}") if sfx else fila.get("eta_min"))
    rec = fila.get(f"recomendacion{sfx}") if sfx else fila.get("recomendacion")
    hw = _num(fila.get("intervalo_min"))
    eta2 = (eta + hw) if (eta is not None and hw is not None) else _num(fila.get("eta2_min"))
    out = {
        "datetime": pd.Timestamp(fila["fecha_hora"]).strftime("%Y-%m-%d %H:%M"),
        "value": _num(fila.get("validaciones_pron"), 1),
        "band80": {"low": _num(fila.get("banda80_lo"), 1), "high": _num(fila.get("banda80_hi"), 1)},
        "crowding": _num(ocup, 3),
        "class": clase,
        "eta_min": _num(eta, 1),
        "eta2_min": _num(eta2, 1),
        "recommendation": rec,
    }
    if fila.get("metodo_aforo") == "ocupacion_od_v2_paradero":
        out.update({"sentido": fila.get("sentido_critico"), "paradero": fila.get("nombre"),
                    "pasajeros_por_tren": _num(fila.get("carga"), 0),
                    "bajadas_estimadas": _num(fila.get("bajadas_est"), 1),
                    "intervalo_min": _num(hw, 1)})
    elif fila.get("metodo_aforo") == "ocupacion_od_v2":
        out.update({
            "sentido": (sentidos or {}).get(sentido) if sfx else fila.get("sentido_critico"),
            "pasajeros_por_tren": _num(fila.get(f"carga{sfx}") if sfx else None, 0),
            "bajadas_estimadas": _num(fila.get("bajadas_est"), 1),
            "intervalo_min": _num(hw, 1),
            "por_sentido": {
                s: {"descripcion": (sentidos or {}).get(s),
                    "ocupacion": _num(fila.get(f"ocupacion_{s}"), 3),
                    "pasajeros_por_tren": _num(fila.get(f"carga_{s}"), 0),
                    "class": fila.get(f"aforo_{s}"),
                    "eta_min": _num(fila.get(f"eta_{s}"), 1),
                    "recommendation": fila.get(f"recomendacion_{s}")}
                for s in ("asc", "desc")
            },
        })
        if not sfx:
            crit = "asc" if fila.get("sentido_critico") == (sentidos or {}).get("asc") else "desc"
            out["pasajeros_por_tren"] = out["por_sentido"][crit]["pasajeros_por_tren"]
    return out


def _hora_param() -> int:
    hora = request.args.get("hour", default=None, type=int)
    if hora is None:
        return calendario.ahora_lima().hour
    if not 0 <= hora <= 23:
        raise MS.PrediccionError("La hora debe estar entre 0 y 23")
    return hora


def _meta(res: dict) -> dict:
    m = MS.modelo(res["sistema"])
    return {
        "system": res["sistema"], "series": res["serie"], "model": "lightgbm",
        "nivel": res.get("nivel", "serie"),
        "metodo_aforo": "ocupacion_od_v2" if m.linea is not None else "presion_abordaje_v1",
        "unidades": {
            "value": "subidas (validaciones) por hora en la estación",
            "crowding": ("ocupación del vehículo = pasajeros a bordo / capacidad nominal"
                         if m.linea is not None else "índice de presión de abordaje (v1)"),
            "eta_min": "espera esperada en andén (min)",
        },
        "sentido": res["sentido"], "sentidos": res["sentidos"], "referencia": res["referencia"],
    }


def _error(e: Exception):
    if isinstance(e, MS.PrediccionError):
        return jsonify({"error": str(e)}), e.status
    return jsonify({"error": f"Error interno: {e}"}), 500


def _buscar_estacion(station_id):
    objetivo = str(station_id).strip().lower()
    return next((s for s in load_all_stations_into_memory()
                 if str(s["id"]).strip().lower() == objetivo), None)


@predictions_bp.route("/estacion/<path:station_id>", methods=["GET"])
def get_station_prediction(station_id):
    """Estado actual + pronóstico del día de una estación/paradero (una sola llamada).

    Query: ``dia`` (YYYY-MM-DD, por defecto hoy en Lima), ``hour`` (0-23, por defecto la hora
    actual de Lima), ``hacia_id`` (id de la estación destino: define el sentido de viaje).
    ---
    tags:
      - Predictions
    """
    try:
        est = _buscar_estacion(station_id)
        if est is None:
            return jsonify({"error": f"Estación no encontrada: {station_id}"}), 404
        sistema, serie = serie_de_estacion(est)
        if serie is None:
            return jsonify({"error": "Sin predicción: este paradero no tiene datos de validaciones ATU",
                            "station": est["name"], "system": est["system"]}), 404
        hacia = None
        if request.args.get("hacia_id"):
            dest = _buscar_estacion(request.args["hacia_id"])
            hacia = serie_de_estacion(dest)[1] if dest else None
        # Corredores: el código del paradero ya fija ruta y sentido -> detalle del paradero
        paradero = (est.get("raw_properties") or {}).get("cod_paradero") if sistema == "corredor" else None
        res = MS.prediccion_serie(sistema, serie, request.args.get("dia"), hacia=hacia, paradero=paradero)
        hora = _hora_param()
        items = [_item(f, res["sentido"], res["sentidos"]) for f in res["filas"]]
        actual = next(i for i, f in zip(items, res["filas"]) if pd.Timestamp(f["fecha_hora"]).hour == hora)
        return jsonify({**_meta(res), "station": est["name"], "current": actual, "forecast": items}), 200
    except Exception as e:  # noqa: BLE001
        return _error(e)


@predictions_bp.route("/<system>/<series>", methods=["GET"])
def get_prediction(system, series):
    """Predicción de una hora (por defecto la hora actual de Lima).

    Query: ``dia``, ``hour``, ``sentido`` (asc|desc) o ``hacia`` (estación destino).
    """
    try:
        res = MS.prediccion_serie(system, series, request.args.get("dia"),
                                  sentido=request.args.get("sentido"), hacia=request.args.get("hacia"))
        hora = _hora_param()
        fila = next(f for f in res["filas"] if pd.Timestamp(f["fecha_hora"]).hour == hora)
        it = _item(fila, res["sentido"], res["sentidos"])
        return jsonify({
            **_meta(res),
            "datetime": it["datetime"],
            "prediction": it["value"],
            "band80": it["band80"],
            "crowding": {"index": it["crowding"], "class": it["class"],
                         "pasajeros_por_tren": it.get("pasajeros_por_tren")},
            "eta": {"next": it["eta_min"], "second": it["eta2_min"]},
            "recomendacion": it["recommendation"],
            "detalle": it,
        }), 200
    except Exception as e:  # noqa: BLE001
        return _error(e)


@predictions_bp.route("/<system>/<series>/forecast", methods=["GET"])
def get_forecast(system, series):
    """Pronóstico horario del día (``hours`` horas desde ``desde_hora``, por defecto 0-23)."""
    try:
        hours = request.args.get("hours", default=24, type=int)
        desde = request.args.get("desde_hora", default=0, type=int)
        if not 1 <= hours <= 24:
            return jsonify({"error": "hours debe estar entre 1 y 24"}), 400
        if not 0 <= desde <= 23:
            return jsonify({"error": "desde_hora debe estar entre 0 y 23"}), 400
        res = MS.prediccion_serie(system, series, request.args.get("dia"),
                                  sentido=request.args.get("sentido"), hacia=request.args.get("hacia"))
        filas = [f for f in res["filas"] if pd.Timestamp(f["fecha_hora"]).hour >= desde][:hours]
        return jsonify({**_meta(res),
                        "predictions": [_item(f, res["sentido"], res["sentidos"]) for f in filas]}), 200
    except Exception as e:  # noqa: BLE001
        return _error(e)
