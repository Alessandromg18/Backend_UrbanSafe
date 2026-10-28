from datetime import datetime
from flask import Blueprint, jsonify, request
from services.model_service import ModelService
import pandas as pd

predictions_bp = Blueprint(
    "predictions",
    __name__
)

@predictions_bp.route("/<system>/<series>", methods=["GET"])
def get_prediction(system, series):
    modo = request.args.get("modo", "ligero")
    target_hour = request.args.get("hour", default=None, type=int)  # Parámetro opcional de hora (0-23)

    # Fecha actual de referencia (2026) y valor por defecto limpio
    hoy_str = datetime.now().strftime("%Y-%m-%d")
    dia = request.args.get("dia", hoy_str)
    if not dia or str(dia).strip() in ["", "undefined", "null", "None"]:
        dia = hoy_str

    try:
        resultados = ModelService.obtener_prediccion(system, dia, modo=modo, unidad=series)
        
        if not resultados:
            return jsonify({"error": f"No se encontró información para la serie '{series}'"}), 404

        # Si se especifica una hora, filtramos el resultado de esa hora exacta (0 a 23)
        if target_hour is not None:
            if target_hour < 0 or target_hour > 23:
                return jsonify({"error": "La hora debe estar entre 0 y 23"}), 400
            
            p = next((res for res in resultados if pd.to_datetime(res["fecha_hora"]).hour == target_hour), resultados[0])
        else:
            p = resultados[0]  # Por defecto la primera

        # 1. Ajuste de fecha al 2026 en formato corto (YYYY-MM-DD HH:MM)
        raw_dt = p.get("fecha_hora")
        if raw_dt:
            dt_parsed = pd.to_datetime(raw_dt)
            hora_del_modelo = dt_parsed.hour
            minuto_actual = datetime.now().minute
            hora_str = f"{hora_del_modelo:02d}:{minuto_actual:02d}"
            formatted_date = f"{hoy_str} {hora_str}"
        else:
            formatted_date = f"{hoy_str} 00:00"

        # 2. Redondear los valores de ETA para evitar decimales largos
        eta_next = p.get("eta_min")
        eta_second = p.get("eta2_min")
        
        rounded_eta_next = round(float(eta_next), 1) if eta_next is not None else None
        rounded_eta_second = round(float(eta_second), 1) if eta_second is not None else None

        return jsonify({
            "system": system,
            "series": series,
            "datetime": formatted_date,  # <--- Fecha limpia y en 2026
            "prediction": p.get("validaciones_pron"),
            "band80": {
                "low": p.get("banda80_lo"),
                "high": p.get("banda80_hi")
            },
            "crowding": {
                "index": p.get("crowding_index"),
                "class": p.get("aforo")
            },
            "eta": {
                "next": rounded_eta_next,       # <--- ETA redondeado
                "second": rounded_eta_second    # <--- ETA redondeado
            },
            "recomendacion": p.get("recomendacion"),
            "model": p.get("modelo", modo)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@predictions_bp.route("/<system>/<series>/forecast", methods=["GET"])
def get_forecast(system, series):
    """
    Obtiene las próximas predicciones de una serie con fechas actualizadas al 2026 y ETAs redondeados.
    """
    hours = request.args.get("hours", default=6, type=int)
    dia = request.args.get("dia")
    modo = request.args.get("modo", "ligero")

    # Fecha actual de referencia (2026)
    hoy_str = datetime.now().strftime("%Y-%m-%d")
    
    if not dia or str(dia).strip() in ["", "undefined", "null", "None"]:
        dia = hoy_str

    if hours <= 0 or hours > 24:
        return jsonify({
            "error": "hours debe estar entre 1 y 24"
        }), 400

    try:
        # Obtenemos las predicciones del modelo (procesa internamente el histórico)
        resultados = ModelService.obtener_prediccion(system, dia, modo=modo, unidad=series)
        resultados_cortados = resultados[:hours]

        predictions_list = []
        for p in resultados_cortados:
            # 1. Ajuste estricto de fecha: extraemos solo la hora del modelo y la unimos al día actual de 2026
            raw_dt = p.get("fecha_hora")
            if raw_dt:
                dt_parsed = pd.to_datetime(raw_dt)
                hora_del_modelo = dt_parsed.hour
                minuto_actual = datetime.now().minute
                hora_str = f"{hora_del_modelo:02d}:{minuto_actual:02d}"
                # Forzamos que la fecha visible sea la de hoy (2026) con la hora del pronóstico
                formatted_date = f"{hoy_str} {hora_str}"
            else:
                formatted_date = f"{hoy_str} 00:00"

            # 2. Redondear los valores de ETA para evitar decimales extensos
            eta_val = p.get("eta_min")
            eta2_val = p.get("eta2_min")
            
            rounded_eta = round(float(eta_val), 1) if eta_val is not None else None
            rounded_eta2 = round(float(eta2_val), 1) if eta2_val is not None else None

            predictions_list.append({
                "datetime": formatted_date, # Formato limpio: "2026-10-28 08:00"
                "value": p.get("validaciones_pron"),
                "crowding": p.get("crowding_index"),
                "class": p.get("aforo"),
                "eta_min": rounded_eta,
                "eta2_min": rounded_eta2,
                "recommendation": p.get("recomendacion")
            })

        return jsonify({
            "system": system,
            "series": series,
            "predictions": predictions_list,
            "model": modo
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
