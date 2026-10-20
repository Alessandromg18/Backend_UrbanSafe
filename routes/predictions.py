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
    dia = request.args.get("dia", "2025-12-31")
    modo = request.args.get("modo", "ligero")
    target_hour = request.args.get("hour", default=None, type=int)  # <-- Nuevo parámetro opcional de hora (0-23)

    try:
        resultados = ModelService.obtener_prediccion(system, dia, modo=modo, unidad=series)
        
        if not resultados:
            return jsonify({"error": f"No se encontró información para la serie '{series}'"}), 404

        # Si se especifica una hora, filtramos el resultado de esa hora exacta (0 a 23)
        if target_hour is not None:
            if target_hour < 0 or target_hour > 23:
                return jsonify({"error": "La hora debe estar entre 0 y 23"}), 400
            
            # Buscamos el registro que coincida con la hora
            p = next((res for res in resultados if pd.to_datetime(res["fecha_hora"]).hour == target_hour), resultados[0])
        else:
            p = resultados[0]  # Por defecto la primera

        return jsonify({
            "system": system,
            "series": series,
            "datetime": p.get("fecha_hora"),
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
                "next": p.get("eta_min"),
                "second": p.get("eta2_min")
            },
            "recomendacion": p.get("recomendacion"),
            "model": p.get("modelo", modo)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@predictions_bp.route("/<system>/<series>/forecast", methods=["GET"])
def get_forecast(system, series):
    """
    Obtiene las próximas predicciones de una serie.
    ---
    tags:
      - Predictions
    parameters:
      - in: path
        name: system
        required: true
        type: string
        example: metro_l1
      - in: path
        name: series
        required: true
        type: string
        example: Gamarra
      - in: query
        name: hours
        required: false
        type: integer
        example: 6
        description: Cantidad de horas a predecir (máximo 24)
      - in: query
        name: dia
        required: false
        type: string
        description: Fecha objetivo en formato YYYY-MM-DD
      - in: query
        name: modo
        required: false
        type: string
        description: Modo de inferencia (por defecto ligero)
    responses:
      200:
        description: Pronóstico obtenido correctamente
    """
    hours = request.args.get("hours", default=6, type=int)
    dia = request.args.get("dia", "2025-12-31")
    modo = request.args.get("modo", "ligero")

    if hours <= 0 or hours > 24:
        return jsonify({
            "error": "hours debe estar entre 1 y 24"
        }), 400

    try:
        # Obtenemos las 24 horas completas del modelo
        resultados = ModelService.obtener_prediccion(system, dia, modo=modo, unidad=series)
        
        # Limitamos la respuesta al número de horas solicitadas
        resultados_cortados = resultados[:hours]

        predictions_list = []
        for p in resultados_cortados:
            predictions_list.append({
                "datetime": p.get("fecha_hora"),
                "value": p.get("validaciones_pron"),
                "crowding": p.get("crowding_index"),
                "class": p.get("aforo"),  # Añadido de forma consistente por si el frontend lo requiere en el forecast
                "eta_min": p.get("eta_min"),
                "eta2_min": p.get("eta2_min"),
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