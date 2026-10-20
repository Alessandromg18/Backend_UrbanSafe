from flask import Blueprint, jsonify
import json
import os

systems_bp = Blueprint("systems", __name__)

GEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/geo"))

def load_geojson_count(filename):
    """Función auxiliar para contar cuántos elementos tiene un sistema en su GeoJSON"""
    filepath = os.path.join(GEO_DIR, filename)
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return len(data.get("features", []))

@systems_bp.route("", methods=["GET"])
def get_systems():
    """
    Obtiene todos los sistemas de transporte de Lima con estadísticas 
    calculadas directamente desde los archivos GeoJSON.
    ---
    tags:
      - Systems
    responses:
      200:
        description: Lista de sistemas con metadatos y conteo de estaciones/paraderos
    """
    systems = [
        {
            "id": "metropolitano",
            "name": "Metropolitano",
            "description": "Sistema de autobuses de tránsito rápido (BRT)",
            "total_stations": load_geojson_count("puntos_troncal.geojson"),
            "has_routes": True
        },
        {
            "id": "corredores",
            "name": "Corredores Complementarios",
            "description": "Corredores viales principales (Rojo, Azul, Amarillo)",
            "total_stations": load_geojson_count("puntos_corredores.geojson"),
            "has_routes": True
        },
        {
            "id": "metro_l1",
            "name": "Metro L1",
            "description": "Línea 1 del Metro de Lima y Callao",
            "total_stations": load_geojson_count("puntos_metro_l1.geojson"),
            "has_routes": True
        },
        {
            "id": "alimentadores",
            "name": "Alimentadores",
            "description": "Rutas alimentadoras del Metropolitano",
            "total_stations": load_geojson_count("paradas_alimentadores.geojson"),
            "has_routes": True
        }
    ]

    return jsonify(systems), 200


@systems_bp.route("/<string:system_id>", methods=["GET"])
def get_system(system_id):
    """
    Obtiene la información detallada de un sistema específico por su ID.
    ---
    tags:
      - Systems
    parameters:
      - in: path
        name: system_id
        required: true
        type: string
        example: metro_l1
    responses:
      200:
        description: Sistema encontrado
      404:
        description: Sistema no encontrado
    """
    # Reutilizamos la misma estructura base
    systems_map = {
        "metropolitano": {
            "id": "metropolitano",
            "name": "Metropolitano",
            "description": "Sistema de autobuses de tránsito rápido (BRT)",
            "total_stations": load_geojson_count("puntos_troncal.geojson")
        },
        "corredores": {
            "id": "corredores",
            "name": "Corredores Complementarios",
            "description": "Corredores viales principales (Rojo, Azul, Amarillo)",
            "total_stations": load_geojson_count("puntos_corredores.geojson")
        },
        "metro_l1": {
            "id": "metro_l1",
            "name": "Metro L1",
            "description": "Línea 1 del Metro de Lima y Callao",
            "total_stations": load_geojson_count("puntos_metro_l1.geojson")
        },
        "alimentadores": {
            "id": "alimentadores",
            "name": "Alimentadores",
            "description": "Rutas alimentadoras del Metropolitano",
            "total_stations": load_geojson_count("paradas_alimentadores.geojson")
        }
    }

    system = systems_map.get(system_id)

    if not system:
        return jsonify({"error": "Sistema no encontrado"}), 404

    return jsonify(system), 200