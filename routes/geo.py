from flask import Blueprint, jsonify, send_from_directory
import os

geo_bp = Blueprint("geo", __name__)
GEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/geo"))

@geo_bp.route("/routes/<system_name>", methods=["GET"])
def get_route_geojson(system_name):
    """
    Devuelve las polilíneas de trazados para el mapa móvil.
    Sistemas: metropolitano, corredores, metro_l1, alimentadores
    """
    file_map = {
        "corredores": "trazados_corredores.geojson",
        "metro_l1": "trazado_metro_l1.geojson",
        "troncal": "trazado_troncal.geojson",
        "alimentadores": "trazados_alimentadores_atu.geojson",
        "metropolitano": "trazados_metropolitano_servicios.geojson"
    }

    filename = file_map.get(system_name)
    if not filename:
        return jsonify({"error": "Sistema no válido"}), 400

    filepath = os.path.join(GEO_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Archivo de trazado no encontrado"}), 404

    return send_from_directory(GEO_DIR, filename)