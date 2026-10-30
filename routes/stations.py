from flask import Blueprint, request, jsonify
import json
import os
import unicodedata

stations_bp = Blueprint("stations", __name__)

# Ruta base donde se encuentran los archivos GeoJSON
GEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/geo"))

# Caché en memoria para evitar latencia de lectura de disco en cada request
_STATIONS_CACHE = None

def normalize_text(text):
    """
    Función para limpiar tildes, diacríticos y convertir a minúsculas
    para lograr una búsqueda tolerante a errores ortográficos comunes.
    """
    if not text:
        return ""
    # Normaliza caracteres Unicode y elimina tildes
    nfkd_form = unicodedata.normalize('NFKD', str(text))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()

def load_all_stations_into_memory():
    """
    Carga y unifica todos los geojson en memoria RAM una sola vez.
    Mapea las propiedades específicas de cada archivo de transporte de Lima.
    """
    global _STATIONS_CACHE
    if _STATIONS_CACHE is not None:
        return _STATIONS_CACHE

    stations = []
    
    files_mapping = [
        ("puntos_troncal.geojson", "metropolitano"),
        ("puntos_corredores.geojson", "corredores"),
        ("puntos_metro_l1.geojson", "metro_l1"),
        ("paradas_alimentadores.geojson", "alimentadores")
    ]

    for filename, system_name in files_mapping:
        filepath = os.path.join(GEO_DIR, filename)
        if not os.path.exists(filepath):
            continue
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for index, feat in enumerate(data.get("features", [])):
                    props = feat.get("properties", {})
                    geom = feat.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    
                    if not coords or len(coords) < 2:
                        continue
                    
                    # Normalización de campos según el origen del GeoJSON
                    station_id = (
                        props.get("estacion") or 
                        props.get("cod_paradero") or 
                        props.get("codigo") or 
                        f"{system_name}_{index}"
                    )
                    
                    name = (
                        props.get("paradero") or 
                        props.get("estacion") or 
                        props.get("unidad") or 
                        "Paradero sin nombre"
                    )
                    
                    avenida = (
                        props.get("osm_name") or 
                        props.get("ruta") or 
                        props.get("servicio") or 
                        ""
                    )

                    stations.append({
                        "id": str(station_id),
                        "name": str(name).strip(),
                        "system": system_name,
                        "avenida": str(avenida).strip(),
                        "latitude": float(coords[1]),
                        "longitude": float(coords[0]),
                        "raw_properties": props
                    })
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")

    # Serie del modelo de cada estación (None = sin datos ATU propios: la app no debe inventarla)
    from services.mapeo_estaciones import serie_de_estacion
    for s in stations:
        s["model_system"], s["model_series"] = serie_de_estacion(s)

    _STATIONS_CACHE = stations
    return _STATIONS_CACHE


@stations_bp.route("", methods=["GET"])
def get_stations():
    """
    Obtiene las estaciones con opción de filtrar por sistema o buscar por texto/avenida de forma flexible.
    Parámetros query:
      - system: Filtrar por ID de sistema (metropolitano, corredores, metro_l1, alimentadores)
      - q: Texto para buscar coincidencias (ignora tildes, mayúsculas y minúsculas).
    ---
    tags:
      - Stations
    """
    system_filter = request.args.get("system", type=str)
    query_text = request.args.get("q", type=str)

    stations = load_all_stations_into_memory()

    if system_filter:
        stations = [s for s in stations if s["system"] == system_filter]

    if query_text:
        q_normalized = normalize_text(query_text)
        filtered_stations = []
        
        for s in stations:
            # Comparamos nombre y avenida normalizados
            name_norm = normalize_text(s["name"])
            avenida_norm = normalize_text(s["avenida"])
            
            if q_normalized in name_norm or q_normalized in avenida_norm:
                filtered_stations.append(s)
                
        stations = filtered_stations

    return jsonify(stations), 200


@stations_bp.route("/<path:station_id>", methods=["GET"])
def get_station(station_id):
    """
    Obtiene el detalle de una estación específica por su ID de forma flexible.
    Usamos <path:station_id> en lugar de <string:station_id> por si el ID incluye barras u otros caracteres.
    """
    stations = load_all_stations_into_memory()
    
    # Limpiamos el ID que viene en la URL para buscarlo de forma segura
    clean_target_id = station_id.strip().lower()

    # Buscamos haciendo match ignorando mayúsculas/minúsculas y espacios
    station = next(
        (s for s in stations if str(s.get("id", "")).strip().lower() == clean_target_id), 
        None
    )

    if not station:
        return jsonify({"error": f"Estación no encontrada: {station_id}"}), 404

    return jsonify(station), 200

@stations_bp.route("/nearby", methods=["GET"])
def get_nearby_stations():
    """
    Busca estaciones cercanas por geolocalización optimizada en memoria.
    Parámetros query:
      - latitude: latitud del usuario (ej: -12.085)
      - longitude: longitud del usuario (ej: -77.035)
      - radius: radio en metros (por defecto 1500m)
    ---
    tags:
      - Stations
    """
    latitude = request.args.get("latitude", type=float)
    longitude = request.args.get("longitude", type=float)
    radius = request.args.get("radius", default=1500, type=float)

    if latitude is None or longitude is None:
        return jsonify({"error": "latitude y longitude son obligatorios"}), 400

    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return jsonify({"error": "Coordenadas geográficas inválidas"}), 400

    if radius <= 0:
        return jsonify({"error": "radius debe ser mayor que 0"}), 400

    stations = load_all_stations_into_memory()
    nearby = []

    for station in stations:
        lat_diff = station["latitude"] - latitude
        lon_diff = station["longitude"] - longitude
        
        # Distancia euclidiana rápida convertida a metros de forma aproximada
        distance = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111320

        if distance <= radius:
            station_copy = station.copy()
            station_copy["distance_m"] = round(distance, 2)
            nearby.append(station_copy)

    # Ordenar del más cercano al más lejano para dar prioridad al usuario móvil
    nearby.sort(key=lambda x: x["distance_m"])

    return jsonify(nearby), 200