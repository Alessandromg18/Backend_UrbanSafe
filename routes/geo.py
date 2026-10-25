from flask import Blueprint, jsonify, request, send_from_directory  # <--- Agregado send_from_directory aquí
import os
import json
import math
from routes.stations import load_all_stations_into_memory
import urllib.request
import urllib.parse

geo_bp = Blueprint("geo", __name__)
GEO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/geo"))

@geo_bp.route("/routes/<system_name>", methods=["GET"])
def get_route_geojson(system_name):
    file_map = {
        "corredores": "trazados_corredores.geojson",
        "corredor": "trazados_corredores.geojson",
        "metro_l1": "trazado_metro_l1.geojson",
        "troncal": "trazado_troncal.geojson",
        "troncales": "trazado_troncal.geojson",
        "alimentadores": "trazados_alimentadores_atu.geojson",
        "alimentador": "trazados_alimentadores_atu.geojson",
        "metropolitano": "trazados_metropolitano_servicios.geojson"
    }

    filename = file_map.get(system_name)
    if not filename:
        return jsonify({"error": f"Sistema no válido: {system_name}"}), 400
    
    filepath = os.path.join(GEO_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "Archivo de trazado no encontrado"}), 404

    route_code = request.args.get("code")

    if not route_code:
        # Aquí es donde fallaba porque send_from_directory no estaba importado
        return send_from_directory(GEO_DIR, filename)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        filtered_features = [
            feature for feature in geojson_data.get("features", [])
            if str(feature.get("properties", {}).get("code")) == str(route_code) or
               str(feature.get("properties", {}).get("ruta")) == str(route_code) or
               str(feature.get("properties", {}).get("osm_ref")) == str(route_code)
        ]

        filtered_geojson = {
            "type": "FeatureCollection",
            "name": geojson_data.get("name", system_name),
            "features": filtered_features
        }

        return jsonify(filtered_geojson), 200

    except Exception as e:
        return jsonify({"error": f"Error al procesar el archivo geojson: {str(e)}"}), 500

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Radio de la Tierra en metros
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def geocode_destination(query):
    """Consulta OSM Nominatim con reintentos automáticos si la primera búsqueda exacta falla"""
    base_url = "https://nominatim.openstreetmap.org/search"
    
    # Creamos una lista de variaciones para intentar encontrar el lugar
    clean_q = query.strip()
    queries_to_try = [
        f"{clean_q}, Lima, Peru",
        f"{clean_q}, Lima"
    ]
    
    # Si incluye palabras clave específicas, agregamos alternativas simplificadas
    lower_q = clean_q.lower()
    if "metropolitano" in lower_q:
        simpler = lower_q.replace("metropolitano", "").strip()
        if simpler:
            queries_to_try.append(f"{simpler}, Lima, Peru")
    if "estacion" in lower_q:
        simpler = lower_q.replace("estacion", "").strip()
        if simpler:
            queries_to_try.append(f"{simpler}, Lima, Peru")

    for q_str in queries_to_try:
        try:
            params = {
                "q": q_str,
                "format": "json",
                "limit": 1,
                "countrycodes": "pe"
            }
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'UrbanSafe-Backend/1.0'}
            )
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                if data and len(data) > 0:
                    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name")
        except Exception as e:
            print(f"Error en intento con '{q_str}': {e}")
            
    return None, None, None

@geo_bp.route("/match-trip", methods=["GET"])
def match_trip():
    """
    Planificador con datos de diagnóstico detallados para depurar fallos y rutas ordenadas por caminata.
    """
    try:
        origin_lat = request.args.get("origin_lat", type=float)
        origin_lon = request.args.get("origin_lon", type=float)
        
        dest_query = request.args.get("destination", type=str)
        dest_lat = request.args.get("dest_lat", type=float)
        dest_lon = request.args.get("dest_lon", type=float)
        
        radius = request.args.get("radius", default=500, type=float)

        if origin_lat is None or origin_lon is None:
            return jsonify({"error": "Se requieren 'origin_lat' y 'origin_lon' de tu ubicación actual"}), 400

        dest_name = "Coordenadas directas"
        
        if dest_query:
            d_lat, d_lon, d_display = geocode_destination(dest_query)
            if d_lat is None or d_lon is None:
                return jsonify({
                    "error": f"No se pudo encontrar '{dest_query}' en Lima usando OpenStreetMap.",
                    "debug_tip": "Intenta escribir el nombre con más detalles (ej. 'Centro Cívico Lima' o 'Parque Kennedy Miraflores')."
                }), 404
            dest_lat, dest_lon = d_lat, d_lon
            dest_name = d_display
        elif dest_lat is None or dest_lon is None:
            return jsonify({"error": "Debes proporcionar 'destination' (texto) o 'dest_lat'/'dest_lon'"}), 400

        # Cargamos estaciones en memoria
        all_stations = load_all_stations_into_memory()

        origin_stations = []
        destination_stations = []

        # Barrido de proximidad
        for station in all_stations:
            st_lat = station.get("latitude")
            st_lon = station.get("longitude")
            if st_lat is None or st_lon is None:
                continue

            # Distancia al origen
            dist_orig = calculate_distance(origin_lat, origin_lon, st_lat, st_lon)
            if dist_orig <= radius:
                st_o = station.copy()
                st_o["origin_distance_meters"] = round(dist_orig, 1) # 👈 Clave diferenciada
                origin_stations.append(st_o)

            # Distancia al destino
            dist_dest = calculate_distance(dest_lat, dest_lon, st_lat, st_lon)
            if dist_dest <= radius:
                st_d = station.copy()
                st_d["dest_distance_meters"] = round(dist_dest, 1) # 👈 Clave diferenciada
                destination_stations.append(st_d)

        # --- SECCIÓN DE DIAGNÓSTICO (DEBUG) ---
        debug_info = {
            "origin_stations_found": len(origin_stations),
            "destination_stations_found": len(destination_stations),
        }

        if len(origin_stations) == 0:
            debug_info["warning"] = "No se encontraron paraderos cerca de tu ORIGEN con el radio especificado."
        elif len(destination_stations) == 0:
            debug_info["warning"] = "No se encontraron paraderos cerca de tu DESTINO con el radio especificado."

        # Cruzar información para rutas directas
        direct_routes = []
        for o_st in origin_stations:
            o_route = o_st.get("raw_properties", {}).get("ruta") or o_st.get("raw_properties", {}).get("servicio")
            o_system = o_st.get("system")

            for d_st in destination_stations:
                d_route = d_st.get("raw_properties", {}).get("ruta") or d_st.get("raw_properties", {}).get("servicio")
                d_system = d_st.get("system")

                # 1. Match por ruta específica (Corredores / alimentadores)
                if o_route and d_route and str(o_route) == str(d_route):
                    direct_routes.append({
                        "type": "DIRECTO_POR_RUTA",
                        "route_code": str(o_route),
                        "system": o_system,
                        "boarding_station": o_st.get("name"),
                        "boarding_id": o_st.get("id"),
                        "origin_walk_m": o_st.get("origin_distance_meters"),
                        "leaving_station": d_st.get("name"),
                        "leaving_id": d_st.get("id"),
                        "dest_walk_m": d_st.get("dest_distance_meters") # 👈 Corregido con d_st
                    })
                
                # 2. Match por sistema troncal (Metropolitano o Metro L1)
                elif o_system == d_system and o_system in ["metropolitano", "metro_l1"]:
                    direct_routes.append({
                        "type": "DIRECTO_POR_SISTEMA",
                        "route_code": o_system.upper(),
                        "system": o_system,
                        "boarding_station": o_st.get("name"),
                        "boarding_id": o_st.get("id"),
                        "origin_walk_m": o_st.get("origin_distance_meters"),
                        "leaving_station": d_st.get("name"),
                        "leaving_id": d_st.get("id"),
                        "dest_walk_m": d_st.get("dest_distance_meters") # 👈 Corregido con d_st
                    })

        # Filtrar duplicados
        seen = set()
        unique_routes = []
        for r in direct_routes:
            key = (r["route_code"], r["boarding_id"], r["leaving_id"])
            if key not in seen:
                seen.add(key)
                unique_routes.append(r)

        # Ordenar de menor a mayor distancia total de caminata
        unique_routes.sort(key=lambda x: x["origin_walk_m"] + x["dest_walk_m"])

        if len(unique_routes) == 0 and "warning" not in debug_info:
            debug_info["warning"] = "Se encontraron paraderos en ambos extremos, pero ninguna ruta directa conecta ambos puntos sin transbordo."

        return jsonify({
            "origin": {"lat": origin_lat, "lon": origin_lon},
            "destination": {"lat": dest_lat, "lon": dest_lon, "resolved_name": dest_name},
            "radius_meters": radius,
            "diagnostic": debug_info,
            "found_routes": unique_routes
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error al procesar el match de viaje: {str(e)}"}), 500

@geo_bp.route("/autocomplete", methods=["GET"])
def autocomplete_locations():
    """Sugiere ubicaciones en Lima en tiempo real mientras el usuario escribe"""
    query = request.args.get("q", type=str)
    if not query or len(query.strip()) < 3:
        return jsonify([]), 200

    try:
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{query}, Lima, Peru",
            "format": "json",
            "limit": 5,
            "countrycodes": "pe",
            "addressdetails": 1
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'UrbanSafe-Backend/1.0'})
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            suggestions = []
            for item in data:
                suggestions.append({
                    "display_name": item.get("display_name"),
                    "lat": float(item.get("lat")),
                    "lon": float(item.get("lon"))
                })
            return jsonify(suggestions), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500