import os

from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flasgger import Swagger

from db.connection import init_db

from routes.auth import auth_bp
from routes.systems import systems_bp
from routes.stations import stations_bp
from routes.predictions import predictions_bp
from routes.geo import geo_bp  # <-- NUEVO: Blueprint para los GeoJSON de rutas y mapas


app = Flask(__name__)

CORS(app)

app.config["JWT_SECRET_KEY"] = os.getenv(
    "JWT_SECRET_KEY",
    "dev_secret_key"
)

jwt = JWTManager(app)

# Base de datos (si la usas para usuarios u otras entidades)
init_db(app)

# Swagger
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "UrbanSafe API",
        "description": "API backend del proyecto UrbanSafe AI",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": [
        "http"
    ]
})

# Routes / Blueprints
app.register_blueprint(
    auth_bp,
    url_prefix="/api/auth"
)

app.register_blueprint(
    systems_bp,
    url_prefix="/api/systems"
)

app.register_blueprint(
    stations_bp,
    url_prefix="/api/stations"
)

app.register_blueprint(
    predictions_bp,
    url_prefix="/api/predictions"
)

app.register_blueprint(
    geo_bp,
    url_prefix="/api/geo"  # <-- NUEVO: Ruta base para los trazados geoespaciales
)


@app.route("/api/health", methods=["GET"])
def health():
    """
    Verifica que la API esté funcionando.
    ---
    tags:
      - Health
    responses:
      200:
        description: API funcionando correctamente
    """
    return jsonify({
        "status": "ok",
        "message": "UrbanSafe API funcionando"
    }), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )