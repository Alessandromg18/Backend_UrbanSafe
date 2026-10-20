import bcrypt

from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity
)

from db.connection import db


auth_bp = Blueprint("auth", __name__)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Registra un nuevo usuario.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
            - email
            - password
          properties:
            name:
              type: string
              example: Alessandro
            email:
              type: string
              example: alessandro@gmail.com
            password:
              type: string
              example: 123456
    responses:
      201:
        description: Usuario registrado correctamente
      400:
        description: Datos incompletos
      409:
        description: El correo ya está registrado
    """

    data = request.get_json()

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({
            "error": "name, email y password son obligatorios"
        }), 400

    email = email.lower().strip()

    existing_user = User.query.filter_by(
        email=email
    ).first()

    if existing_user:
        return jsonify({
            "error": "El correo ya está registrado"
        }), 409

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    user = User(
        name=name,
        email=email,
        password_hash=password_hash
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({
        "message": "Usuario registrado correctamente",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Inicia sesión y obtiene un JWT.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - email
            - password
          properties:
            email:
              type: string
              example: alessandro@gmail.com
            password:
              type: string
              example: 123456
    responses:
      200:
        description: Login exitoso
      400:
        description: Datos incompletos
      401:
        description: Credenciales incorrectas
    """

    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({
            "error": "email y password son obligatorios"
        }), 400

    email = email.lower().strip()

    user = User.query.filter_by(
        email=email
    ).first()

    if not user:
        return jsonify({
            "error": "Credenciales incorrectas"
        }), 401

    valid_password = bcrypt.checkpw(
        password.encode("utf-8"),
        user.password_hash.encode("utf-8")
    )

    if not valid_password:
        return jsonify({
            "error": "Credenciales incorrectas"
        }), 401

    access_token = create_access_token(
        identity=str(user.id)
    )

    return jsonify({
        "message": "Login exitoso",
        "access_token": access_token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def get_current_user():
    """
    Obtiene los datos del usuario autenticado.
    ---
    tags:
      - Auth
    securityDefinitions:
      - Bearer:
          type: apiKey
          name: Authorization
          in: header
    security:
      - Bearer: []
    responses:
      200:
        description: Usuario actual
      401:
        description: Token inválido o ausente
      404:
        description: Usuario no encontrado
    """

    user_id = get_jwt_identity()

    user = User.query.get(user_id)

    if not user:
        return jsonify({
            "error": "Usuario no encontrado"
        }), 404

    return jsonify({
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at
    }), 200

@auth_bp.route("/me", methods=["PUT"])
@jwt_required()
def update_profile():
    """
    Actualiza el perfil del usuario autenticado.
    ---
    tags:
      - Auth
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
              example: Alessandro Matius
            email:
              type: string
              example: alessandro@example.com
    responses:
      200:
        description: Perfil actualizado correctamente
      400:
        description: Datos inválidos
      409:
        description: El email ya está registrado
      404:
        description: Usuario no encontrado
    """

    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))

    if not user:
        return jsonify({
            "error": "Usuario no encontrado"
        }), 404

    data = request.get_json() or {}

    name = data.get("name")
    email = data.get("email")

    if not name and not email:
        return jsonify({
            "error": "Debe enviar al menos name o email"
        }), 400

    if name:
        user.name = name

    if email and email != user.email:

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            return jsonify({
                "error": "El email ya está registrado"
            }), 409

        user.email = email

    db.session.commit()

    return jsonify({
        "message": "Perfil actualizado correctamente",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "created_at": user.created_at
        }
    }), 200

@auth_bp.route("/me/password", methods=["PUT"])
@jwt_required()
def update_password():
    """
    Cambia la contraseña del usuario autenticado.
    ---
    tags:
      - Auth
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - current_password
            - new_password
          properties:
            current_password:
              type: string
              example: password123
            new_password:
              type: string
              example: nuevaPassword456
    responses:
      200:
        description: Contraseña actualizada correctamente
      400:
        description: Datos incompletos
      401:
        description: Contraseña actual incorrecta
      404:
        description: Usuario no encontrado
    """

    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))

    if not user:
        return jsonify({
            "error": "Usuario no encontrado"
        }), 404

    data = request.get_json() or {}

    current_password = data.get("current_password")
    new_password = data.get("new_password")

    if not current_password or not new_password:
        return jsonify({
            "error": "current_password y new_password son obligatorios"
        }), 400

    if not bcrypt.checkpw(
        current_password.encode("utf-8"),
        user.password_hash.encode("utf-8")
    ):
        return jsonify({
            "error": "La contraseña actual es incorrecta"
        }), 401

    if len(new_password) < 6:
        return jsonify({
            "error": "La nueva contraseña debe tener al menos 6 caracteres"
        }), 400

    hashed_password = bcrypt.hashpw(
        new_password.encode("utf-8"),
        bcrypt.gensalt()
    )

    user.password_hash = hashed_password.decode("utf-8")

    db.session.commit()

    return jsonify({
        "message": "Contraseña actualizada correctamente"
    }), 200


@auth_bp.route("/me", methods=["DELETE"])
@jwt_required()
def delete_account():
    """
    Elimina la cuenta del usuario autenticado.
    ---
    tags:
      - Auth
    security:
      - Bearer: []
    responses:
      200:
        description: Cuenta eliminada correctamente
      404:
        description: Usuario no encontrado
    """

    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))

    if not user:
        return jsonify({
            "error": "Usuario no encontrado"
        }), 404

    db.session.delete(user)
    db.session.commit()

    return jsonify({
        "message": "Cuenta eliminada correctamente"
    }), 200

