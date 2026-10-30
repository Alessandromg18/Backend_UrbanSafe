"""Resolución paradero/estación → (sistema del modelo, serie del modelo).

Reemplaza las reglas de texto del frontend que, sin coincidencia, devolvían por defecto la serie
de otra estación (Naranjal, 301, Angamos, Estacion Central). Aquí, si un paradero no tiene datos
ATU propios, se devuelve ``None`` y la API responde "sin predicción" en lugar de inventarla.
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
GEO_DIR = BASE / "data" / "geo"

# nombre en la API/geo -> sistema del modelo
SISTEMA_MODELO = {
    "metro_l1": "metro_l1", "metrol_l1": "metro_l1", "metro_linea1": "metro_l1",
    "metropolitano": "troncal", "troncal": "troncal",
    "corredores": "corredor", "corredor": "corredor",
    "alimentadores": "alimentador", "alimentador": "alimentador",
}


def norm(texto) -> str:
    s = unicodedata.normalize("NFKD", str(texto or ""))
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).lower().split())


def sistema_modelo(sistema: str) -> str | None:
    return SISTEMA_MODELO.get(norm(sistema).replace(" ", "_"))


@lru_cache(maxsize=None)
def series_modelo(sistema: str) -> tuple[str, ...]:
    card = json.loads((MODELS_DIR / sistema / "model_card.json").read_text(encoding="utf-8"))
    return tuple(card["series"])


@lru_cache(maxsize=None)
def _indice_series(sistema: str) -> dict[str, str]:
    return {norm(s): s for s in series_modelo(sistema)}


def resolver_serie(sistema: str, texto: str) -> str | None:
    """Serie exacta del modelo para ``texto`` (ignora tildes, mayúsculas y espacios)."""
    return _indice_series(sistema).get(norm(texto))


@lru_cache(maxsize=None)
def _mapeo_alimentadores() -> dict[str, str | None]:
    d = json.loads((GEO_DIR / "mapeo_alimentadores.json").read_text(encoding="utf-8"))
    return d["servicio_a_serie"]


def serie_de_estacion(estacion: dict) -> tuple[str | None, str | None]:
    """(sistema_modelo, serie) de una estación de ``routes.stations``; serie None si no hay datos."""
    sistema = sistema_modelo(estacion.get("system", ""))
    if sistema is None:
        return None, None
    props = estacion.get("raw_properties") or {}
    if sistema in ("metro_l1", "troncal"):
        return sistema, resolver_serie(sistema, props.get("unidad") or estacion.get("name"))
    if sistema == "corredor":
        # El modelo de corredores es por ruta: el paradero hereda la serie de su ruta
        return sistema, resolver_serie(sistema, props.get("ruta"))
    serie = _mapeo_alimentadores().get(props.get("servicio"))
    return sistema, (resolver_serie(sistema, serie) if serie else None)
