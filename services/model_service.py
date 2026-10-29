"""Servicio de predicción: pronóstico de demanda (LightGBM) + ocupación a bordo (matriz O-D).

- **Modelo en producción: LightGBM** (``modo="ligero"``). En el test retenido (61 días, nov-dic
  2025) iguala o supera al ensemble con Chronos en todos los sistemas (WMAPE Metro L1 0,0996 vs
  0,1021; troncal 0,115 vs 0,123; corredores 0,113 vs 0,120), corre en milisegundos y no
  requiere torch (que no entra en la memoria del servidor). Ver ``docs/MODELO_OCUPACION.md``.
- **Día de referencia**: los datos ATU cubren 2025; un día posterior se pronostica con su día
  homólogo de 2025 (``services.calendario``) y la respuesta lo declara.
- **Caché**: modelos, contexto anual y pronósticos por día se cargan una sola vez por proceso.
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import pandas as pd

from model.inferencia import ModeloSistema
from services import calendario
from services.mapeo_estaciones import resolver_serie, series_modelo, sistema_modelo

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "context"
MODO_PRODUCCION = "ligero"
SISTEMAS = ("metro_l1", "troncal", "corredor", "alimentador")


class PrediccionError(ValueError):
    """Error de entrada (sistema, serie o fecha inválidos): la API responde 4xx."""

    def __init__(self, mensaje: str, status: int = 400):
        super().__init__(mensaje)
        self.status = status


@lru_cache(maxsize=None)
def modelo(sistema: str) -> ModeloSistema:
    return ModeloSistema.cargar(sistema, verificar_hash=False)


@lru_cache(maxsize=None)
def contexto(sistema: str) -> pd.DataFrame:
    """Validaciones horarias observadas de 2025 (hora × serie), mismas que vio el entrenamiento."""
    return pd.read_parquet(DATA_DIR / f"{sistema}_2025.parquet")


@lru_cache(maxsize=512)
def _pronostico_referencia(sistema: str, fecha_ref: str) -> pd.DataFrame:
    return modelo(sistema).pronosticar(contexto(sistema), fecha_ref, modo=MODO_PRODUCCION)


def validar_sistema(sistema: str) -> str:
    s = sistema_modelo(sistema)
    if s not in SISTEMAS:
        raise PrediccionError(f"Sistema no válido: {sistema}. Opciones: {list(SISTEMAS)}", 404)
    return s


def validar_serie(sistema: str, serie: str) -> str:
    real = resolver_serie(sistema, serie)
    if real is None:
        raise PrediccionError(
            f"'{serie}' no tiene datos de validaciones ATU en el sistema '{sistema}'; "
            f"series disponibles: {list(series_modelo(sistema))}", 404)
    return real


def parsear_dia(dia: str | None) -> dt.date:
    if dia is None or str(dia).strip() in ("", "undefined", "null", "None"):
        return calendario.ahora_lima().date()
    try:
        return dt.date.fromisoformat(str(dia).strip()[:10])
    except ValueError as exc:
        raise PrediccionError(f"Fecha inválida '{dia}': use YYYY-MM-DD") from exc


def pronostico_dia(sistema: str, fecha: dt.date) -> tuple[pd.DataFrame, calendario.Referencia]:
    """Pronóstico de las 24 h de ``fecha`` (todas las series) con la fecha objetivo real."""
    ref = calendario.dia_referencia(fecha)
    df = _pronostico_referencia(sistema, ref.referencia.isoformat()).copy()
    desfase = pd.Timestamp(fecha) - pd.Timestamp(ref.referencia)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"]) + desfase
    return df, ref


def sentido_hacia(sistema: str, origen: str, destino: str | None) -> str | None:
    """'asc' | 'desc' según la posición del destino en la línea (None si no aplica)."""
    linea = modelo(sistema).linea
    if linea is None or not destino:
        return None
    dest = resolver_serie(sistema, destino)
    if dest is None or dest not in linea.estaciones:
        raise PrediccionError(f"Estación destino desconocida: {destino}")
    if dest == origen:
        raise PrediccionError("El destino coincide con la estación de origen")
    return "asc" if linea.estaciones.index(dest) > linea.estaciones.index(origen) else "desc"


def prediccion_serie(sistema: str, serie: str, dia: str | None = None,
                     sentido: str | None = None, hacia: str | None = None) -> dict:
    """Pronóstico de un día para una serie, listo para la API."""
    sistema = validar_sistema(sistema)
    serie = validar_serie(sistema, serie)
    fecha = parsear_dia(dia)
    if hacia:
        sentido = sentido_hacia(sistema, serie, hacia)
    if sentido not in (None, "asc", "desc"):
        raise PrediccionError("sentido debe ser 'asc' o 'desc'")
    df, ref = pronostico_dia(sistema, fecha)
    filas = df[df["unidad"] == serie].sort_values("fecha_hora")
    linea = modelo(sistema).linea
    return {
        "sistema": sistema, "serie": serie, "sentido": sentido,
        "sentidos": linea.sentidos if linea is not None else None,
        "referencia": ref.to_dict(), "modelo": MODO_PRODUCCION,
        "filas": filas.to_dict(orient="records"),
    }


# ── compatibilidad con la firma anterior (lista de dicts por hora) ──
class ModelService:
    @staticmethod
    def obtener_prediccion(sistema: str, dia: str | None, modo: str = MODO_PRODUCCION,
                           unidad: str | None = None) -> list[dict]:
        if unidad is None:
            s = validar_sistema(sistema)
            return pronostico_dia(s, parsear_dia(dia))[0].to_dict(orient="records")
        return prediccion_serie(sistema, unidad, dia)["filas"]
