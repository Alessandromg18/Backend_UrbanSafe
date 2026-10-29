"""Ocupación a bordo por estación, hora y sentido a partir de las subidas pronosticadas.

Cadena: subidas O (pronóstico ML) → bajadas D (viaje de retorno, ``od_matrix.atracciones_espejo``)
→ matriz O-D horaria (gravedad doblemente acotada / Furness) → carga por tramo → pasajeros por
tren (oferta oficial: trenes por hora) → clase de aforo por densidad de pie → espera esperada.

La configuración de cada línea vive en ``models/<sistema>/linea_config.json`` (geometría,
horario, vehículo, parámetros O-D y fuentes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import od_matrix as OD

SIN_SERVICIO = "Sin_servicio"


@dataclass
class ResultadoLinea:
    """Arreglos (S, 24) en el orden de ``series`` pedido; asc/desc según el orden de la línea."""

    bajadas: np.ndarray
    carga: dict            # sentido -> pasajeros por hora a bordo al salir de la estación
    carga_tren: dict       # sentido -> pasajeros por tren al salir de la estación
    ocupacion: dict        # sentido -> carga_tren / capacidad (puede superar 1: demanda > oferta)
    clase: dict            # sentido -> Asientos | De_pie | Saturado | Sin_servicio
    espera_min: dict       # sentido -> espera esperada en andén (min)
    intervalo_min: np.ndarray  # (24,)
    od: np.ndarray         # (24, S, S) en el orden de la línea


class LineaOD:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sistema = cfg["sistema"]
        self.estaciones = list(cfg["estaciones"])
        km = np.asarray(cfg["km"], dtype=float)
        v = float(cfg["velocidad_comercial_kmh"])
        self.C = np.abs(km[:, None] - km[None, :]) / v * 60.0        # minutos en vehículo
        self.mascara = ~np.eye(len(self.estaciones), dtype=bool)      # nadie baja donde sube
        od = cfg["od"]
        self.beta = float(od["beta"])
        self.n = float(od.get("n", 0.0))
        self.bloques = {k: list(v) for k, v in od["bloques"].items()}
        self.espejo = dict(od["espejo"])
        veh = cfg["vehiculo"]
        self.capacidad = float(veh["capacidad"])
        self.asientos = float(veh["asientos"])
        self.umbral_saturado = float(veh["umbral_saturado_pasajeros"])
        self.trenes_hora = {k: np.asarray(v, dtype=float) for k, v in cfg["oferta"]["trenes_hora"].items()}
        self.sentidos = dict(cfg["sentidos"])

    @classmethod
    def cargar(cls, carpeta: Path) -> "LineaOD | None":
        ruta = Path(carpeta) / "linea_config.json"
        if not ruta.exists():
            return None
        return cls(json.loads(ruta.read_text(encoding="utf-8")))

    def clases(self, carga_tren: np.ndarray, servicio: np.ndarray) -> np.ndarray:
        c = np.where(carga_tren > self.umbral_saturado, "Saturado",
                     np.where(carga_tren > self.asientos, "De_pie", "Asientos")).astype(object)
        c[~servicio] = SIN_SERVICIO
        return c

    def aplicar(self, Y: np.ndarray, series: list, bucket: str) -> ResultadoLinea:
        """``Y`` (S, 24): subidas por hora de las ``series`` (pronóstico u observación)."""
        idx = [list(series).index(e) for e in self.estaciones]
        O = np.clip(np.asarray(Y, dtype=float)[idx], 0, None)
        D = OD.atracciones_espejo(O, self.bloques, self.espejo)
        T = OD.od_dia(O, D, self.C, self.beta, self.mascara, self.n)
        asc, desc = OD.carga_al_salir(T)                        # (24, S)
        th = self.trenes_hora.get(bucket, self.trenes_hora["LAB"])
        servicio_h = th > 0
        hw = np.where(servicio_h, 60.0 / np.where(servicio_h, th, 1.0), np.nan)
        inv = np.argsort(idx)                                   # orden de la línea -> orden de series

        res = {"carga": {}, "carga_tren": {}, "ocupacion": {}, "clase": {}, "espera_min": {}}
        for sentido, L in (("asc", asc.T), ("desc", desc.T)):   # (S, 24)
            por_tren = np.where(servicio_h, L / np.where(servicio_h, th, 1.0), 0.0)
            # terminal de llegada en ese sentido: no hay salida
            sin_salida = np.zeros_like(L, dtype=bool)
            sin_salida[-1 if sentido == "asc" else 0] = True
            servicio = servicio_h[None, :] & ~sin_salida
            # Espera esperada con llegada aleatoria y servicio regular: h/2 (Ortúzar ec. 10.19,
            # sigma = 0). Si la demanda supera la capacidad, la fracción que no cabe espera
            # trenes adicionales: + h · min(1, exceso / subidas por tren en la estación).
            sube_tren = np.where(servicio_h, O / np.where(servicio_h, th, 1.0), 0.0)
            exceso = np.clip(por_tren - self.capacidad, 0, None)
            p_no_cabe = np.divide(exceso, sube_tren, out=np.zeros_like(exceso), where=sube_tren > 0)
            espera = hw[None, :] / 2.0 + hw[None, :] * np.clip(p_no_cabe, 0, 1)
            res["carga"][sentido] = L[inv]
            res["carga_tren"][sentido] = por_tren[inv]
            res["ocupacion"][sentido] = (por_tren / self.capacidad)[inv]
            res["clase"][sentido] = self.clases(por_tren, servicio)[inv]
            res["espera_min"][sentido] = np.where(servicio, espera, np.nan)[inv]
        return ResultadoLinea(bajadas=D[inv], intervalo_min=hw, od=T, **res)
