"""Recomendación (abordar / esperar el siguiente / cambiar de paradero) por costo generalizado.

La regla U de la v1 comparaba la clase de la hora actual con la de la hora siguiente. Con trenes
cada 3 minutos el siguiente vehículo cae en la misma hora, así que decía "abordar" en el 95-99 % de
las horas. Ahora cada opción se valora en **minutos equivalentes** (Ortúzar y Willumsen, §10.6.2.4,
ec. 10.18):

    costo = A_CAMINATA · caminata + A_ESPERA · espera + multiplicador(hacinamiento) · viaje

- A_ESPERA = A_CAMINATA = 2: el libro indica que el usuario percibe el tiempo de espera y de
  caminata "dos o tres veces" más que el tiempo a bordo (se usa el extremo inferior).
- Multiplicador de hacinamiento sobre el tiempo a bordo (Whelan y Crockett, citados en Li y
  Hensher, 2013, *Crowding in Public Transport*): sentado sube de 1,00 a 1,63 y de pie de 1,53 a
  2,04 cuando la densidad de pie pasa de 0 a 6 pasajeros/m².

Opciones:
1. **Siguiente vehículo:** si es probable que el primer vehículo llegue sin espacio para subir
   (P(no subir) ≥ 0,5), "esperar el siguiente".
2. **Paradero cercano (corredores):** subir en un paradero anterior de la misma ruta y sentido, a
   menos de 600 m, donde el bus viene más vacío. Se recomienda "cambiar de paradero" si ahorra al
   menos 3 minutos equivalentes, incluida la caminata.
3. En otro caso, "abordar".
"""
from __future__ import annotations

import numpy as np

A_ESPERA = 2.0
A_CAMINATA = 2.0
VELOCIDAD_PIE_M_MIN = 75.0          # 4,5 km/h
P_NO_SUBIR_ESPERAR = 0.5
AHORRO_MINIMO_MIN = 3.0
CAMINATA_MAX_M = 600.0
SIN_SERVICIO = "Sin_servicio"


def multiplicador(carga: np.ndarray, asientos: float, area_pie: float) -> np.ndarray:
    """Multiplicador del tiempo a bordo según la carga del vehículo al subir.

    Si la carga no supera los asientos, el pasajero viaja sentado (1,00 a 1,63 según la densidad
    de pie, que aquí es 0). Si la supera, viaja de pie: 1,53 + 0,085 · densidad (tope 6 pax/m²).
    """
    carga = np.asarray(carga, dtype=float)
    dens = np.clip((carga - asientos) / max(area_pie, 1e-9), 0.0, 6.0)
    sentado = 1.0 + (1.63 - 1.0) * dens / 6.0
    de_pie = 1.53 + (2.04 - 1.53) * dens / 6.0
    return np.where(carga <= asientos, sentado, de_pie)


def costo(espera: np.ndarray, viaje: np.ndarray, mult: np.ndarray, caminata_m=0.0) -> np.ndarray:
    """Costo generalizado en minutos equivalentes (ec. 10.18)."""
    return (A_CAMINATA * np.asarray(caminata_m) / VELOCIDAD_PIE_M_MIN
            + A_ESPERA * np.asarray(espera) + np.asarray(mult) * np.asarray(viaje))


def decidir(clase: np.ndarray, p_no_subir: np.ndarray, costo_aqui: np.ndarray | None = None,
            costo_alternativa: np.ndarray | None = None) -> np.ndarray:
    clase = np.asarray(clase)
    servicio = clase != SIN_SERVICIO
    rec = np.where(servicio, "abordar", "sin_servicio").astype(object)
    rec = np.where(servicio & (np.nan_to_num(p_no_subir) >= P_NO_SUBIR_ESPERAR), "esperar_siguiente", rec)
    if costo_alternativa is not None and costo_aqui is not None:
        mejor = servicio & np.isfinite(costo_alternativa) & (costo_alternativa <= costo_aqui - AHORRO_MINIMO_MIN)
        rec = np.where(mejor, "cambiar_paradero", rec)
    return rec
