"""Matrices origen-destino y perfiles de carga a partir de subidas (validaciones).

La ATU solo entrega subidas por estación y hora: no hay bajadas. Este módulo reconstruye la
matriz O-D de una línea y, con ella, cuántos pasajeros viajan a bordo en cada tramo.

Referencia: Ortúzar, J. de D. y Willumsen, L. G. (2008). *Modelos de Transporte*.
PUbliCan, Universidad de Cantabria.

- §5.2.3, ec. (5.6)-(5.7): método de Furness (bi-proporcional) con factores a_i, b_j.
- §5.3.2, ec. (5.14)-(5.18): modelo gravitacional doblemente acotado T_ij = a_i b_j f(c_ij).
- §5.3.1, ec. (5.11)-(5.13): funciones de fricción exponencial, potencial y combinada.
- §5.5.2, ec. (5.42): calibración de Hyman (beta tal que el coste medio modelado = c*).
- §12.4.4, ec. (12.21)-(12.23): ME2; sin matriz a priori, la más probable es la uniforme,
  y Furness conserva los ceros estructurales de la semilla.

Las bajadas D_j (que no se observan) se aproximan con la hipótesis de viaje de retorno
(encadenamiento de viajes agregado): quien baja en j por la mañana vuelve a subir en j por la
tarde, de modo que D_j(mañana) ∝ O_j(tarde) y viceversa (ver ``atracciones_espejo``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InfoFurness:
    iteraciones: int
    error_filas: float      # máx. |sum_j T_ij - O_i| / max(O_i)
    error_columnas: float   # máx. |sum_i T_ij - D_j| / max(D_j)
    convergio: bool


def furness(semilla: np.ndarray, O: np.ndarray, D: np.ndarray, tol: float = 1e-6,
            max_iter: int = 1000) -> tuple[np.ndarray, InfoFurness]:
    """Balancea ``semilla`` para que sus filas sumen ``O`` y sus columnas ``D`` (ec. 5.7).

    Pasos (Ortúzar §5.2.3): b_j = 1; a_i = O_i / sum_j t_ij b_j; b_j = D_j / sum_i t_ij a_i;
    repetir. Los ceros de la semilla se conservan (§5.6.1), lo que permite imponer ceros
    estructurales (p. ej. nadie baja en la estación donde subió).
    """
    t = np.asarray(semilla, dtype=float)
    O = np.asarray(O, dtype=float)
    D = np.asarray(D, dtype=float)
    a = np.ones(t.shape[0])
    b = np.ones(t.shape[1])
    err_f = err_c = np.inf
    k = 0
    for k in range(1, max_iter + 1):
        fila = t @ b
        a = np.divide(O, fila, out=np.zeros_like(O), where=fila > 0)
        col = a @ t
        b = np.divide(D, col, out=np.zeros_like(D), where=col > 0)
        T = a[:, None] * t * b[None, :]
        err_f = np.abs(T.sum(1) - O).max() / max(O.max(), 1e-12)
        err_c = np.abs(T.sum(0) - D).max() / max(D.max(), 1e-12)
        if max(err_f, err_c) < tol:
            break
    T = a[:, None] * t * b[None, :]
    return T, InfoFurness(k, float(err_f), float(err_c), bool(max(err_f, err_c) < tol))


def friccion(c: np.ndarray, beta: float, n: float = 0.0) -> np.ndarray:
    """f(c) = c^n · exp(-beta·c): exponencial (n = 0, ec. 5.11) o combinada (ec. 5.13)."""
    c = np.asarray(c, dtype=float)
    f = np.exp(-beta * c)
    if n:
        f = f * np.power(np.where(c > 0, c, 1.0), n)
    return f


def gravedad(O: np.ndarray, D: np.ndarray, C: np.ndarray, beta: float, mascara: np.ndarray,
             n: float = 0.0, tol: float = 1e-6) -> tuple[np.ndarray, InfoFurness]:
    """Modelo gravitacional doblemente acotado (ec. 5.15): T_ij = a_i b_j f(c_ij).

    ``mascara`` (bool) marca los pares permitidos; con beta = 0 y n = 0 se obtiene la
    solución ME2 sin matriz a priori (máxima entropía sujeta a O y D).
    """
    semilla = friccion(C, beta, n) * mascara
    return furness(semilla, O, D, tol=tol)


def costo_medio(T: np.ndarray, C: np.ndarray) -> float:
    tot = T.sum()
    return float((T * C).sum() / tot) if tot > 0 else float("nan")


def calibrar_hyman(O: np.ndarray, D: np.ndarray, C: np.ndarray, mascara: np.ndarray,
                   c_objetivo: float, n: float = 0.0, tol: float = 1e-3,
                   max_iter: int = 30) -> tuple[float, list[tuple[float, float]]]:
    """Beta tal que el coste medio de viaje modelado sea ``c_objetivo`` (ec. 5.42).

    Paso 1: beta0 = 1/c*; paso 2: beta1 = beta0·c0/c*; pasos 3-4: secante
    beta_{m+1} = [(c* - c_{m-1})·beta_m - (c* - c_m)·beta_{m-1}] / (c_m - c_{m-1}).
    """
    historial = []
    b_prev = 1.0 / c_objetivo
    c_prev = costo_medio(gravedad(O, D, C, b_prev, mascara, n)[0], C)
    historial.append((b_prev, c_prev))
    b = b_prev * c_prev / c_objetivo
    for _ in range(max_iter):
        c = costo_medio(gravedad(O, D, C, b, mascara, n)[0], C)
        historial.append((b, c))
        if abs(c - c_objetivo) / c_objetivo < tol or c == c_prev:
            break
        b_next = ((c_objetivo - c_prev) * b - (c_objetivo - c) * b_prev) / (c - c_prev)
        b_prev, c_prev, b = b, c, max(b_next, 0.0)
    return b, historial


def atracciones_espejo(O_dia: np.ndarray, bloques: dict[str, list[int]],
                       espejo: dict[str, str]) -> np.ndarray:
    """Bajadas estimadas D (S, 24) a partir de las subidas O (S, 24) del mismo día.

    Para cada bloque horario B, las bajadas de cada hora h ∈ B se reparten entre estaciones
    según las subidas del bloque espejo (``espejo[B]``): D_j(h) = sum_i O_i(h) · s_j, con
    s_j = O_j(espejo) / sum_k O_k(espejo). Así el total de bajadas de cada hora iguala al de
    subidas (condición de existencia de Furness, ec. 5.8). El bloque "dia" usa el día completo.
    """
    O_dia = np.asarray(O_dia, dtype=float)
    S, H = O_dia.shape
    D = np.zeros_like(O_dia)
    tot_h = O_dia.sum(0)
    horas_bloque = {**bloques, "dia": list(range(H))}
    for bloque, horas in bloques.items():
        ref = O_dia[:, horas_bloque[espejo[bloque]]].sum(1)
        s = ref / ref.sum() if ref.sum() > 0 else np.full(S, 1.0 / S)
        for h in horas:
            D[:, h] = tot_h[h] * s
    return D


def od_dia(O_dia: np.ndarray, D_dia: np.ndarray, C: np.ndarray, beta: float,
           mascara: np.ndarray, n: float = 0.0) -> np.ndarray:
    """Matrices O-D horarias (24, S, S) de un día."""
    S, H = O_dia.shape
    T = np.zeros((H, S, S))
    for h in range(H):
        if O_dia[:, h].sum() > 0:
            T[h] = gravedad(O_dia[:, h], D_dia[:, h], C, beta, mascara, n)[0]
    return T


def carga_por_tramo(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pasajeros a bordo en cada tramo k (entre las estaciones k y k+1 del orden de la línea).

    ``T[..., i, j]`` = viajes que suben en i y bajan en j. Devuelve (asc, desc), cada uno de
    forma (..., S-1): ``asc`` en el sentido de orden creciente y ``desc`` en el decreciente.
    L_asc[k] = sum_{i<=k<j} T_ij = cumsum(subidas_asc - bajadas_asc)[k].
    """
    S = T.shape[-1]
    sup = np.triu(np.ones((S, S), dtype=bool), 1)
    Ta = np.where(sup, T, 0.0)          # j > i: sentido creciente
    Td = np.where(sup.T, T, 0.0)        # j < i: sentido decreciente
    neto_a = Ta.sum(-1) - Ta.sum(-2)    # subidas - bajadas por estación
    neto_d = Td.sum(-1) - Td.sum(-2)
    asc = np.cumsum(neto_a, axis=-1)[..., :-1]
    desc = np.cumsum(neto_d[..., ::-1], axis=-1)[..., :-1][..., ::-1]
    return np.clip(asc, 0, None), np.clip(desc, 0, None)


def carga_al_salir(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carga del vehículo al salir de cada estación (..., S) en cada sentido.

    Es lo que experimenta quien sube en esa estación. En la estación terminal de cada sentido
    no hay salida y la carga es 0.
    """
    asc, desc = carga_por_tramo(T)
    pad = np.zeros(asc.shape[:-1] + (1,))
    return np.concatenate([asc, pad], -1), np.concatenate([pad, desc], -1)
