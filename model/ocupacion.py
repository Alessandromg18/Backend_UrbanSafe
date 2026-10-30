"""Ocupación a bordo por serie, hora y sentido a partir de las subidas pronosticadas.

Tres capas, según lo que permiten los datos de cada sistema:

- ``LineaOD`` (Metro L1, Metropolitano troncal): línea con estaciones ordenadas y subidas sin
  sentido. Subidas O → bajadas D por viaje de retorno → O-D horaria doblemente acotada (Furness)
  → carga por tramo y sentido → pasajeros por vehículo según la oferta.
- ``RutasParaderoOD`` (corredores): el pronóstico es por ruta; se reparte a paraderos con las
  participaciones históricas (el código ATU del paradero ya indica el sentido). Por ruta y
  sentido: O-D acotada a orígenes (Ortúzar ec. 5.16) con atracciones por viaje de retorno
  medidas en el paradero del sentido opuesto.
- ``RutasCargaCritica`` (alimentadores): casi todos los viajes empiezan o terminan en la
  estación de integración, así que la carga máxima del bus ≈ subidas del sentido por hora.

Todas devuelven ``ResultadoOcupacion`` con arreglos (S, 24) en el orden de ``series`` y dos
"ranuras" de sentido, ``asc`` y ``desc``, cuya descripción depende de la serie
(``sentidos_de``). La espera en andén usa Ortúzar ec. (10.19): t = (h² + σ²) / 2h.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import od_matrix as OD
import recomendacion as R

SIN_SERVICIO = "Sin_servicio"
BUCKETS = ("LAB", "SAB", "DOM")
SLOTS = ("asc", "desc")


@dataclass
class ResultadoOcupacion:
    carga_tren: dict            # slot -> (S, 24) pasajeros por vehículo (en el tramo relevante)
    ocupacion: dict             # slot -> (S, 24) carga / capacidad (puede superar 1)
    clase: dict                 # slot -> (S, 24) Asientos | De_pie | Saturado | Sin_servicio
    espera_min: dict            # slot -> (S, 24)
    intervalo_min: np.ndarray   # (S, 24)
    bajadas: np.ndarray | None = None
    paradas: dict = field(default_factory=dict)   # (serie, paradero) -> dict de arreglos (24,)
    p_no_subir: dict = field(default_factory=dict)      # slot -> (S, 24) P(el primer vehículo no tiene espacio)
    viaje_min: dict = field(default_factory=dict)       # slot -> (S, 24) tiempo medio a bordo de quien sube ahí
    multiplicador: dict = field(default_factory=dict)   # slot -> (S, 24) hacinamiento (Whelan y Crockett)
    recomendacion: dict = field(default_factory=dict)   # slot -> (S, 24) abordar | esperar_siguiente | ...


class Vehiculo:
    def __init__(self, d: dict):
        self.capacidad = float(d["capacidad"])
        self.asientos = float(d["asientos"])
        self.umbral_saturado = float(d["umbral_saturado_pasajeros"])
        # área de pie: dato del vehículo o, si no está, la que da 6 pasajeros/m² a capacidad
        self.area_pie = float(d.get("area_de_pie_m2") or (self.capacidad - self.asientos) / 6.0)

    def clases(self, carga: np.ndarray, servicio: np.ndarray) -> np.ndarray:
        c = np.where(carga > self.umbral_saturado, "Saturado",
                     np.where(carga > self.asientos, "De_pie", "Asientos")).astype(object)
        c[~servicio] = SIN_SERVICIO
        return c


def p_no_subir(carga: np.ndarray, capacidad: float, sube_por_vehiculo: np.ndarray,
               cv: float = 0.0) -> np.ndarray:
    """Probabilidad de que el vehículo que le toca al pasajero llegue sin espacio para subir.

    - Exceso medio: si la carga media supera la capacidad, el exceso se reparte entre quienes
      esperan subir en el paradero.
    - Irregularidad del servicio (cv > 0): con llegadas aleatorias de pasajeros, la carga de cada
      vehículo es proporcional al intervalo que lo precede (los buses atrasados llegan más
      llenos), y el pasajero cae con más probabilidad en los intervalos largos. Con intervalos
      Gamma de coeficiente de variación cv (el mismo de la espera, ec. 10.19), la probabilidad
      ponderada por pasajero de que su vehículo supere la capacidad es Q(k+1, k·cap/carga), con
      k = 1/cv² (Q = función gamma incompleta superior regularizada).
    """
    carga = np.asarray(carga, float)
    exceso = np.clip(carga - capacidad, 0, None)
    sube = np.asarray(sube_por_vehiculo, float)
    p = np.clip(np.divide(exceso, sube, out=np.zeros_like(exceso), where=sube > 0), 0, 1)
    if cv > 0:
        from scipy.special import gammaincc
        k = 1.0 / cv ** 2
        r = np.divide(capacidad, carga, out=np.full_like(carga, np.inf), where=carga > 0)
        p_lleno = np.where(np.isfinite(r), gammaincc(k + 1.0, k * np.where(np.isfinite(r), r, 0.0)), 0.0)
        p = np.maximum(p, p_lleno)
    return p


def espera(hw: np.ndarray, cv: float, carga: np.ndarray, capacidad: float,
           sube_por_vehiculo: np.ndarray) -> np.ndarray:
    """Espera esperada (min): (h² + σ²)/2h con σ = cv·h (ec. 10.19), más h · P(no subir)."""
    return hw * (1 + cv ** 2) / 2.0 + hw * p_no_subir(carga, capacidad, sube_por_vehiculo, cv)


def _por_vehiculo(x: np.ndarray, vh: np.ndarray) -> np.ndarray:
    return np.divide(x, vh, out=np.zeros_like(x, dtype=float), where=vh > 0)


def _intervalo(vh: np.ndarray) -> np.ndarray:
    return np.where(vh > 0, 60.0 / np.where(vh > 0, vh, 1.0), np.nan)


def cargar_capa(carpeta: Path):
    carpeta = Path(carpeta)
    if (carpeta / "linea_config.json").exists():
        return LineaOD(json.loads((carpeta / "linea_config.json").read_text(encoding="utf-8")))
    if (carpeta / "rutas_config.json").exists():
        cfg = json.loads((carpeta / "rutas_config.json").read_text(encoding="utf-8"))
        shares = dict(np.load(carpeta / "rutas_participaciones.npz"))
        cls = RutasParaderoOD if cfg["tipo"] == "rutas_paradero" else RutasCargaCritica
        return cls(cfg, shares)
    return None


# ── Líneas (Metro L1, troncal) ─────────────────────────────────────────────
class LineaOD:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sistema = cfg["sistema"]
        self.estaciones = list(cfg["estaciones"])
        km = np.asarray(cfg["km"], dtype=float)
        v = float(cfg["velocidad_comercial_kmh"])
        self.C = np.abs(km[:, None] - km[None, :]) / v * 60.0
        # nadie baja donde sube (ni en una estación contigua de la misma ubicación)
        self.mascara = np.abs(km[:, None] - km[None, :]) > 0.05
        od = cfg["od"]
        self.beta = float(od["beta"])
        self.n = float(od.get("n", 0.0))
        self.bloques = {k: list(x) for k, x in od["bloques"].items()}
        self.espejo = dict(od["espejo"])
        self.veh = Vehiculo(cfg["vehiculo"])
        self.capacidad = self.veh.capacidad
        self.cv = float(cfg["oferta"].get("cv_intervalo", 0.0))
        self.trenes_hora = {k: np.asarray(x, dtype=float) for k, x in cfg["oferta"]["trenes_hora"].items()}
        # Intervalo del servicio que puede tomar el usuario (troncal: cada servicio para en un
        # subconjunto de estaciones); si no se define, el de todos los vehículos (Metro)
        self.intervalo_servicio = {k: np.asarray(x, dtype=float)
                                   for k, x in cfg["oferta"].get("intervalo_servicio_min", {}).items()}
        self.sentidos = dict(cfg["sentidos"])

    def sentidos_de(self, serie: str) -> dict:
        return self.sentidos

    def posicion(self, serie: str) -> int | None:
        return self.estaciones.index(serie) if serie in self.estaciones else None

    def clases(self, carga_tren, servicio):
        return self.veh.clases(carga_tren, servicio)

    def aplicar(self, Y: np.ndarray, series: list, bucket: str) -> ResultadoOcupacion:
        idx = [list(series).index(e) for e in self.estaciones]
        O = np.clip(np.asarray(Y, dtype=float)[idx], 0, None)
        D = OD.atracciones_espejo(O, self.bloques, self.espejo)
        T = OD.od_dia(O, D, self.C, self.beta, self.mascara, self.n)
        asc, desc = OD.carga_al_salir(T)                        # (24, S)
        vh = self.trenes_hora.get(bucket, self.trenes_hora["LAB"])
        servicio_h = vh > 0
        hw = _intervalo(vh)
        if bucket in self.intervalo_servicio:
            hw = np.where(servicio_h, self.intervalo_servicio[bucket], np.nan)
        inv = np.argsort(idx)
        S = len(self.estaciones)
        # tiempo medio a bordo de quien sube en i, por sentido (pesos = viajes T_ij de la matriz O-D)
        sup = np.triu(np.ones((S, S), dtype=bool), 1)
        viaje = {}
        for slot, m in (("asc", sup), ("desc", sup.T)):
            Tm = np.where(m, T, 0.0)                                # (24, S, S)
            num, den = (Tm * self.C).sum(-1), Tm.sum(-1)            # (24, S)
            viaje[slot] = np.divide(num, den, out=np.zeros_like(num), where=den > 0).T
        claves = ("carga_tren", "ocupacion", "clase", "espera_min", "p_no_subir", "viaje_min",
                  "multiplicador", "recomendacion")
        res = {k: {} for k in claves}
        for slot, L in (("asc", asc.T), ("desc", desc.T)):
            por_tren = _por_vehiculo(L, np.broadcast_to(vh, L.shape))
            sin_salida = np.zeros_like(L, dtype=bool)
            sin_salida[-1 if slot == "asc" else 0] = True
            servicio = servicio_h[None, :] & ~sin_salida
            sube = _por_vehiculo(O, np.broadcast_to(vh, O.shape))
            esp = espera(np.broadcast_to(hw, L.shape), self.cv, por_tren, self.capacidad, sube)
            pns = p_no_subir(por_tren, self.capacidad, sube, self.cv)
            mult = R.multiplicador(por_tren, self.veh.asientos, self.veh.area_pie)
            clase = self.veh.clases(por_tren, servicio)
            res["carga_tren"][slot] = por_tren[inv]
            res["ocupacion"][slot] = (por_tren / self.capacidad)[inv]
            res["clase"][slot] = clase[inv]
            res["espera_min"][slot] = np.where(servicio, esp, np.nan)[inv]
            res["p_no_subir"][slot] = np.where(servicio, pns, np.nan)[inv]
            res["viaje_min"][slot] = np.where(servicio, viaje[slot], np.nan)[inv]
            res["multiplicador"][slot] = np.where(servicio, mult, np.nan)[inv]
            res["recomendacion"][slot] = R.decidir(clase, pns)[inv]
        return ResultadoOcupacion(intervalo_min=np.broadcast_to(hw, (S, 24)).copy(),
                                  bajadas=D[inv], **res)

    # compatibilidad con la v2.0 de Metro L1 (scripts y tests)
    @property
    def asientos(self):
        return self.veh.asientos

    @property
    def umbral_saturado(self):
        return self.veh.umbral_saturado


# ── Rutas de bus ───────────────────────────────────────────────────────────
class _Rutas:
    def __init__(self, cfg: dict, shares: dict):
        self.cfg = cfg
        self.sistema = cfg["sistema"]
        self.rutas = cfg["rutas"]
        self.shares = shares                   # serie -> (unidades, 3, 24)
        self.cv = float(cfg.get("cv_intervalo", 0.5))
        self.veh_sistema = Vehiculo(cfg["vehiculo"]) if "vehiculo" in cfg else None

    def vehiculo(self, serie: str) -> Vehiculo:
        r = self.rutas[serie]
        return Vehiculo(r["vehiculo"]) if "vehiculo" in r else self.veh_sistema

    def slots(self, serie: str) -> dict:
        """slot -> clave de sentido en la configuración de la ruta (orden alfabético)."""
        claves = sorted(self.rutas[serie]["sentidos"])
        return dict(zip(SLOTS, claves))

    def sentidos_de(self, serie: str) -> dict:
        sent = self.rutas[serie]["sentidos"]
        return {slot: sent[k]["descripcion"] for slot, k in self.slots(serie).items()}

    def vh(self, serie: str, bucket: str) -> np.ndarray:
        return np.asarray(self.rutas[serie]["buses_hora"].get(bucket, self.rutas[serie]["buses_hora"]["LAB"]), float)

    def subidas_unidad(self, serie: str, y: np.ndarray, bucket: str) -> np.ndarray:
        """Pronóstico de la ruta (24,) repartido a sus unidades (paraderos o sentidos)."""
        return np.clip(y, 0, None)[None, :] * self.shares[serie][:, BUCKETS.index(bucket), :]

    def _vacio(self, S):
        z = lambda v=0.0: {s: np.full((S, 24), v, dtype=object if v == SIN_SERVICIO else float) for s in SLOTS}  # noqa: E731
        rec = {s: np.full((S, 24), "sin_servicio", dtype=object) for s in SLOTS}
        return {"carga_tren": z(), "ocupacion": z(), "clase": z(SIN_SERVICIO),
                "espera_min": z(np.nan), "p_no_subir": z(np.nan), "viaje_min": z(np.nan),
                "multiplicador": z(np.nan), "recomendacion": rec}

    def velocidad_kmh(self) -> float:
        return float(self.cfg.get("oferta", {}).get("velocidad_comercial_kmh", 20.0))


class RutasParaderoOD(_Rutas):
    def __init__(self, cfg, shares):
        super().__init__(cfg, shares)
        od = cfg["od"]
        self.bloques = {k: list(x) for k, x in od["bloques"].items()}
        self.espejo = dict(od["espejo"])

    def perfil(self, O: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ruta de un sentido (paraderos en orden, (n, 24)): O-D acotada a orígenes
        T_ij = O_i W_j / sum_{k>i} W_k (j > i). Devuelve (bajadas, carga al salir)."""
        W = W + 1e-3 * (W.mean(0, keepdims=True) + 1e-9)    # celdas vacías (Ortúzar p. 588)
        suf = np.cumsum(W[::-1], 0)[::-1] - W                # sum_{k>i} W_k
        c = np.divide(O, suf, out=np.zeros_like(O), where=suf > 0)
        prev = np.cumsum(c, 0) - c                          # sum_{i<j} O_i / S_i
        bajadas = W * prev
        carga = np.cumsum(O, 0) - np.cumsum(bajadas, 0)
        return bajadas, np.clip(carga, 0, None)

    def aplicar(self, Y: np.ndarray, series: list, bucket: str,
                observado: dict | None = None) -> ResultadoOcupacion:
        """``observado`` (opcional): serie -> (unidades, 24) subidas observadas por paradero o
        sentido, en lugar de repartir ``Y`` con las participaciones (se usa en la evaluación)."""
        S = len(series)
        res = self._vacio(S)
        hw_all = np.full((S, 24), np.nan)
        paradas = {}
        for si, serie in enumerate(series):
            if serie not in self.rutas:
                continue
            r = self.rutas[serie]
            veh = self.vehiculo(serie)
            unidades = r["unidades"]
            O_u = (observado[serie] if observado and serie in observado
                   else self.subidas_unidad(serie, np.asarray(Y[si], float), bucket))
            vh = self.vh(serie, bucket)
            hw = _intervalo(vh)
            hw_all[si] = hw
            for slot, clave in self.slots(serie).items():
                sen = r["sentidos"][clave]
                orden = [unidades.index(c) for c in sen["paradas"]]
                pareja = [unidades.index(c) for c in sen["pareja"]]
                O = O_u[orden]
                W = OD.atracciones_espejo(O_u[pareja], self.bloques, self.espejo)
                baj, carga = self.perfil(O, W)
                ct = _por_vehiculo(carga, np.broadcast_to(vh, carga.shape))
                servicio = np.broadcast_to(vh > 0, ct.shape).copy()
                servicio[-1] = False                      # última parada: no hay salida
                cl = veh.clases(ct, servicio)
                sube = _por_vehiculo(O, np.broadcast_to(vh, O.shape))
                esp = espera(np.broadcast_to(hw, ct.shape), self.cv, ct, veh.capacidad, sube)
                esp = np.where(servicio, esp, np.nan)
                pns = p_no_subir(ct, veh.capacidad, sube, self.cv)
                mult = R.multiplicador(ct, veh.asientos, veh.area_pie)
                # tiempo medio a bordo de quien sube en k: km medio hasta su bajada (pesos W_j, j > k)
                km = np.asarray(sen["km"], float)[:, None]
                Wp = W + 1e-3 * (W.mean(0, keepdims=True) + 1e-9)
                suf_w = np.cumsum(Wp[::-1], 0)[::-1] - Wp
                suf_wkm = np.cumsum((Wp * km)[::-1], 0)[::-1] - Wp * km
                km_medio = np.divide(suf_wkm, suf_w, out=np.zeros_like(suf_w), where=suf_w > 0) - km
                viaje = np.clip(km_medio, 0, None) / self.velocidad_kmh() * 60.0
                costo_aqui = R.costo(esp, viaje, mult)
                # alternativa: subir en un paradero anterior (mismo bus, viene más vacío)
                costo_alt = np.full_like(costo_aqui, np.inf)
                alt_idx = np.full(costo_aqui.shape, -1)
                for k in range(1, len(km)):
                    for j in range(k - 1, -1, -1):
                        dist_m = (km[k, 0] - km[j, 0]) * 1000.0
                        if dist_m > R.CAMINATA_MAX_M:
                            break
                        viaje_j = viaje[k] + (km[k, 0] - km[j, 0]) / self.velocidad_kmh() * 60.0
                        c = R.costo(esp[j], viaje_j, mult[j], caminata_m=dist_m)
                        mejor = servicio[j] & (c < costo_alt[k])
                        costo_alt[k] = np.where(mejor, c, costo_alt[k])
                        alt_idx[k] = np.where(mejor, j, alt_idx[k])
                rec = R.decidir(cl, pns, costo_aqui, costo_alt)
                for k, cod in enumerate(sen["paradas"]):
                    alternativa = [
                        {"paradero": sen["nombres"][j], "codigo": sen["paradas"][j],
                         "distancia_m": round(float((km[k, 0] - km[j, 0]) * 1000.0)),
                         "ahorro_min": round(float(costo_aqui[k, h] - costo_alt[k, h]), 1),
                         "ocupacion": round(float(ct[j, h] / veh.capacidad), 3)}
                        if j >= 0 and rec[k, h] == "cambiar_paradero" else None
                        for h, j in enumerate(alt_idx[k])]
                    paradas[(serie, cod)] = {
                        "slot": slot, "nombre": sen["nombres"][k], "sentido": sen["descripcion"],
                        "subidas": O[k], "bajadas": baj[k], "carga_tren": ct[k],
                        "ocupacion": ct[k] / veh.capacidad, "clase": cl[k], "espera_min": esp[k],
                        "p_no_subir": np.where(servicio[k], pns[k], np.nan),
                        "viaje_min": np.where(servicio[k], viaje[k], np.nan),
                        "multiplicador": np.where(servicio[k], mult[k], np.nan),
                        "recomendacion": rec[k], "alternativa": alternativa}
                # nivel ruta: tramo más cargado del sentido en cada hora
                kmax = np.argmax(ct, 0)
                h = np.arange(24)
                res["carga_tren"][slot][si] = ct[kmax, h]
                res["ocupacion"][slot][si] = ct[kmax, h] / veh.capacidad
                res["clase"][slot][si] = np.where(vh > 0, cl[kmax, h], SIN_SERVICIO)
                res["espera_min"][slot][si] = np.where(vh > 0, esp[kmax, h], np.nan)
                res["p_no_subir"][slot][si] = np.where(vh > 0, pns[kmax, h], np.nan)
                res["viaje_min"][slot][si] = np.where(vh > 0, viaje[kmax, h], np.nan)
                res["multiplicador"][slot][si] = np.where(vh > 0, mult[kmax, h], np.nan)
                res["recomendacion"][slot][si] = R.decidir(res["clase"][slot][si], pns[kmax, h])
        return ResultadoOcupacion(intervalo_min=hw_all, paradas=paradas, **res)


class RutasCargaCritica(_Rutas):
    def aplicar(self, Y: np.ndarray, series: list, bucket: str,
                observado: dict | None = None) -> ResultadoOcupacion:
        """``observado`` (opcional): serie -> (unidades, 24) subidas observadas por paradero o
        sentido, en lugar de repartir ``Y`` con las participaciones (se usa en la evaluación)."""
        S = len(series)
        res = self._vacio(S)
        hw_all = np.full((S, 24), np.nan)
        for si, serie in enumerate(series):
            if serie not in self.rutas:
                continue
            r = self.rutas[serie]
            veh = self.vehiculo(serie)
            O_s = (observado[serie] if observado and serie in observado
                   else self.subidas_unidad(serie, np.asarray(Y[si], float), bucket))   # (sentidos, 24)
            vh = self.vh(serie, bucket)
            hw = _intervalo(vh)
            hw_all[si] = hw
            for slot, clave in self.slots(serie).items():
                carga = O_s[r["unidades"].index(clave)]
                ct = _por_vehiculo(carga, vh)
                servicio = vh > 0
                res["carga_tren"][slot][si] = ct
                res["ocupacion"][slot][si] = ct / veh.capacidad
                res["clase"][slot][si] = veh.clases(ct, servicio)
                # en el terminal todos suben al mismo bus: P(no cabe) con exceso / carga
                res["espera_min"][slot][si] = np.where(servicio, espera(hw, self.cv, ct, veh.capacidad, ct), np.nan)
                pns = p_no_subir(ct, veh.capacidad, ct, self.cv)
                # sin detalle por paradero: viaje medio = la mitad del recorrido de la ruta
                largo = float(r.get("oferta", {}).get("largo_km", 8.0))
                res["p_no_subir"][slot][si] = np.where(servicio, pns, np.nan)
                res["viaje_min"][slot][si] = np.where(servicio, largo / 2 / self.velocidad_kmh() * 60.0, np.nan)
                res["multiplicador"][slot][si] = np.where(servicio, R.multiplicador(ct, veh.asientos, veh.area_pie), np.nan)
                res["recomendacion"][slot][si] = R.decidir(res["clase"][slot][si], pns)
        return ResultadoOcupacion(intervalo_min=hw_all, **res)
