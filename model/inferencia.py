"""Inferencia con los artefactos serializados de ``deliveries/week10/models/`` (Issue #19).

Uso típico (backend o batch diario que llena la tabla ``prediccion``):

    import sys; sys.path.insert(0, "deliveries/week10/code/model")
    from inferencia import ModeloSistema

    m = ModeloSistema.cargar("metro_l1")             # verifica sha256 contra el manifiesto
    contexto = ...                                   # DataFrame hora × unidad, ≥ 28 días hasta dia-1 23:00
    salida = m.pronosticar(contexto, "2026-01-01")   # pronóstico 24 h + banda 80 % + aforo + ETA + regla U

El contexto es la serie horaria de validaciones (índice ``Fecha_Hora``, una columna por unidad),
exactamente como ``Panel.Y`` de ``model_atu_v2``; puede armarse desde la vista ``demanda_fiable``.
"""
from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODEL_DIR.parent / "models"
sys.path.insert(0, str(MODEL_DIR))

import model_atu_v2 as V  # noqa: E402
from ocupacion import cargar_capa  # noqa: E402

DIAS_CONTEXTO = 28


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with Path(ruta).open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_json(ruta: Path) -> dict:
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _pipeline(tipo: str, repo: str, revision: str):
    if tipo == "chronos2":
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(repo, revision=revision, device_map="cpu")
    from chronos import ChronosBoltPipeline
    return ChronosBoltPipeline.from_pretrained(repo, revision=revision, device_map="cpu")


def recomendar(clase: np.ndarray, espera: np.ndarray) -> np.ndarray:
    """Regla U (week08) con la clase de la hora siguiente; "sin_servicio" fuera de horario."""
    rec = V.recomendaciones(np.asarray(clase)[None], np.nan_to_num(espera, nan=99.0)[None])[0]
    return np.where(np.asarray(clase) == "Sin_servicio", "sin_servicio", rec)


def panel_desde_contexto(sistema: str, contexto: pd.DataFrame, dia) -> tuple[V.Panel, int]:
    """Panel con el contexto observado y el día objetivo en cero (no se usa para predecir)."""
    dia = pd.Timestamp(dia).normalize()
    ctx = contexto.sort_index()
    ctx = ctx.loc[:dia - pd.Timedelta(hours=1)]
    inicio = ctx.index.min().normalize()
    if (dia - inicio).days < DIAS_CONTEXTO:
        raise ValueError(f"se requieren ≥ {DIAS_CONTEXTO} días de contexto antes de {dia.date()}")
    idx = pd.date_range(inicio, dia + pd.Timedelta(hours=23), freq="h")
    Y = ctx.reindex(idx).fillna(0.0)
    dias = idx.normalize()
    P = V.Panel(
        sistema=sistema, index=idx, Y=Y.to_numpy(dtype=np.float64),
        obs=np.ones(Y.shape, dtype=bool), series=[str(c) for c in Y.columns], df_h=None,
        fer=dias.isin(V.FERIADOS_TS).astype(np.float32),
        paro=dias.isin(V.PAROS_TS).astype(np.float32),
    )
    return P, P.pos(dia)


class ModeloSistema:
    """Modelo operativo de un sistema: pronosticador principal + aforo + ETA + regla U."""

    def __init__(self, sistema: str, carpeta: Path, card: dict):
        self.sistema = sistema
        self.carpeta = carpeta
        self.card = card
        self.series = card["series"]
        self.principal = card["pronostico"]["modelo_principal"]
        self.crowding = self._config_aforo(leer_json(carpeta / "crowding_config.json"))
        self.conformal = leer_json(carpeta / "conformal.json")
        self.conformal_ligero = leer_json(carpeta / "conformal_ligero.json")
        self.anomalias = leer_json(carpeta / "anomalias.json")
        self.linea = cargar_capa(carpeta)   # capa de ocupación (None = índice v1)
        # Demanda por ruta × sentido (alimentadores, v1.2.0): si existe, reemplaza al LightGBM por
        # serie; el total de cada serie es la suma de sus sentidos
        self.sentido = leer_json(carpeta / "demanda_sentido.json") if (carpeta / "demanda_sentido.json").exists() else None
        self.conformal_sentido = (leer_json(carpeta / self.sentido["conformal"]) if self.sentido else None)
        self._lgb = None
        self._lgb_sentido = None

    # ── carga ──
    @classmethod
    def cargar(cls, sistema: str, models_dir: Path = MODELS_DIR, verificar_hash: bool = True):
        models_dir = Path(models_dir)
        carpeta = models_dir / sistema
        if verificar_hash:
            manifiesto = leer_json(models_dir / "manifest.json")
            for rel, meta in manifiesto["artefactos"].items():
                if rel.startswith(f"{sistema}/"):
                    real = sha256(models_dir / rel)
                    if real != meta["sha256"]:
                        raise ValueError(f"sha256 distinto en {rel}: artefacto alterado o corrupto")
        return cls(sistema, carpeta, leer_json(carpeta / "model_card.json"))

    @staticmethod
    def _config_aforo(d: dict) -> V.ConfigAforo:
        return V.ConfigAforo(
            sistema=d["sistema"],
            umbrales={fr: (u["thr_mid"], u["thr_hi"]) for fr, u in d["umbrales"].items()},
            headway={k: float(v) for k, v in d["headway_min"].items()},
            cap_vehiculo=float(d["cap_vehiculo"]), orden=list(d["orden_upstream"]),
            alpha=float(d["alpha_upstream"]), k=int(d["k_upstream"]),
        )

    def _booster(self):
        if self._lgb is None:
            import lightgbm as lgb
            self._lgb = lgb.Booster(model_file=str(self.carpeta / "lightgbm.txt"))
        return self._lgb

    def _booster_sentido(self):
        if self._lgb_sentido is None:
            import lightgbm as lgb
            self._lgb_sentido = lgb.Booster(model_file=str(self.carpeta / self.sentido["archivo"]))
        return self._lgb_sentido

    def pronostico_por_sentido(self, contexto: pd.DataFrame, dia) -> np.ndarray:
        """(series_sentido, 24) con el LightGBM por ruta × sentido."""
        cols = self.sentido["series_sentido"]
        faltan = [c for c in cols if c not in contexto.columns]
        if faltan:
            raise ValueError(f"faltan series por sentido en el contexto: {faltan[:3]}")
        Pd, q = panel_desde_contexto(self.sistema, contexto[cols], dia)
        pred = self._booster_sentido().predict(V.features_dia(Pd.Y, q, Pd.index[q]))
        return np.clip(pred, 0, None).reshape(len(cols), 24)

    def agrupar_sentidos(self, yd: np.ndarray) -> tuple[np.ndarray, dict]:
        """(total por serie (S, 24), {serie: (sentidos, 24)} en el orden de la capa de ocupación)."""
        cols = self.sentido["series_sentido"]
        total, por_serie = [], {}
        for s in self.series:
            filas = {c.split("|", 1)[1]: yd[j] for j, c in enumerate(cols) if c.split("|", 1)[0] == s}
            total.append(np.sum(list(filas.values()), axis=0) if filas else np.zeros(24))
            unidades = self.linea.rutas[s]["unidades"] if self.linea is not None and hasattr(self.linea, "rutas") else list(filas)
            por_serie[s] = np.stack([filas.get(u, np.zeros(24)) for u in unidades])
        return np.stack(total), por_serie

    # ── pronóstico ──
    def _pronostico_modelo(self, nombre: str, P: V.Panel, p: int):
        spec = self.card["pronostico"]["modelos"][nombre]
        if nombre == "lightgbm":
            X = V.features_dia(P.Y, p, P.index[p])
            pred = self._booster().predict(X)
            return np.clip(pred, 0, None).reshape(P.S, 24), None
        if nombre in ("chronos2", "chronos_bolt"):
            pipe = _pipeline(nombre, spec["repo"], spec["revision"])
            if nombre == "chronos2":
                return V.hacer_chronos2(pipe, ctx_max=spec["contexto_horas"])(P, p)
            return V.hacer_bolt(pipe, ctx_dias=spec["contexto_dias"])(P, p)
        if nombre == "autoets":
            return V.hacer_autoets(ventana_dias=spec["ventana_dias"])(P, p)
        if nombre == "mstl":
            return V.hacer_mstl(ventana_dias=spec["ventana_dias"])(P, p)
        fn = {"media_4sem": V.f_media_4sem, "naive_7d": V.f_naive_7d, "hora_media": V.f_hora_media}[nombre]
        return fn(P, p)

    def pronostico_puntual(self, P: V.Panel, p: int, modo: str = "principal") -> tuple[np.ndarray, np.ndarray | None, str]:
        """Pronóstico del modelo principal (o de LightGBM en ``modo="ligero"``).

        Si el modelo falla, degrada a ``media_4sem`` (espíritu RF-05).
        """
        try:
            if modo == "ligero":
                return self._pronostico_modelo("lightgbm", P, p)[0], None, "lightgbm"
            if self.principal == "ensemble":
                miembros = self.card["pronostico"]["ensemble"]["miembros"]
                pts = [self._pronostico_modelo(m, P, p)[0] for m in miembros]
                return np.mean(pts, axis=0), None, "ensemble"
            pt, Q = self._pronostico_modelo(self.principal, P, p)
            return pt, Q, self.principal
        except Exception as exc:  # pragma: no cover - ruta de contingencia
            print(f"[inferencia] {self.sistema}: {self.principal} falló ({exc}); uso media_4sem")
            pt, _ = V.f_media_4sem(P, p)
            return pt, None, "media_4sem_fallback"

    def _banda(self, pt, Q, escala, conformal: dict) -> tuple[np.ndarray, np.ndarray]:
        por_hora = conformal["por_hora"]
        lo, hi = np.zeros_like(pt), np.zeros_like(pt)
        for h in range(24):
            c = por_hora.get(str(h), por_hora.get(h, {}))
            if conformal["tipo"] == "cqr" and Q is not None:
                lo[:, h] = Q[:, V.Q_LO, h] - c.get("q", 0.0) * escala[:, h]
                hi[:, h] = Q[:, V.Q_HI, h] + c.get("q", 0.0) * escala[:, h]
            else:
                lo[:, h] = pt[:, h] - c.get("q_lo", 0.0) * escala[:, h]
                hi[:, h] = pt[:, h] + c.get("q_hi", 0.0) * escala[:, h]
        return np.clip(lo, 0, None), np.clip(hi, 0, None)

    def pronosticar(self, contexto: pd.DataFrame, dia, modo: str = "principal", con_paradas: bool = False):
        """``modo="ligero"`` usa solo LightGBM: sin torch ni Chronos, milisegundos por sistema y
        precisión estadísticamente equivalente al principal en la evaluación (ver model_card)."""
        faltan = [s for s in self.series if s not in contexto.columns]
        if faltan:
            raise ValueError(f"faltan unidades en el contexto: {faltan[:5]}")
        P, p = panel_desde_contexto(self.sistema, contexto[self.series], dia)
        observado = None
        if self.sentido is not None and modo == "ligero":
            pt, observado = self.agrupar_sentidos(self.pronostico_por_sentido(contexto, dia))
            Q, usado, conformal = None, "lightgbm_sentido", self.conformal_sentido
        else:
            pt, Q, usado = self.pronostico_puntual(P, p, modo)
            if usado == "lightgbm" and modo == "ligero":
                conformal = self.conformal_ligero
            else:
                conformal = self.conformal if usado == self.principal else {"tipo": "residuo", "por_hora": {}}
        lo, hi = self._banda(pt, Q if usado == self.principal else None, V.escala_local(P, p), conformal)
        # subidas por sentido (si el modelo las pronostica) en lugar del reparto con participaciones
        dia = P.index[p]
        S = P.S
        base = {
            "sistema": self.sistema,
            "unidad": np.repeat(P.series, 24),
            "fecha_hora": np.tile(pd.date_range(dia, periods=24, freq="h"), S),
            "validaciones_pron": pt.ravel(),
            "banda80_lo": lo.ravel(), "banda80_hi": hi.ravel(),
        }
        operativa, r = self._operativa(P.series, pt, dia, observado)
        operativa = {k: (v.ravel() if isinstance(v, np.ndarray) else v) for k, v in operativa.items()}
        df = pd.DataFrame({**base, **operativa, "modelo": usado, "tipo_dia": V.tipo_evento(dia)})
        if not con_paradas:
            return df
        return df, self._tabla_paradas(r, dia)

    def salida_operativa(self, series: list, Yd: np.ndarray, dia) -> dict:
        """Aforo, espera y recomendación (arreglos S × 24) a partir de subidas por hora.

        Sirve igual para subidas pronosticadas (app) que observadas (evaluación). Si el sistema
        tiene capa de ocupación (``linea_config.json`` o ``rutas_config.json``) usa la ocupación
        a bordo estimada con la matriz O-D; si no, el índice de presión de abordaje de la v1.
        """
        return self._operativa(series, Yd, dia)[0]

    def _operativa(self, series: list, Yd: np.ndarray, dia, por_unidad: dict | None = None):
        dia = pd.Timestamp(dia)
        if self.linea is None:
            crow = V.crowding_dia(self.crowding, list(series), Yd, dia)
            clases = V.clases_desde_crowding(self.crowding, crow)
            eta = V.eta_minutos(self.crowding, clases[None], [dia])[0]
            hw = self.crowding.headway.get(V.bucket_dia(dia), 6.0)
            return {"crowding_index": crow, "aforo": clases, "eta_min": eta,
                    "eta2_min": eta + hw,
                    "recomendacion": V.recomendaciones(clases[None], eta[None])[0],
                    "metodo_aforo": "presion_abordaje_v1"}, None

        r = (self.linea.aplicar(Yd, list(series), V.bucket_dia(dia), por_unidad) if por_unidad is not None
             else self.linea.aplicar(Yd, list(series), V.bucket_dia(dia)))
        S = len(series)
        hw = r.intervalo_min
        out = {"intervalo_min": hw}
        if r.bajadas is not None:
            out["bajadas_est"] = r.bajadas
        for s in ("asc", "desc"):
            clase = r.clase[s]
            # recomendación por costo generalizado (model/recomendacion.py); regla U solo de respaldo
            rec = r.recomendacion.get(s) if r.recomendacion else None
            out.update({f"carga_{s}": r.carga_tren[s], f"ocupacion_{s}": r.ocupacion[s],
                        f"aforo_{s}": clase, f"eta_{s}": r.espera_min[s],
                        f"recomendacion_{s}": rec if rec is not None else recomendar(clase, r.espera_min[s])})
            if r.p_no_subir:
                out.update({f"p_no_subir_{s}": r.p_no_subir[s], f"viaje_{s}": r.viaje_min[s],
                            f"multiplicador_{s}": r.multiplicador[s]})
        # Sin sentido conocido se informa el sentido más cargado (criterio conservador)
        critico_asc = np.nan_to_num(r.ocupacion["asc"], nan=-1) >= np.nan_to_num(r.ocupacion["desc"], nan=-1)
        elegir = lambda k: np.where(critico_asc, out[f"{k}_asc"], out[f"{k}_desc"])  # noqa: E731
        desc = np.array([[self.linea.sentidos_de(s)["asc"], self.linea.sentidos_de(s)["desc"]] for s in series])
        out.update({
            "sentido_critico": np.where(critico_asc, desc[:, :1], desc[:, 1:]),
            "crowding_index": elegir("ocupacion"), "aforo": elegir("aforo"),
            "eta_min": elegir("eta"), "eta2_min": elegir("eta") + hw,
            "recomendacion": elegir("recomendacion"), "metodo_aforo": "ocupacion_od_v2",
        })
        if r.p_no_subir:
            out.update({"p_no_subir": elegir("p_no_subir"), "viaje_min": elegir("viaje"),
                        "multiplicador": elegir("multiplicador")})
        assert all(v.shape == (S, 24) for v in out.values() if isinstance(v, np.ndarray))
        return out, r

    @staticmethod
    def _tabla_paradas(r, dia) -> pd.DataFrame | None:
        """Detalle por paradero (corredores): una fila por (serie, paradero, hora)."""
        if r is None or not r.paradas:
            return None
        horas = pd.date_range(pd.Timestamp(dia), periods=24, freq="h")
        filas = []
        for (serie, cod), d in r.paradas.items():
            filas.append(pd.DataFrame({
                "unidad": serie, "paradero": cod, "nombre": d["nombre"], "sentido": d["sentido"],
                "fecha_hora": horas, "subidas_est": d["subidas"], "bajadas_est": d["bajadas"],
                "carga": d["carga_tren"], "ocupacion": d["ocupacion"], "aforo": d["clase"],
                "eta": d["espera_min"], "recomendacion": d["recomendacion"],
                "p_no_subir": d["p_no_subir"], "viaje_min": d["viaje_min"],
                "multiplicador": d["multiplicador"], "alternativa": d["alternativa"]}))
        return pd.concat(filas, ignore_index=True)

    # ── anomalías (residuo diario vs media de 4 semanas, z robusto) ──
    def anomalia_dia(self, contexto: pd.DataFrame, dia, reales: pd.DataFrame) -> dict:
        P, p = panel_desde_contexto(self.sistema, contexto[self.series], dia)
        base, _ = V.f_media_4sem(P, p)
        esperado = float(base.sum())
        real = float(reales[self.series].to_numpy().sum())
        rel = (real - esperado) / max(esperado, 1e-9)
        ref = self.anomalias["referencia"]
        z = (rel - ref["mediana"]) / max(ref["mad"], 1e-9)
        return {"sistema": self.sistema, "dia": str(P.index[p].date()), "residuo_rel": rel,
                "z": z, "anomalo": bool(abs(z) >= self.anomalias["umbral_z"]),
                "direccion": "alza" if rel > 0 else "baja"}
