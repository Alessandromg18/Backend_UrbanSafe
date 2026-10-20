"""Evaluación v2 de los modelos de UrbanSafe AI (week09).

Extiende ``model_atu_common`` (week08) sin cambiar su API:

* calendario 2025 completo (16 feriados) y paros de transportistas verificados;
* rolling-origin día a día: validación 15–31 oct (selección) y test 1 nov–31 dic (reporte);
* modelos nuevos: ``media_4sem``, MSTL, LightGBM directo (Tweedie), Chronos-2 y ensemble;
* intervalos conformal por hora con escala local (cobertura objetivo 80 %);
* aforo en cascada, ETA proxy, regla U (AC-03) y anomalías contra el calendario.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import model_atu_common as M

warnings.filterwarnings("ignore")

# ── Calendario ────────────────────────────────────────────────────────────

FERIADOS = {
    # 2025: 16 feriados nacionales (MTPE / El Peruano)
    "2025-01-01": "Año Nuevo",
    "2025-04-17": "Jueves Santo",
    "2025-04-18": "Viernes Santo",
    "2025-05-01": "Día del Trabajo",
    "2025-06-07": "Batalla de Arica y Día de la Bandera",
    "2025-06-29": "San Pedro y San Pablo",
    "2025-07-23": "Día de la Fuerza Aérea del Perú",
    "2025-07-28": "Fiestas Patrias",
    "2025-07-29": "Fiestas Patrias",
    "2025-08-06": "Batalla de Junín",
    "2025-08-30": "Santa Rosa de Lima",
    "2025-10-08": "Combate de Angamos",
    "2025-11-01": "Todos los Santos",
    "2025-12-08": "Inmaculada Concepción",
    "2025-12-09": "Batalla de Ayacucho",
    "2025-12-25": "Navidad",
    # 2026 (para inferencia más allá de los datos)
    "2026-01-01": "Año Nuevo",
    "2026-04-02": "Jueves Santo",
    "2026-04-03": "Viernes Santo",
    "2026-05-01": "Día del Trabajo",
    "2026-06-07": "Batalla de Arica y Día de la Bandera",
    "2026-06-29": "San Pedro y San Pablo",
    "2026-07-23": "Día de la Fuerza Aérea del Perú",
    "2026-07-28": "Fiestas Patrias",
    "2026-07-29": "Fiestas Patrias",
    "2026-08-06": "Batalla de Junín",
    "2026-08-30": "Santa Rosa de Lima",
    "2026-10-08": "Combate de Angamos",
    "2026-11-01": "Todos los Santos",
    "2026-12-08": "Inmaculada Concepción",
    "2026-12-09": "Batalla de Ayacucho",
    "2026-12-25": "Navidad",
}

PAROS = {
    "2025-02-06": ("Paro de transportistas Lima y Callao",
                   "https://larepublica.pe/sociedad/2025/02/05/paro-de-transportistas-en-lima-y-callao-horario-del-metropolitano-metro-de-lima-y-corredores-este-6-de-febrero-82545"),
    "2025-04-07": ("Paro de transportistas Lima Norte (Metropolitano colapsado)",
                   "https://www.infobae.com/peru/2025/04/07/metropolitano-colapsa-ante-paro-de-transportistas-se-registra-gran-congestion-y-atu-habilita-nuevos-servicios/"),
    "2025-04-10": ("Paro de transportistas de 24 h",
                   "https://www.infobae.com/peru/2025/04/10/paro-de-transportistas-10-de-abril-el-metropolitano-los-corredores-y-metro-seguiran-operativos-en-lima-y-callao-confirmo-la-atu/"),
    "2025-08-21": ("Paro de transportistas de 24 h",
                   "https://www.infobae.com/peru/2025/08/21/paro-de-transportistas-21-de-agosto-estos-seran-los-horarios-del-metropolitano-corredores-metro-de-lima-y-aerodirecto/"),
    "2025-10-02": ("Paro de transportistas",
                   "https://gestion.pe/peru/atu-metropolitano-corredores-y-metro-operaran-con-normalidad-este-jueves-2-de-octubre-noticia/"),
    "2025-10-06": ("Paro de transportistas (alta demanda en Línea 1)",
                   "https://caretas.pe/?p=769420"),
    "2025-10-15": ("Paro de transportistas",
                   "https://larepublica.pe/sociedad/2025/10/14/paro-del-15-de-octubre-horarios-del-metropolitano-corredores-y-metro-de-lima-segun-atu-769468"),
    "2025-11-04": ("Paro de transportistas",
                   "https://www.infobae.com/peru/2025/11/04/paro-de-transportistas-en-vivo-situacion-del-transporte-en-lima-callao-y-otros-distritos-por-asesinatos-y-extorsiones-a-empresas/"),
}

VISPERAS = {"2025-12-24": "Nochebuena", "2025-12-31": "Fin de año"}

FERIADOS_TS = frozenset(pd.Timestamp(d) for d in FERIADOS)
PAROS_TS = frozenset(pd.Timestamp(d) for d in PAROS)
VISPERAS_TS = frozenset(pd.Timestamp(d) for d in VISPERAS)


def tipo_evento(fecha) -> str:
    f = pd.Timestamp(fecha).normalize()
    if f in FERIADOS_TS:
        return "feriado"
    if f in PAROS_TS:
        return "paro"
    if f in VISPERAS_TS:
        return "vispera"
    return "normal"


def calendario_eventos(anio: int = 2025) -> pd.DataFrame:
    filas = [{"fecha": f, "tipo": "feriado", "descripcion": n, "fuente": "MTPE (calendario oficial)"}
             for f, n in FERIADOS.items() if f.startswith(str(anio))]
    filas += [{"fecha": f, "tipo": "paro", "descripcion": d, "fuente": u}
              for f, (d, u) in PAROS.items() if f.startswith(str(anio))]
    filas += [{"fecha": f, "tipo": "vispera", "descripcion": n, "fuente": "calendario"}
              for f, n in VISPERAS.items() if f.startswith(str(anio))]
    return pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)


# ── Protocolo de evaluación ───────────────────────────────────────────────

DIAS_VAL = pd.date_range("2025-10-15", "2025-10-31", freq="D")
DIAS_TEST = pd.date_range("2025-11-01", "2025-12-31", freq="D")
FIN_TRAIN_ES = pd.Timestamp("2025-09-30")       # LightGBM: early stopping con 01–14 oct
FIN_VAL_ES = pd.Timestamp("2025-10-14")
QUANTILES = tuple(M.QUANTILES)
Q_LO, Q_MED, Q_HI = 0, 4, 8                     # índices de q10 / q50 / q90
VENTANA_CONFORMAL = 14                          # días previos para calibrar
COBERTURA = 0.80

CHRONOS2_ID = "amazon/chronos-2"
CHRONOS2_REV = "29ec3766d36d6f73f0696f85560a422f50e8498c"
BOLT_ID = M.MODELO_CHRONOS
BOLT_REV = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"

BASELINES = ("naive_7d", "hora_media", "media_4sem")


# ── Panel ─────────────────────────────────────────────────────────────────

@dataclass
class Panel:
    """Matriz hora × serie con días completos (00–23 h) y banderas de calendario."""

    sistema: str
    index: pd.DatetimeIndex
    Y: np.ndarray          # (T, S) validaciones
    obs: np.ndarray        # (T, S) True si la celda existe en la fuente (horario operativo)
    series: list
    df_h: pd.DataFrame     # panel largo de model_atu_common con es_feriado corregido
    fer: np.ndarray        # (T,) 1.0 si la hora cae en feriado
    paro: np.ndarray       # (T,) 1.0 si la hora cae en paro

    @property
    def S(self) -> int:
        return len(self.series)

    def pos(self, dia) -> int:
        return int(self.index.get_loc(pd.Timestamp(dia)))


def cargar_panel(sistema: str, raiz: Path | None = None) -> Panel:
    df_h = M.load_sistema(sistema, raiz)
    df_h["es_feriado"] = df_h["Fecha_Hora"].dt.normalize().isin(FERIADOS_TS)
    grid, ok = M.build_grid(df_h)
    idx = pd.date_range(grid.index.min().normalize(),
                        grid.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    grid = grid.reindex(idx).fillna(0.0)[ok]
    obs = (
        df_h[df_h["serie_id"].isin(ok)].assign(_v=1.0)
        .pivot_table(index="Fecha_Hora", columns="serie_id", values="_v", aggfunc="max")
        .reindex(index=idx, columns=ok).fillna(0.0).to_numpy() > 0
    )
    dias = idx.normalize()
    return Panel(
        sistema=sistema, index=idx, Y=grid.to_numpy(dtype=np.float64), obs=obs,
        series=list(ok), df_h=df_h,
        fer=dias.isin(FERIADOS_TS).astype(np.float32),
        paro=dias.isin(PAROS_TS).astype(np.float32),
    )


def flags_futuros(dia: pd.Timestamp, horas: int = 24) -> np.ndarray:
    return np.full(horas, 1.0 if pd.Timestamp(dia).normalize() in FERIADOS_TS else 0.0,
                   dtype=np.float32)


# ── Baselines ─────────────────────────────────────────────────────────────

def f_naive_7d(P: Panel, p: int):
    return P.Y[p - 168:p - 144].T.copy(), None


def f_hora_media(P: Panel, p: int):
    return P.Y[:p].reshape(-1, 24, P.S).mean(axis=0).T, None


def f_media_4sem(P: Panel, p: int):
    sem = np.stack([P.Y[p - 168 * k:p - 168 * k + 24] for k in range(1, 5)])
    return sem.mean(axis=0).T, None


# ── Modelos estadísticos (statsforecast) ──────────────────────────────────

def _statsforecast(P: Panel, p: int, modelo, ventana_dias: int | None):
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    ini = 0 if ventana_dias is None else max(0, p - ventana_dias * 24)
    blk = P.Y[ini:p]
    n = blk.shape[0]
    df = pd.DataFrame({
        "unique_id": np.repeat(np.arange(P.S), n),
        "ds": np.tile(P.index[ini:p].values, P.S),
        "y": blk.T.ravel(),
    })
    sf = StatsForecast(models=[modelo], freq="h", n_jobs=1,
                       fallback_model=SeasonalNaive(season_length=24))
    fc = sf.forecast(df=df, h=24)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    fc = fc.sort_values(["unique_id", "ds"])
    out = fc[col].to_numpy(dtype=np.float64).reshape(P.S, 24)
    return np.clip(np.nan_to_num(out), 0, None)


def hacer_autoets(ventana_dias: int | None = 56) -> Callable:
    from statsforecast.models import AutoETS

    def f(P: Panel, p: int):
        return _statsforecast(P, p, AutoETS(season_length=24), ventana_dias), None
    return f


def hacer_mstl(ventana_dias: int = 56) -> Callable:
    from statsforecast.models import MSTL, AutoETS

    def f(P: Panel, p: int):
        modelo = MSTL(season_length=[24, 168], trend_forecaster=AutoETS(model="ZZN"))
        return _statsforecast(P, p, modelo, ventana_dias), None
    return f


# ── LightGBM directo (sin recursión: todos los lags ≥ 24 h) ──────────────

LGB_FEATS = [
    "serie", "hora", "dow", "fer", "fer_prev", "fer_next", "fer_lag7",
    "paro_prev", "paro_lag7", "lag24", "lag48", "lag168", "lag336",
    "m4s", "s4s", "nivel1", "nivel7",
]
LGB_PARAMS = dict(
    objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05,
    num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=1.0, random_state=0, n_jobs=-1, verbosity=-1,
)
MIN_P_LGB = 28 * 24          # primer día con 4 semanas de lags


def features_dia(Y: np.ndarray, p: int, dia: pd.Timestamp) -> np.ndarray:
    """Matriz (S·24, F) del día que empieza en la posición p (contexto hasta p-1)."""
    S = Y.shape[1]
    lag = lambda k: Y[p - k:p - k + 24].T                       # noqa: E731
    sem = np.stack([lag(168 * k) for k in range(1, 5)])
    dia = pd.Timestamp(dia).normalize()
    un_dia, una_sem = pd.Timedelta(days=1), pd.Timedelta(days=7)
    const = {
        "dow": dia.dayofweek,
        "fer": float(dia in FERIADOS_TS),
        "fer_prev": float(dia - un_dia in FERIADOS_TS),
        "fer_next": float(dia + un_dia in FERIADOS_TS),
        "fer_lag7": float(dia - una_sem in FERIADOS_TS),
        "paro_prev": float(dia - un_dia in PAROS_TS),
        "paro_lag7": float(dia - una_sem in PAROS_TS),
    }
    cols = {
        "serie": np.repeat(np.arange(S), 24).astype(float),
        "hora": np.tile(np.arange(24), S).astype(float),
        "lag24": lag(24).ravel(), "lag48": lag(48).ravel(),
        "lag168": lag(168).ravel(), "lag336": lag(336).ravel(),
        "m4s": sem.mean(0).ravel(), "s4s": sem.std(0).ravel(),
        "nivel1": np.repeat(Y[p - 24:p].mean(0), 24),
        "nivel7": np.repeat(Y[p - 168:p].mean(0), 24),
    }
    n = S * 24
    return np.column_stack([cols[c] if c in cols else np.full(n, const[c], dtype=float)
                            for c in LGB_FEATS])


def dataset_lgb(P: Panel, p_ini: int, p_fin: int) -> tuple[np.ndarray, np.ndarray]:
    """Filas de entrenamiento para los días que empiezan en [p_ini, p_fin)."""
    X, y = [], []
    for p in range(max(p_ini, MIN_P_LGB), p_fin, 24):
        X.append(features_dia(P.Y, p, P.index[p]))
        y.append(P.Y[p:p + 24].T.ravel())
    return np.vstack(X), np.concatenate(y)


class LGBMDirecto:
    """LightGBM global por sistema; se re-entrena cada ``refit_cada`` días."""

    def __init__(self, n_estimators: int, refit_cada: int = 7, params: dict | None = None):
        self.n_estimators = int(n_estimators)
        self.refit_cada = refit_cada
        self.params = {**LGB_PARAMS, **(params or {})}
        self.model = None
        self._p_fit = None

    def fit(self, P: Panel, p_fin: int):
        import lightgbm as lgb

        X, y = dataset_lgb(P, 0, p_fin)
        self.model = lgb.LGBMRegressor(n_estimators=self.n_estimators, **self.params)
        self.model.fit(X, y, feature_name=LGB_FEATS, categorical_feature=["serie"])
        self._p_fit = p_fin
        return self

    def __call__(self, P: Panel, p: int):
        if self.model is None or (p - self._p_fit) >= self.refit_cada * 24:
            self.fit(P, p)
        pred = self.model.predict(features_dia(P.Y, p, P.index[p]))
        return np.clip(pred, 0, None).reshape(P.S, 24), None


def lgb_early_stopping(P: Panel, max_iter: int = 2000) -> int:
    import lightgbm as lgb

    p_tr = P.pos(FIN_TRAIN_ES + pd.Timedelta(days=1))
    p_va = P.pos(FIN_VAL_ES + pd.Timedelta(days=1))
    Xtr, ytr = dataset_lgb(P, 0, p_tr)
    Xva, yva = dataset_lgb(P, p_tr, p_va)
    m = lgb.LGBMRegressor(n_estimators=max_iter, **LGB_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1",
          feature_name=LGB_FEATS, categorical_feature=["serie"],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return int(m.best_iteration_ or max_iter)


# ── Modelos fundacionales ─────────────────────────────────────────────────

def cargar_bolt():
    from chronos import ChronosBoltPipeline
    return ChronosBoltPipeline.from_pretrained(BOLT_ID, revision=BOLT_REV, device_map="cpu")


def cargar_chronos2():
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(CHRONOS2_ID, revision=CHRONOS2_REV, device_map="cpu")


def hacer_bolt(pipe, ctx_dias: int = 60) -> Callable:
    """Chronos-Bolt como en week08: contexto de 60 días, series con ≥ 24 h con señal."""
    import torch

    def f(P: Panel, p: int):
        ctx = P.Y[max(0, p - ctx_dias * 24):p].T.astype(np.float32)
        Q = np.zeros((P.S, len(QUANTILES), 24))
        mask = (ctx > 0).sum(axis=1) >= 24
        if mask.any():
            pred = pipe.predict(torch.from_numpy(ctx[mask]), prediction_length=24)
            fc = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
            Q[mask] = np.clip(fc[:, :, :24], 0, None)
        return Q[:, Q_MED, :].copy(), Q
    return f


CHRONOS2_CTX = 2048       # ~85 días: en CPU, 8192 h tarda ~4x más sin mejorar el WMAPE


def hacer_chronos2(pipe, cross_learning: bool = False, ctx_max: int = CHRONOS2_CTX,
                   covariables: bool = True, usar_paro: bool = False) -> Callable:
    """Chronos-2 zero-shot con el feriado como covariable conocida (pasada y futura).

    ``usar_paro`` agrega el paro como covariable solo pasada; triplica el costo en CPU
    y no mejoró la validación, por eso queda apagado por defecto.
    """

    def f(P: Panel, p: int):
        ini = max(0, p - ctx_max)
        fut = flags_futuros(P.index[p])
        inputs = []
        for s in range(P.S):
            item = {"target": P.Y[ini:p, s].astype(np.float32)}
            if covariables:
                past = {"feriado": P.fer[ini:p]}
                if usar_paro:
                    past["paro"] = P.paro[ini:p]
                item["past_covariates"] = past
                item["future_covariates"] = {"feriado": fut}
            inputs.append(item)
        q, _ = pipe.predict_quantiles(inputs, prediction_length=24,
                                      quantile_levels=list(QUANTILES),
                                      cross_learning=cross_learning, batch_size=256)
        Q = np.clip(np.stack([qq[0].numpy().T for qq in q]).astype(np.float64), 0, None)
        return Q[:, Q_MED, :].copy(), Q
    return f


# ── Rolling-origin ────────────────────────────────────────────────────────

@dataclass
class Resultado:
    sistema: str
    dias: pd.DatetimeIndex
    series: list
    y: np.ndarray                  # (D, S, 24)
    obs: np.ndarray                # (D, S, 24)
    point: dict                    # modelo -> (D, S, 24)
    quant: dict                    # modelo -> (D, S, 9, 24) | None
    escala: np.ndarray             # (D, S, 24) escala local para conformal
    segundos: dict


def escala_local(P: Panel, p: int) -> np.ndarray:
    """Nivel estacional de cada serie/hora (media de 4 semanas + 10 % del nivel semanal)."""
    m4s, _ = f_media_4sem(P, p)
    nivel7 = P.Y[p - 168:p].mean(0)[:, None]
    return m4s + 0.1 * nivel7 + 1.0


def rolling_origin(P: Panel, dias, modelos: dict, verbose: bool = True) -> Resultado:
    dias = pd.DatetimeIndex(dias)
    D = len(dias)
    point = {m: np.zeros((D, P.S, 24)) for m in modelos}
    quant: dict = {m: None for m in modelos}
    segundos = {m: 0.0 for m in modelos}
    y = np.zeros((D, P.S, 24))
    obs = np.zeros((D, P.S, 24), dtype=bool)
    escala = np.zeros((D, P.S, 24))
    t0 = time.time()
    for i, dia in enumerate(dias):
        p = P.pos(dia)
        y[i] = P.Y[p:p + 24].T
        obs[i] = P.obs[p:p + 24].T
        escala[i] = escala_local(P, p)
        for m, f in modelos.items():
            t = time.time()
            pt, q = f(P, p)
            segundos[m] += time.time() - t
            point[m][i] = pt
            if q is not None:
                if quant[m] is None:
                    quant[m] = np.zeros((D, P.S, len(QUANTILES), 24))
                quant[m][i] = q
        if verbose and (i % 10 == 0 or i == D - 1):
            print(f"  [{P.sistema}] {dia.date()} ({i + 1}/{D}) {time.time() - t0:.0f}s", flush=True)
    return Resultado(P.sistema, dias, P.series, y, obs, point, quant, escala, segundos)


def agregar_ensemble(R: Resultado, miembros: list, nombre: str = "ensemble") -> None:
    R.point[nombre] = np.mean([R.point[m] for m in miembros], axis=0)
    R.quant[nombre] = None
    R.segundos[nombre] = sum(R.segundos[m] for m in miembros)


# ── Métricas ──────────────────────────────────────────────────────────────

def escala_mase(P: Panel, fin=M.SPLIT) -> np.ndarray:
    Y = P.Y[:P.pos(pd.Timestamp(fin).normalize() + pd.Timedelta(days=1))]
    return np.abs(Y[168:] - Y[:-168]).mean(axis=0)


def wql(y: np.ndarray, Q: np.ndarray) -> float:
    """Weighted quantile loss promedio sobre los 9 cuantiles (y: (...,24), Q: (...,9,24))."""
    tot = 0.0
    for k, q in enumerate(QUANTILES):
        e = y - Q[..., k, :]
        tot += 2 * np.sum(np.maximum(q * e, (q - 1) * e))
    return float(tot / len(QUANTILES) / max(np.sum(np.abs(y)), 1e-9))


def metricas_punto(y, yhat, mase_scale) -> dict:
    e = y - yhat
    ae = np.abs(e)
    ok = mase_scale > 0
    mae_s = ae.mean(axis=(0, 2))
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "WMAPE": float(ae.sum() / max(np.abs(y).sum(), 1e-9)),
        "MASE": float(np.mean(mae_s[ok] / mase_scale[ok])),
    }


def mascara_dias(R: Resultado, estrato: str) -> np.ndarray:
    tipos = np.array([tipo_evento(d) for d in R.dias])
    if estrato == "todos":
        return np.ones(len(R.dias), dtype=bool)
    if estrato == "normales":
        return tipos == "normal"
    if estrato == "atipicos":
        return tipos != "normal"
    dow = np.array([d.dayofweek for d in R.dias])
    if estrato == "laborables":
        return (tipos == "normal") & (dow <= 4)
    if estrato == "fin_de_semana":
        return (tipos == "normal") & (dow >= 5)
    return tipos == estrato


ESTRATOS = ("todos", "normales", "atipicos", "laborables", "fin_de_semana", "feriado", "paro")


def tabla_metricas(R: Resultado, mase_scale, modelos=None, estratos=ESTRATOS) -> pd.DataFrame:
    filas = []
    for m in modelos or list(R.point):
        for est in estratos:
            k = mascara_dias(R, est)
            if not k.any():
                continue
            fila = {"sistema": R.sistema, "modelo": m, "estrato": est, "n_dias": int(k.sum()),
                    **metricas_punto(R.y[k], R.point[m][k], mase_scale)}
            if R.quant.get(m) is not None:
                fila["WQL"] = wql(R.y[k], R.quant[m][k])
            filas.append(fila)
    return pd.DataFrame(filas)


def metricas_diarias(R: Resultado, modelos=None) -> pd.DataFrame:
    filas = []
    for m in modelos or list(R.point):
        for i, d in enumerate(R.dias):
            e = np.abs(R.y[i] - R.point[m][i])
            filas.append({"sistema": R.sistema, "modelo": m, "dia": d.date().isoformat(),
                          "tipo_dia": tipo_evento(d), "MAE": float(e.mean()),
                          "WMAPE": float(e.sum() / max(R.y[i].sum(), 1e-9))})
    return pd.DataFrame(filas)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """DM con corrección de Harvey-Leybourne-Newbold (h = 1). p-valor bilateral."""
    from scipy import stats

    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    var = d.var(ddof=0)
    if n < 3 or var == 0:
        return float("nan"), float("nan")
    dm = d.mean() / math.sqrt(var / n)
    dm *= math.sqrt((n - 1) / n)              # HLN para h = 1
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


# ── Intervalos conformal ──────────────────────────────────────────────────

def _cuantil_conformal(scores: np.ndarray, nivel: float) -> float:
    n = scores.size
    if n == 0:
        return 0.0
    k = min(1.0, math.ceil((n + 1) * nivel) / n)
    return float(np.quantile(scores, k))


def intervalos_conformal(R: Resultado, modelo: str, dias_eval) -> tuple[np.ndarray, np.ndarray]:
    """Split-conformal con ventana deslizante de 14 días y residuos normalizados por hora.

    Modelos con cuantiles nativos: CQR sobre q10–q90. Modelos puntuales: residuos con signo
    (cuantiles 10 % y 90 %) escalados por el nivel local de cada serie.
    """
    alfa = 1 - COBERTURA
    pos = {d: i for i, d in enumerate(R.dias)}
    lo = np.zeros_like(R.y)
    hi = np.zeros_like(R.y)
    Q = R.quant.get(modelo)
    for d in pd.DatetimeIndex(dias_eval):
        i = pos[d]
        cal = [j for j in range(i - VENTANA_CONFORMAL, i) if j >= 0]
        if not cal:
            continue
        for h in range(24):
            obs = R.obs[cal, :, h]
            esc = R.escala[cal, :, h]
            yc = R.y[cal, :, h]
            if Q is not None:
                s = np.maximum(Q[cal, :, Q_LO, h] - yc, yc - Q[cal, :, Q_HI, h]) / esc
                qh = _cuantil_conformal(s[obs], COBERTURA)
                lo[i, :, h] = Q[i, :, Q_LO, h] - qh * R.escala[i, :, h]
                hi[i, :, h] = Q[i, :, Q_HI, h] + qh * R.escala[i, :, h]
            else:
                r = ((yc - R.point[modelo][cal, :, h]) / esc)[obs]
                if r.size == 0:
                    continue
                q_lo = _cuantil_conformal(-r, 1 - alfa / 2)
                q_hi = _cuantil_conformal(r, 1 - alfa / 2)
                lo[i, :, h] = R.point[modelo][i, :, h] - q_lo * R.escala[i, :, h]
                hi[i, :, h] = R.point[modelo][i, :, h] + q_hi * R.escala[i, :, h]
    return np.clip(lo, 0, None), np.clip(hi, 0, None)


def cuantiles_conformal(R: Resultado, modelo: str, dias_cal) -> dict:
    """Cuantiles por hora para producción, calibrados con los días ``dias_cal`` (últimos 14)."""
    alfa = 1 - COBERTURA
    k = np.isin(R.dias, pd.DatetimeIndex(dias_cal))
    Q = R.quant.get(modelo)
    out = {"modelo": modelo, "cobertura": COBERTURA, "tipo": "cqr" if Q is not None else "residuo",
           "dias_calibracion": [str(d.date()) for d in R.dias[k]], "por_hora": {}}
    for h in range(24):
        obs = R.obs[k, :, h]
        esc = R.escala[k, :, h]
        yc = R.y[k, :, h]
        if Q is not None:
            s = np.maximum(Q[k, :, Q_LO, h] - yc, yc - Q[k, :, Q_HI, h]) / esc
            out["por_hora"][h] = {"q": _cuantil_conformal(s[obs], COBERTURA)}
        else:
            r = ((yc - R.point[modelo][k, :, h]) / esc)[obs]
            out["por_hora"][h] = {"q_lo": _cuantil_conformal(-r, 1 - alfa / 2) if r.size else 0.0,
                                  "q_hi": _cuantil_conformal(r, 1 - alfa / 2) if r.size else 0.0}
    return out


def evaluar_intervalos(R: Resultado, modelos, dias_eval) -> pd.DataFrame:
    k = np.isin(R.dias, pd.DatetimeIndex(dias_eval))
    filas = []
    for m in modelos:
        variantes = []
        if R.quant.get(m) is not None:
            variantes.append(("nativo", R.quant[m][:, :, Q_LO, :], R.quant[m][:, :, Q_HI, :]))
        lo, hi = intervalos_conformal(R, m, dias_eval)
        variantes.append(("conformal", lo, hi))
        if R.quant.get(m) is None:
            pt = R.point[m]
            variantes.append(("proxy_15pct", np.clip(pt * 0.85, 0, None), pt * 1.15))
        for nombre, lo_, hi_ in variantes:
            o = R.obs[k]
            yk = R.y[k][o]
            cub = (lo_[k][o] <= yk) & (yk <= hi_[k][o])
            filas.append({
                "sistema": R.sistema, "modelo": m, "intervalo": nombre,
                "cobertura": float(cub.mean()),
                "ancho_rel": float((hi_[k][o] - lo_[k][o]).sum() / max(yk.sum(), 1e-9)),
            })
    return pd.DataFrame(filas)


# ── Aforo en cascada, ETA y regla U ───────────────────────────────────────

@dataclass
class ConfigAforo:
    sistema: str
    umbrales: dict              # franja -> (thr_mid, thr_hi)
    headway: dict               # bucket -> minutos
    cap_vehiculo: float
    orden: list                 # unidades norte→sur (upstream); vacío si grano ruta
    alpha: float = M.ALPHA_UPSTREAM
    k: int = M.K_UPSTREAM

    def to_dict(self) -> dict:
        return {
            "sistema": self.sistema,
            "umbrales": {fr: {"thr_mid": a, "thr_hi": b} for fr, (a, b) in self.umbrales.items()},
            "headway_min": self.headway, "cap_vehiculo": self.cap_vehiculo,
            "orden_upstream": self.orden, "alpha_upstream": self.alpha, "k_upstream": self.k,
            "clases": ["Asientos", "De_pie", "Saturado"],
            "definicion": "crowding = val/Cap_hora + alpha*mean(val k previas)/Cap_hora; "
                          "Cap_hora = cap_vehiculo*60/headway(bucket); umbrales p60/p90 por franja en train",
        }


def config_aforo(P: Panel, fuentes: Path) -> tuple[ConfigAforo, pd.DataFrame]:
    """Umbrales y oferta exactamente como attach_ops (week08), con feriados corregidos."""
    df_ops, _, _, HW, CAPV, umbrales = M.attach_ops(P.df_h, P.sistema, fuentes)
    thr = {str(r["franja"]): (float(r["thr_mid"]), float(r["thr_hi"])) for _, r in umbrales.iterrows()}
    cfg = ConfigAforo(
        sistema=P.sistema, umbrales=thr,
        headway={k: float(v) for k, v in HW.get(P.sistema, HW["troncal"]).items()},
        cap_vehiculo=float(CAPV.get(P.sistema, 80.0)),
        orden=M.station_order_north_south(fuentes, P.sistema),
    )
    return cfg, df_ops


FRANJA_H = np.array([M.franja_hora(h) for h in range(24)])
FACTOR_ETA = {"Asientos": 1.0, "De_pie": 1.1, "Saturado": 1.3}


def bucket_dia(dia) -> str:
    d = pd.Timestamp(dia)
    return M.tipo_dia_bucket(None, d.normalize() in FERIADOS_TS, d.dayofweek)


def crowding_dia(cfg: ConfigAforo, series: list, Yd: np.ndarray, dia) -> np.ndarray:
    """crowding_index (S, 24) para un día a partir de validaciones observadas o pronosticadas."""
    hw = cfg.headway.get(bucket_dia(dia), 6.0)
    cap_h = cfg.cap_vehiculo * 60.0 / max(hw, 1.0)
    local = np.clip(Yd / cap_h, 0, None)
    up = np.zeros_like(Yd)
    if cfg.orden:
        cols = [u for u in cfg.orden if u in series]
        idx = {u: series.index(u) for u in cols}
        for i, u in enumerate(cols):
            prev = cols[max(0, i - cfg.k):i]
            if prev:
                up[idx[u]] = Yd[[idx[v] for v in prev]].mean(axis=0)
    return local + cfg.alpha * np.clip(up / cap_h, 0, None)


def clases_desde_crowding(cfg: ConfigAforo, crow: np.ndarray) -> np.ndarray:
    mid = np.array([cfg.umbrales.get(fr, cfg.umbrales["valle"])[0] for fr in FRANJA_H])
    hi = np.array([cfg.umbrales.get(fr, cfg.umbrales["valle"])[1] for fr in FRANJA_H])
    return np.where(crow >= hi, "Saturado", np.where(crow >= mid, "De_pie", "Asientos"))


def clases_resultado(cfg: ConfigAforo, R: Resultado, Ymat: np.ndarray) -> np.ndarray:
    return np.stack([clases_desde_crowding(cfg, crowding_dia(cfg, R.series, Ymat[i], d))
                     for i, d in enumerate(R.dias)])


def eta_minutos(cfg: ConfigAforo, clases: np.ndarray, dias) -> np.ndarray:
    hw = np.array([cfg.headway.get(bucket_dia(d), 6.0) for d in dias])[:, None, None]
    fac = np.vectorize(FACTOR_ETA.get)(clases)
    return hw * fac


def recomendaciones(clases: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Regla U de week08 vectorizada; la clase siguiente es la de la hora h+1 del mismo día."""
    nxt = np.concatenate([clases[..., 1:], clases[..., -1:]], axis=-1)
    sat = clases == "Saturado"
    return np.where(sat & (eta <= 10.0) & (nxt != "Saturado"), "esperar_siguiente",
                    np.where(sat & (eta > 10.0), "cambiar_paradero", "abordar"))


def _metricas_clf(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    labels = ["Asientos", "De_pie", "Saturado"]
    f1c = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        **{f"f1_{c}": float(v) for c, v in zip(labels, f1c)},
    }


def clasificador_directo(df_ops: pd.DataFrame, R: Resultado, split=M.SPLIT):
    """HistGB de week08 (features calendario + unidad, sin validaciones) evaluado en las celdas de test."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    d = df_ops.copy()
    d["hora_sin"] = np.sin(2 * np.pi * d["hora_n"] / 24)
    d["hora_cos"] = np.cos(2 * np.pi * d["hora_n"] / 24)
    d["es_feriado_i"] = d["es_feriado"].astype(int)
    cod = {s: i for i, s in enumerate(sorted(d["serie_id"].unique()))}
    d["est_cod"] = d["serie_id"].map(cod)
    feats = ["hora_n", "hora_sin", "hora_cos", "mes", "dow", "es_feriado_i", "est_cod"]
    tr = d[d["Fecha_Hora"] <= split]
    if len(tr) > 200_000:
        tr = tr.sample(200_000, random_state=0)
    clf = HistGradientBoostingClassifier(max_depth=8, random_state=0).fit(tr[feats], tr["aforo"])

    pred = np.full(R.y.shape, "Asientos", dtype=object)
    for i, dia in enumerate(R.dias):
        horas = np.arange(24)
        X = pd.DataFrame({
            "hora_n": np.tile(horas, R.y.shape[1]),
            "mes": dia.month, "dow": dia.dayofweek,
            "es_feriado_i": int(dia.normalize() in FERIADOS_TS),
            "est_cod": np.repeat([cod.get(s, -1) for s in R.series], 24),
        })
        X["hora_sin"] = np.sin(2 * np.pi * X["hora_n"] / 24)
        X["hora_cos"] = np.cos(2 * np.pi * X["hora_n"] / 24)
        pred[i] = clf.predict(X[feats]).reshape(R.y.shape[1], 24)
    return clf, cod, feats, pred


def moda_historica(cfg: ConfigAforo, P: Panel, R: Resultado, split=M.SPLIT) -> np.ndarray:
    """Clase más frecuente por serie × hora × bucket en train (baseline)."""
    p_fin = P.pos(pd.Timestamp(split).normalize() + pd.Timedelta(days=1))
    cnt: dict = {}
    for p in range(0, p_fin, 24):
        dia = P.index[p]
        cl = clases_desde_crowding(cfg, crowding_dia(cfg, P.series, P.Y[p:p + 24].T, dia))
        b = bucket_dia(dia)
        ob = P.obs[p:p + 24].T
        for s in range(P.S):
            for h in range(24):
                if ob[s, h]:
                    cnt.setdefault((s, h, b), {}).setdefault(cl[s, h], 0)
                    cnt[(s, h, b)][cl[s, h]] += 1
    out = np.full(R.y.shape, "Asientos", dtype=object)
    for i, d in enumerate(R.dias):
        b = bucket_dia(d)
        for s in range(P.S):
            for h in range(24):
                c = cnt.get((s, h, b))
                if c:
                    out[i, s, h] = max(c, key=c.get)
    return out


def evaluar_aforo(cfg: ConfigAforo, R: Resultado, pred_clases: dict, dias_eval) -> tuple:
    """Aforo multiclase, ETA proxy y regla U contra el oráculo (clases con validaciones reales)."""
    k = np.isin(R.dias, pd.DatetimeIndex(dias_eval))
    verdad = clases_resultado(cfg, R, R.y)
    dias = R.dias[k]
    eta_true = eta_minutos(cfg, verdad[k], dias)
    rec_true = recomendaciones(verdad[k], eta_true)
    obs = R.obs[k]
    # la regla U usa la hora siguiente: exige que también exista en la fuente
    obs_next = np.concatenate([obs[..., 1:], obs[..., -1:]], axis=-1) & obs

    filas_af, filas_rec, conf = [], [], {}
    for nombre, cl in pred_clases.items():
        c = cl[k]
        yt, yp = verdad[k][obs], c[obs]
        filas_af.append({"sistema": R.sistema, "modelo": nombre, **_metricas_clf(yt, yp),
                         "eta_mae_min": float(np.abs(eta_minutos(cfg, c, dias) - eta_true)[obs].mean()),
                         "n": int(obs.sum())})
        conf[nombre] = pd.crosstab(pd.Series(yt, name="real"), pd.Series(yp, name="pred"))
        rec = recomendaciones(c, eta_minutos(cfg, c, dias))
        rt, rp = rec_true[obs_next], rec[obs_next]
        from sklearn.metrics import balanced_accuracy_score, f1_score
        filas_rec.append({
            "sistema": R.sistema, "modelo": nombre,
            "acuerdo": float((rt == rp).mean()),
            "balanced_acc": float(balanced_accuracy_score(rt, rp)),
            "f1_macro": float(f1_score(rt, rp, average="macro", zero_division=0)),
            "pct_no_abordar_real": float((rt != "abordar").mean()),
            "n": int(obs_next.sum()),
        })
    # baseline "siempre abordar"
    rt = rec_true[obs_next]
    from sklearn.metrics import balanced_accuracy_score, f1_score
    rp = np.full(rt.shape, "abordar", dtype=object)
    filas_rec.append({"sistema": R.sistema, "modelo": "siempre_abordar",
                      "acuerdo": float((rt == rp).mean()),
                      "balanced_acc": float(balanced_accuracy_score(rt, rp)),
                      "f1_macro": float(f1_score(rt, rp, average="macro", zero_division=0)),
                      "pct_no_abordar_real": float((rt != "abordar").mean()),
                      "n": int(obs_next.sum())})
    return pd.DataFrame(filas_af), pd.DataFrame(filas_rec), conf


# ── Anomalías ─────────────────────────────────────────────────────────────

def _robust_z(x: np.ndarray, ventana: int = 56) -> np.ndarray:
    """z robusto causal: mediana/MAD de los ``ventana`` días previos."""
    z = np.full(len(x), np.nan)
    for i in range(len(x)):
        hist = x[max(0, i - ventana):i]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 14:
            continue
        med = np.median(hist)
        mad = np.median(np.abs(hist - med)) * 1.4826
        z[i] = (x[i] - med) / max(mad, 1e-9)
    return z


def residuo_diario_baseline(P: Panel, desde: pd.Timestamp) -> pd.DataFrame:
    """Residuo relativo diario del sistema vs media de las 4 semanas previas (causal)."""
    filas = []
    for p in range(P.pos(desde), len(P.index), 24):
        base, _ = f_media_4sem(P, p)
        real = P.Y[p:p + 24].sum()
        esperado = base.sum()
        filas.append({"dia": P.index[p], "real": real, "esperado": esperado,
                      "residuo_rel": (real - esperado) / max(esperado, 1e-9)})
    df = pd.DataFrame(filas)
    df["z"] = _robust_z(df["residuo_rel"].to_numpy())
    df["score"] = df["z"].abs()
    df["tipo_dia"] = [tipo_evento(d) for d in df["dia"]]
    return df


def score_total_global(P: Panel, desde: pd.Timestamp) -> pd.DataFrame:
    """Baseline ingenuo: z del total diario vs media/desv. del mismo día de la semana (todo el año)."""
    tot = pd.Series(P.Y.reshape(-1, 24, P.S).sum(axis=(1, 2)), index=P.index[::24])
    dow = tot.index.dayofweek
    z = (tot - tot.groupby(dow).transform("mean")) / tot.groupby(dow).transform("std")
    df = pd.DataFrame({"dia": tot.index, "score": z.abs().to_numpy()})
    df = df[df["dia"] >= desde].reset_index(drop=True)
    df["tipo_dia"] = [tipo_evento(d) for d in df["dia"]]
    return df


def score_isolation_forest(df_ops: pd.DataFrame, desde: pd.Timestamp, split=M.SPLIT):
    """IsolationForest de week08: % de celdas en el 5 % más anómalo, por día."""
    from sklearn.ensemble import IsolationForest

    cols = ["Validaciones", "hora_n", "dow", "mes", "Headway_Min", "Ocupacion_proxy"]
    tr = df_ops.loc[df_ops["Fecha_Hora"] <= split, cols].fillna(0)
    if len(tr) > 150_000:
        tr = tr.sample(150_000, random_state=0)
    ifo = IsolationForest(contamination=0.05, random_state=0, n_jobs=-1).fit(tr)
    sub = df_ops[df_ops["Fecha_Hora"] >= desde]
    s = -ifo.score_samples(sub[cols].fillna(0))
    thr = float(np.quantile(-ifo.score_samples(tr), 0.95))
    df = (pd.DataFrame({"dia": sub["Fecha_Hora"].dt.normalize().to_numpy(), "out": s >= thr})
          .groupby("dia", as_index=False)["out"].mean().rename(columns={"out": "score"}))
    df["tipo_dia"] = [tipo_evento(d) for d in df["dia"]]
    return ifo, thr, df


def evaluar_detector(df: pd.DataFrame, nombre: str, sistema: str, umbral: float | None,
                     positivos=("feriado", "paro")) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    d = df.dropna(subset=["score"])
    y = d["tipo_dia"].isin(positivos).to_numpy()
    s = d["score"].to_numpy()
    out = {"sistema": sistema, "detector": nombre, "n_dias": int(len(d)), "n_pos": int(y.sum()),
           "pr_auc": float(average_precision_score(y, s)) if y.any() else float("nan"),
           "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")}
    k = int(y.sum())
    if k:
        top = np.argsort(-s)[:k]
        out["precision_at_k"] = float(y[top].mean())
    if umbral is not None:
        pred = s >= umbral
        tp = int((pred & y).sum())
        out.update({"umbral": umbral, "precision": tp / max(pred.sum(), 1),
                    "recall": tp / max(y.sum(), 1)})
        out["f1"] = (2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-9))
    return out


# ── Utilidades de salida ──────────────────────────────────────────────────

def _limpiar(o):
    """Convierte a tipos JSON estándar; NaN/inf → null (JSON válido también para JavaScript)."""
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): _limpiar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpiar(v) for v in o]
    if isinstance(o, np.ndarray):
        return _limpiar(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return o


def a_json(obj) -> str:
    return json.dumps(_limpiar(obj), ensure_ascii=False, indent=2, allow_nan=False)


def predicciones_largas(R: Resultado, modelos=None) -> pd.DataFrame:
    """Formato largo (modelo, fecha_hora, serie, y, yhat[, q10, q90]) para guardar en parquet."""
    D, S, H = R.y.shape
    fh = (np.repeat(R.dias.values, S * H) +
          np.tile(np.arange(H), D * S).astype("timedelta64[h]"))
    base = pd.DataFrame({"fecha_hora": fh, "serie": np.tile(np.repeat(R.series, H), D),
                         "y": R.y.ravel(), "obs": R.obs.ravel()})
    out = []
    for m in modelos or list(R.point):
        df = base.copy()
        df.insert(0, "modelo", m)
        df["yhat"] = R.point[m].ravel()
        if R.quant.get(m) is not None:
            df["q10"] = R.quant[m][:, :, Q_LO, :].ravel()
            df["q90"] = R.quant[m][:, :, Q_HI, :].ravel()
        out.append(df)
    return pd.concat(out, ignore_index=True)
