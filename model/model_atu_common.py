"""Utilidades compartidas para modelamiento ATU por sistema (week08)."""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

SISTEMAS = {
    "metro_l1": {
        "titulo": "Metro Línea 1",
        "grano": "estacion",
        "color": "#2e7d32",
        "cap_key": "metro_l1",
    },
    "troncal": {
        "titulo": "Metropolitano Troncal",
        "grano": "estacion",
        "color": "#1f4e79",
        "cap_key": "troncal",
    },
    "corredor": {
        "titulo": "Corredores Complementarios",
        "grano": "ruta",
        "color": "#4f81bd",
        "cap_key": "corredor",
    },
    "alimentador": {
        "titulo": "Alimentadores",
        "grano": "ruta",
        "color": "#c0504d",
        "cap_key": "alimentador",
    },
}

SPLIT = pd.Timestamp("2025-10-31 23:00:00")
DIAS_EVAL = [
    pd.Timestamp(d) for d in (
        "2025-11-04", "2025-11-11", "2025-11-18", "2025-12-02", "2025-12-09",
    )
]
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_DIAS = 60
MODELO_CHRONOS = "amazon/chronos-bolt-small"
# Prior de producto (no calibrado con APC): local manda; upstream aporta cascada.
ALPHA_UPSTREAM = 0.3
K_UPSTREAM = 2


def find_week08() -> Path:
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / "data_processed" / "clean").exists() and (cand / "code" / "model").exists():
            return cand
        if cand.name == "week08" and (cand / "code" / "model").exists():
            return cand
    raise FileNotFoundError("No encuentro week08")


def paths(raiz: Path | None = None) -> dict:
    raiz = Path(raiz) if raiz is not None else find_week08()
    clean = raiz / "data_processed" / "clean"
    return {
        "RAIZ": raiz,
        "CLEAN": clean,
        "FUENTES": raiz / "data" / "fuentes",
        "MODEL": raiz / "code" / "model",
        "PARQUET": clean / "demanda_consolidada_fiable.parquet",
    }


def out_dirs(sistema: str, raiz: Path | None = None) -> tuple[Path, Path]:
    P = paths(raiz)
    out = P["MODEL"] / "outputs" / sistema
    figs = P["RAIZ"] / "docs" / "images" / "model" / sistema
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    return out, figs


def load_sistema(sistema: str, raiz: Path | None = None) -> pd.DataFrame:
    if sistema not in SISTEMAS:
        raise ValueError(f"sistema debe ser uno de {list(SISTEMAS)}")
    P = paths(raiz)
    if not P["PARQUET"].exists():
        raise FileNotFoundError(f"Falta {P['PARQUET']}")
    cons = pd.read_parquet(P["PARQUET"])
    cons["fecha"] = pd.to_datetime(cons["fecha"])
    cfg = SISTEMAS[sistema]
    sub = cons[(cons["sistema"] == sistema) & (cons["grano"] == cfg["grano"])].copy()
    if sub.empty:
        raise ValueError(f"Sin filas para {sistema}/{cfg['grano']}")
    sub["unidad"] = sub["unidad"].astype(str)
    sub["serie_id"] = sub["unidad"]
    sub["Fecha_Hora"] = sub["fecha"] + pd.to_timedelta(sub["hora"].astype(int), unit="h")

    df_h = (
        sub.groupby(["serie_id", "sistema", "unidad", "Fecha_Hora"], as_index=False)["validaciones"]
        .sum()
        .rename(columns={"validaciones": "Validaciones"})
    )
    meta = (
        sub.groupby(["serie_id", "Fecha_Hora"], as_index=False)
        .agg(
            dia_semana=("dia_semana", "first"),
            tipo_dia=("tipo_dia", "first"),
            es_feriado=("es_feriado", "first"),
            mes=("mes", "first"),
            hora=("hora", "first"),
        )
    )
    df_h = df_h.merge(meta, on=["serie_id", "Fecha_Hora"], how="left")
    df_h = df_h.sort_values(["serie_id", "Fecha_Hora"]).reset_index(drop=True)
    df_h["hora_n"] = df_h["Fecha_Hora"].dt.hour
    df_h["dow"] = df_h["Fecha_Hora"].dt.dayofweek
    df_h["dia"] = df_h["Fecha_Hora"].dt.day
    return df_h


def build_grid(
    df_h: pd.DataFrame,
    min_dias: int | None = None,
    split: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list]:
    if min_dias is None:
        min_dias = MIN_DIAS
    if split is None:
        split = SPLIT
    series = sorted(df_h["serie_id"].unique())
    t0 = df_h["Fecha_Hora"].min().floor("h")
    t1 = df_h["Fecha_Hora"].max().floor("h")
    rango = pd.date_range(t0, t1, freq="h")
    grid = (
        df_h.pivot_table(
            index="Fecha_Hora", columns="serie_id", values="Validaciones", aggfunc="sum",
        )
        .reindex(rango)
        .fillna(0.0)
    )
    grid.index.name = "Fecha_Hora"
    train = grid.loc[:split]
    dias_pos = (train > 0).groupby(train.index.date).any().sum(axis=0)
    ok = [e for e in series if int(dias_pos.get(e, 0)) >= min_dias]
    return grid, ok


def load_chronos(model_name: str = MODELO_CHRONOS):
    from chronos import ChronosBoltPipeline
    return ChronosBoltPipeline.from_pretrained(model_name)


def pronostica_dia(
    pipeline,
    g: pd.DataFrame,
    target_day: pd.Timestamp,
    ok: list,
    horizon: int = 24,
    ctx_days: int = 60,
) -> tuple[list[int], np.ndarray]:
    import torch

    cutoff = target_day - pd.Timedelta(hours=1)
    start = max(cutoff - pd.Timedelta(days=ctx_days), g.index.min())
    contexto = g.loc[start:cutoff, ok].values.T
    mask = (contexto > 0).sum(axis=1) >= 24
    pos_ok = [i for i, m in enumerate(mask) if m]
    if not pos_ok:
        raise RuntimeError(f"Sin series con señal para {target_day.date()}")
    cols = [ok[i] for i in pos_ok]
    ctx = g.loc[start:cutoff, cols].values.T.astype(np.float32)
    pred = pipeline.predict(torch.from_numpy(ctx), prediction_length=horizon)
    if hasattr(pred, "detach"):
        fc = pred.detach().cpu().numpy()
    else:
        fc = np.asarray(pred)
    fc = np.clip(fc[:, :, :horizon], 0, None)
    return pos_ok, fc


def baseline_hora_media(g: pd.DataFrame, target_day: pd.Timestamp, ok: list) -> np.ndarray:
    cutoff = target_day - pd.Timedelta(hours=1)
    hist = g.loc[:cutoff, ok]
    out = np.zeros((len(ok), 24), dtype=float)
    horas = pd.date_range(target_day, target_day + pd.Timedelta(hours=23), freq="h")
    for j, e in enumerate(ok):
        v = hist[e]
        for h_i, ts in enumerate(horas):
            m = v.index.hour == ts.hour
            out[j, h_i] = float(v.loc[m].mean()) if m.any() else 0.0
    return out


def baseline_naive7d(g: pd.DataFrame, target_day: pd.Timestamp, ok: list) -> np.ndarray | None:
    ref = target_day - pd.Timedelta(days=7)
    idx = pd.date_range(ref, ref + pd.Timedelta(hours=23), freq="h")
    if idx.min() not in g.index or idx[-1] not in g.index:
        return None
    return np.maximum(g.loc[idx, ok].values.T, 0.0)


def _feat_matrix_from_grid(
    g: pd.DataFrame,
    ok: list,
    end: pd.Timestamp,
    feriados: set | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build tabular features up to `end` (inclusive) for LightGBM."""
    feriados = feriados or set()
    rows = []
    y = []
    for e in ok:
        s = g[e].loc[:end]
        if len(s) < 170:
            continue
        df = pd.DataFrame({"y": s.values}, index=s.index)
        df["serie_id"] = e
        df["hora"] = df.index.hour
        df["dow"] = df.index.dayofweek
        df["mes"] = df.index.month
        df["es_feriado"] = [1 if ts.normalize() in feriados else 0 for ts in df.index]
        df["lag_24"] = df["y"].shift(24)
        df["lag_168"] = df["y"].shift(168)
        df["roll_mean_7"] = df["y"].shift(24).rolling(7 * 24, min_periods=24).mean()
        df = df.dropna()
        rows.append(df)
        y.append(df["y"].values)
    if not rows:
        return pd.DataFrame(), np.array([])
    X = pd.concat(rows, ignore_index=True)
    return X, X["y"].values


def forecast_lightgbm_dia(
    g: pd.DataFrame,
    target_day: pd.Timestamp,
    ok: list,
    feriados: set | None = None,
) -> np.ndarray:
    """Global LightGBM: train on history < target_day, predict 24h for all ok series."""
    import lightgbm as lgb
    from sklearn.preprocessing import LabelEncoder

    feriados = feriados or set()
    cutoff = target_day - pd.Timedelta(hours=1)
    Xtr, ytr = _feat_matrix_from_grid(g, ok, cutoff, feriados)
    if Xtr.empty or len(ytr) < 500:
        # fallback: hora_media
        return baseline_hora_media(g, target_day, ok)

    le = LabelEncoder()
    Xtr = Xtr.copy()
    Xtr["est_cod"] = le.fit_transform(Xtr["serie_id"].astype(str))
    feats = ["est_cod", "hora", "dow", "mes", "es_feriado", "lag_24", "lag_168", "roll_mean_7"]
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=0,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(Xtr[feats], ytr)

    # recursive 24h forecast per series
    out = np.zeros((len(ok), 24), dtype=float)
    hist = {e: g[e].loc[:cutoff].copy() for e in ok}
    horas = pd.date_range(target_day, target_day + pd.Timedelta(hours=23), freq="h")
    known = set(le.classes_)
    for j, e in enumerate(ok):
        if str(e) not in known:
            out[j] = baseline_hora_media(g, target_day, [e])[0]
            continue
        est_cod = int(le.transform([str(e)])[0])
        series = hist[e]
        preds = []
        for ts in horas:
            lag_24 = float(series.iloc[-24]) if len(series) >= 24 else float(series.iloc[-1])
            lag_168 = float(series.iloc[-168]) if len(series) >= 168 else lag_24
            roll = float(series.iloc[-7 * 24:].mean()) if len(series) >= 24 else lag_24
            row = pd.DataFrame([{
                "est_cod": est_cod,
                "hora": ts.hour,
                "dow": ts.dayofweek,
                "mes": ts.month,
                "es_feriado": 1 if ts.normalize() in feriados else 0,
                "lag_24": lag_24,
                "lag_168": lag_168,
                "roll_mean_7": roll,
            }])
            pred = float(max(0.0, model.predict(row[feats])[0]))
            preds.append(pred)
            series = pd.concat([series, pd.Series([pred], index=[ts])])
        out[j] = np.array(preds)
    return out


def forecast_autoets_dia(
    g: pd.DataFrame,
    target_day: pd.Timestamp,
    ok: list,
) -> np.ndarray:
    """AutoETS (statsforecast) per series, horizon 24, season_length=24."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    cutoff = target_day - pd.Timedelta(hours=1)
    frames = []
    for e in ok:
        s = g[e].loc[:cutoff]
        # need enough history for seasonal ETS
        if len(s) < 48:
            continue
        frames.append(pd.DataFrame({
            "unique_id": str(e),
            "ds": s.index,
            "y": s.values.astype(float),
        }))
    out = np.zeros((len(ok), 24), dtype=float)
    if not frames:
        return baseline_hora_media(g, target_day, ok)

    panel = pd.concat(frames, ignore_index=True)
    sf = StatsForecast(models=[AutoETS(season_length=24)], freq="h", n_jobs=1)
    try:
        fc = sf.forecast(df=panel, h=24)
    except Exception:
        return baseline_hora_media(g, target_day, ok)

    # column name typically AutoETS
    col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    mapa = {str(e): j for j, e in enumerate(ok)}
    for uid, grp in fc.groupby("unique_id"):
        if uid not in mapa:
            continue
        vals = grp[col].values[:24]
        out[mapa[uid], :len(vals)] = np.clip(vals, 0, None)
    # fill missing with hora_media
    miss = [ok[j] for j in range(len(ok)) if out[j].sum() == 0]
    if miss:
        fill = baseline_hora_media(g, target_day, miss)
        for k, e in enumerate(miss):
            out[mapa[str(e)]] = fill[k]
    return out


def feriados_peru_2025() -> set:
    # 16 feriados nacionales 2025 (MTPE); week09 agrega 06-07, 07-23, 07-28, 08-06 y 12-09.
    return {
        pd.Timestamp(d) for d in (
            "2025-01-01", "2025-04-17", "2025-04-18", "2025-05-01",
            "2025-06-07", "2025-06-29", "2025-07-23", "2025-07-28",
            "2025-07-29", "2025-08-06", "2025-08-30", "2025-10-08",
            "2025-11-01", "2025-12-08", "2025-12-09", "2025-12-25",
        )
    }


def walkforward(
    pipeline,
    grid: pd.DataFrame,
    ok: list,
    dias_eval=None,
    quantiles=None,
    feriados: set | None = None,
):
    if dias_eval is None:
        dias_eval = DIAS_EVAL
    if quantiles is None:
        quantiles = QUANTILES
    feriados = feriados or feriados_peru_2025()
    res = []
    t0 = time.time()

    for dia in dias_eval:
        yidx = pd.date_range(dia, dia + pd.Timedelta(hours=23), freq="h")
        if yidx[-1] not in grid.index:
            continue
        yout = np.maximum(grid.loc[yidx, ok].values.T, 0.0)
        pos_ok, fc = pronostica_dia(pipeline, grid, dia, ok)
        yout_ok = yout[pos_ok]
        ok_e = [ok[i] for i in pos_ok]
        q_mid = fc.shape[1] // 2

        maps = {
            "chronos": fc[:, q_mid, :],
            "hora_media": baseline_hora_media(grid, dia, ok)[pos_ok],
        }
        N = baseline_naive7d(grid, dia, ok_e)
        if N is not None:
            maps["naive_7d"] = N

        # competitors on the same ok_e / pos_ok alignment
        try:
            lgbm = forecast_lightgbm_dia(grid, dia, ok, feriados=feriados)
            maps["lightgbm"] = lgbm[pos_ok]
        except Exception as exc:
            print(f"  WARN lightgbm {dia.date()}: {exc}", flush=True)

        try:
            ets = forecast_autoets_dia(grid, dia, ok)
            maps["autoets"] = ets[pos_ok]
        except Exception as exc:
            print(f"  WARN autoets {dia.date()}: {exc}", flush=True)

        for m, p in maps.items():
            e = yout_ok - p
            res.append({
                "dia": str(dia.date()),
                "modelo": m,
                "MAE": float(np.mean(np.abs(e))),
                "RMSE": float(np.sqrt(np.mean(e ** 2))),
                "WMAPE": float(np.sum(np.abs(e)) / (np.sum(np.abs(yout_ok)) + 1e-9)),
            })

        for qi, q in enumerate(quantiles[: fc.shape[1]]):
            e = yout_ok - fc[:, qi, :]
            ql = np.where(e >= 0, q * e, (q - 1) * e)
            res.append({
                "dia": str(dia.date()),
                "modelo": f"pinball_q{int(q * 100)}",
                "MAE": float(np.mean(ql)),
                "RMSE": 0.0,
                "WMAPE": 0.0,
            })

        print(f"  OK {dia.date()} | series Chronos={len(pos_ok)}", flush=True)

    ev = pd.DataFrame(res)
    pivot = (
        ev[~ev["modelo"].astype(str).str.startswith("pinball")]
        .groupby("modelo")[["MAE", "RMSE", "WMAPE"]]
        .mean()
        .sort_values("WMAPE")
    )
    return ev, pivot, time.time() - t0


def headway_maps(fuentes: Path) -> dict:
    HW = {
        "metro_l1": {"LAB": 3.5, "SAB": 6.5, "DOM": 9.0},
        "troncal": {"LAB": 4.0, "SAB": 5.0, "DOM": 8.0},
        "corredor": {"LAB": 10.0, "SAB": 12.0, "DOM": 12.0},
        "alimentador": {"LAB": 6.0, "SAB": 8.0, "DOM": 10.0},
    }
    f1 = Path(fuentes) / "frecuencias_l1_oficial.csv"
    if f1.exists():
        df = pd.read_csv(f1)
        for _, r in df.iterrows():
            td = str(r.get("tipo_dia", "") or "")
            try:
                lo = float(r.get("min_pico", np.nan))
                hi = float(r.get("max_pico", np.nan))
                if not np.isnan(lo) and not np.isnan(hi):
                    mid = (lo + hi) / 2.0
                else:
                    continue
            except Exception:
                continue
            key = "DOM" if "Domingo" in td else ("SAB" if ("Sabado" in td or "Sábado" in td) else "LAB")
            HW["metro_l1"][key] = mid
    return HW


def capacidad_vehiculo(fuentes: Path) -> dict:
    CAP = {"metro_l1": 1200.0, "troncal": 164.0, "corredor": 80.0, "alimentador": 80.0}
    f = Path(fuentes) / "capacidad_vehiculos.csv"
    if f.exists():
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            tipo = str(r.get("tipo", "") or "").lower()
            try:
                pax = float(r.get("pasajeros", np.nan))
            except Exception:
                continue
            if np.isnan(pax):
                continue
            if "linea 1" in tipo or "tren" in tipo:
                CAP["metro_l1"] = pax
            elif "articulado" in tipo:
                CAP["troncal"] = pax
    return CAP


def tipo_dia_bucket(tipo_dia, es_feriado, dow) -> str:
    if bool(es_feriado) or int(dow) == 6:
        return "DOM"
    td = str(tipo_dia or "").upper()
    if td.startswith("SAB") or td.startswith("SÁB") or int(dow) == 5:
        return "SAB"
    return "LAB"


def franja_hora(hora: int) -> str:
    h = int(hora)
    if 6 <= h <= 9:
        return "punta_am"
    if 10 <= h <= 15:
        return "valle"
    if 16 <= h <= 20:
        return "punta_pm"
    return "noche"


def _geo_puntos_path(fuentes: Path, sistema: str) -> Path | None:
    nombres = {
        "metro_l1": "puntos_metro_l1.geojson",
        "troncal": "puntos_troncal.geojson",
        "corredor": "puntos_corredores.geojson",
        "alimentador": "puntos_alimentadores.geojson",
    }
    p = Path(fuentes) / "geo" / nombres.get(sistema, "")
    return p if p.exists() else None


def station_order_north_south(fuentes: Path, sistema: str) -> list[str]:
    """Orden geográfico norte→sur (lat desc). Solo aplicable a grano estación (L1/troncal)."""
    if SISTEMAS.get(sistema, {}).get("grano") != "estacion":
        return []
    path = _geo_puntos_path(fuentes, sistema)
    if path is None:
        return []
    try:
        gj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        unidad = str(props.get("unidad") or props.get("estacion") or "").strip()
        coords = (feat.get("geometry") or {}).get("coordinates")
        if not unidad or not coords or len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        rows.append((unidad, lat))
    rows.sort(key=lambda x: -x[1])  # norte primero
    # unique preserving order
    seen, order = set(), []
    for u, _ in rows:
        if u not in seen:
            seen.add(u)
            order.append(u)
    return order


def upstream_validaciones(
    d: pd.DataFrame,
    order: list[str],
    k: int = K_UPSTREAM,
) -> pd.Series:
    """Media de Validaciones de las k unidades previas en `order` (misma Fecha_Hora)."""
    if not order:
        return pd.Series(0.0, index=d.index)
    piv = (
        d.pivot_table(
            index="Fecha_Hora", columns="serie_id", values="Validaciones", aggfunc="sum",
        )
        .fillna(0.0)
    )
    cols = [c for c in order if c in piv.columns]
    if len(cols) < 2:
        return pd.Series(0.0, index=d.index)
    up_piv = pd.DataFrame(0.0, index=piv.index, columns=cols)
    for i, col in enumerate(cols):
        prev = cols[max(0, i - k):i]
        if prev:
            up_piv[col] = piv[prev].mean(axis=1)
    up_long = up_piv.stack().rename("Validaciones_up").reset_index()
    up_long.columns = ["Fecha_Hora", "serie_id", "Validaciones_up"]
    merged = d[["Fecha_Hora", "serie_id"]].merge(
        up_long, on=["Fecha_Hora", "serie_id"], how="left",
    )
    return merged["Validaciones_up"].fillna(0.0).to_numpy()


def attach_ops(
    df_h: pd.DataFrame,
    sistema: str,
    fuentes: Path,
    split: pd.Timestamp | None = None,
    alpha: float = ALPHA_UPSTREAM,
    out_dir: Path | None = None,
):
    """Adjunta oferta, crowding_index y clases de aforo (proxy de presión).

    Returns
    -------
    d, thr_mid, thr_hi, HW, CAPV, umbrales
        thr_mid/thr_hi = umbrales de punta_am (referencia de impresión).
        umbrales = DataFrame sistema×franja con p60/p90.
    """
    if split is None:
        split = SPLIT
    HW = headway_maps(fuentes)
    CAPV = capacidad_vehiculo(fuentes)
    d = df_h.copy()
    d["bucket"] = [
        tipo_dia_bucket(td, ef, dw)
        for td, ef, dw in zip(d["tipo_dia"], d["es_feriado"], d["dow"])
    ]
    d["franja"] = [franja_hora(h) for h in d["hora_n"]]
    hw_sys = HW.get(sistema, HW["troncal"])
    d["Headway_Min"] = [hw_sys.get(b, 6.0) for b in d["bucket"]]
    cap = CAPV.get(sistema, 80.0)
    d["Cap_vehiculo"] = cap
    d["servicios_hora"] = 60.0 / d["Headway_Min"].clip(lower=1.0)
    d["Cap_hora"] = d["Cap_vehiculo"] * d["servicios_hora"]
    d["Ocupacion_proxy"] = (d["Validaciones"] / d["Cap_hora"]).clip(lower=0)

    order = station_order_north_south(fuentes, sistema)
    d["Validaciones_up"] = upstream_validaciones(d, order, k=K_UPSTREAM)
    d["crowding_up"] = (d["Validaciones_up"] / d["Cap_hora"]).clip(lower=0)
    d["crowding_local"] = d["Ocupacion_proxy"]
    d["crowding_index"] = d["crowding_local"] + float(alpha) * d["crowding_up"]

    train = d.loc[d["Fecha_Hora"] <= split]
    rows_u = []
    thr_map: dict[str, tuple[float, float]] = {}
    for fr, g in train.groupby("franja"):
        mid = float(g["crowding_index"].quantile(0.6))
        hi = float(g["crowding_index"].quantile(0.9))
        thr_map[str(fr)] = (mid, hi)
        rows_u.append({
            "sistema": sistema,
            "franja": fr,
            "thr_mid": mid,
            "thr_hi": hi,
            "n": int(len(g)),
            "alpha_upstream": float(alpha),
            "k_upstream": int(K_UPSTREAM),
            "n_orden_geo": int(len(order)),
        })
    # fallback si alguna franja no aparece en train
    for fr in ("punta_am", "valle", "punta_pm", "noche"):
        if fr not in thr_map:
            mid = float(train["crowding_index"].quantile(0.6))
            hi = float(train["crowding_index"].quantile(0.9))
            thr_map[fr] = (mid, hi)
            rows_u.append({
                "sistema": sistema, "franja": fr,
                "thr_mid": mid, "thr_hi": hi, "n": 0,
                "alpha_upstream": float(alpha),
                "k_upstream": int(K_UPSTREAM),
                "n_orden_geo": int(len(order)),
            })
    umbrales = pd.DataFrame(rows_u).sort_values("franja").reset_index(drop=True)

    def clase_row(idx_val, fr):
        mid, hi = thr_map.get(fr, thr_map["valle"])
        if idx_val >= hi:
            return "Saturado"
        if idx_val >= mid:
            return "De_pie"
        return "Asientos"

    d["aforo"] = [
        clase_row(iv, fr) for iv, fr in zip(d["crowding_index"], d["franja"])
    ]

    thr_mid, thr_hi = thr_map.get("punta_am", thr_map["valle"])
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        umbrales.to_csv(Path(out_dir) / "umbrales_crowding.csv", index=False)

    return d, thr_mid, thr_hi, HW, CAPV, umbrales


def decision_u(eta_min, clase, next_clase) -> str:
    if next_clase is None or (isinstance(next_clase, float) and np.isnan(next_clase)):
        return "desconocido"
    try:
        eta = float(eta_min)
    except Exception:
        eta = 10.0
    if clase == "Saturado" and eta <= 10.0 and next_clase != "Saturado":
        return "esperar_siguiente"
    if clase == "Saturado" and eta > 10.0:
        return "cambiar_paradero"
    return "abordar"


def build_pronostico_24h(
    pipeline,
    grid: pd.DataFrame,
    ok: list,
    sistema: str,
    df_ops: pd.DataFrame,
    thr_mid: float,
    thr_hi: float,
    HW: dict,
    CAPV: dict,
    dia=None,
    modelo: str = "chronos",
    umbrales: pd.DataFrame | None = None,
    alpha: float = ALPHA_UPSTREAM,
    fuentes: Path | None = None,
):
    """Pronóstico 24h. Clases de aforo vía crowding_index (umbrales por franja)."""
    if dia is None:
        dia = DIAS_EVAL[-1]

    thr_map = {"punta_am": (thr_mid, thr_hi), "valle": (thr_mid, thr_hi),
               "punta_pm": (thr_mid, thr_hi), "noche": (thr_mid, thr_hi)}
    if umbrales is not None and len(umbrales):
        for _, r in umbrales.iterrows():
            thr_map[str(r["franja"])] = (float(r["thr_mid"]), float(r["thr_hi"]))

    def clase(idx_val, fr):
        mid, hi = thr_map.get(fr, (thr_mid, thr_hi))
        if idx_val >= hi:
            return "Saturado"
        if idx_val >= mid:
            return "De_pie"
        return "Asientos"

    meta = df_ops.drop_duplicates("serie_id").set_index("serie_id")[["unidad"]]
    hw_sys = HW.get(sistema, HW["troncal"])
    cap = CAPV.get(sistema, 80.0)

    order: list[str] = []
    if fuentes is not None:
        order = station_order_north_south(fuentes, sistema)
    rank = {u: i for i, u in enumerate(order)}

    if modelo == "chronos":
        pos_ok, fc = pronostica_dia(pipeline, grid, dia, ok)
        q_mid = fc.shape[1] // 2
        q_lo, q_hi = 0, fc.shape[1] - 1
        point = fc[:, q_mid, :]
        q10 = fc[:, q_lo, :]
        q90 = fc[:, q_hi, :]
        series_ord = [ok[i] for i in pos_ok]
    else:
        if modelo == "lightgbm":
            point = forecast_lightgbm_dia(grid, dia, ok, feriados=feriados_peru_2025())
        else:
            point = forecast_autoets_dia(grid, dia, ok)
            modelo = "autoets"
        series_ord = list(ok)
        q10 = np.clip(point * 0.85, 0, None)
        q90 = point * 1.15

    # map serie -> row index in point for upstream on forecast
    idx_of = {e: k for k, e in enumerate(series_ord)}

    filas = []
    for h in range(24):
        ts = dia + pd.Timedelta(hours=h)
        fr = franja_hora(ts.hour)
        bucket = tipo_dia_bucket("LAB", False, int(ts.dayofweek))
        hw = float(hw_sys.get(bucket, 6.0))
        servicios = 60.0 / max(hw, 1.0)
        cap_h = cap * servicios
        # upstream from forecast point at same hour
        for k, e in enumerate(series_ord):
            v = float(point[k, h])
            v10 = float(q10[k, h])
            v90 = float(q90[k, h])
            local = max(0.0, v / cap_h) if cap_h > 0 else 0.0
            up_vals = []
            if e in rank and order:
                ri = rank[e]
                for prev in order[max(0, ri - K_UPSTREAM):ri]:
                    if prev in idx_of:
                        up_vals.append(float(point[idx_of[prev], h]))
            up_mean = float(np.mean(up_vals)) if up_vals else 0.0
            crow_up = max(0.0, up_mean / cap_h) if cap_h > 0 else 0.0
            crow = local + float(alpha) * crow_up
            af = clase(crow, fr)
            if h < 23:
                v_n = float(point[k, h + 1])
                local_n = max(0.0, v_n / cap_h) if cap_h > 0 else 0.0
                up_n = []
                if e in rank and order:
                    ri = rank[e]
                    for prev in order[max(0, ri - K_UPSTREAM):ri]:
                        if prev in idx_of:
                            up_n.append(float(point[idx_of[prev], h + 1]))
                crow_n = local_n + float(alpha) * (
                    (float(np.mean(up_n)) / cap_h) if up_n and cap_h > 0 else 0.0
                )
                next_af = clase(crow_n, franja_hora(h + 1))
            else:
                next_af = af
            eta = hw * (1.3 if af == "Saturado" else (1.1 if af == "De_pie" else 1.0))
            filas.append({
                "Fecha_Hora": ts,
                "sistema": sistema,
                "serie_id": e,
                "unidad": meta.loc[e, "unidad"] if e in meta.index else e,
                "Validaciones_pron": v,
                "Val_q10": v10,
                "Val_q90": v90,
                "Headway_base_min": hw,
                "Cap_hora": cap_h,
                "Ocupacion_proxy": local,
                "crowding_index": crow,
                "franja": fr,
                "aforo": af,
                "ETA_pron": eta,
                "modelo_pron": modelo,
                "recomendacion": decision_u(eta, af, next_af),
            })
    pro = pd.DataFrame(filas)
    return pro, dia


def run_aforo_classifiers(df_ops: pd.DataFrame, split: pd.Timestamp | None = None):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    if split is None:
        split = SPLIT
    dfc = df_ops.copy()
    dfc["hora_sin"] = np.sin(2 * np.pi * dfc["hora_n"] / 24)
    dfc["hora_cos"] = np.cos(2 * np.pi * dfc["hora_n"] / 24)
    dfc["es_feriado_i"] = dfc["es_feriado"].astype(int)
    le_est = LabelEncoder().fit(dfc["serie_id"])
    dfc["est_cod"] = le_est.transform(dfc["serie_id"])
    feats = ["hora_n", "hora_sin", "hora_cos", "mes", "dow", "es_feriado_i", "est_cod"]
    tr = dfc["Fecha_Hora"] <= split
    te = dfc["Fecha_Hora"] > split
    Xtr = dfc.loc[tr, feats]
    ytr = dfc.loc[tr, "aforo"]
    Xte = dfc.loc[te, feats]
    yte = dfc.loc[te, "aforo"]
    if len(Xtr) > 200_000:
        Xtr = Xtr.sample(200_000, random_state=0)
        ytr = ytr.loc[Xtr.index]
    if len(Xte) > 80_000:
        Xte = Xte.sample(80_000, random_state=0)
        yte = yte.loc[Xte.index]

    le_y = LabelEncoder().fit(ytr)
    ytr_i = le_y.transform(ytr)
    yte_i = le_y.transform(yte)

    modelos = {
        "LogReg": LogisticRegression(max_iter=800),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=0),
        "HistGB": HistGradientBoostingClassifier(max_depth=8, random_state=0),
    }
    try:
        from xgboost import XGBClassifier
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=120, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=-1,
        )
    except Exception:
        pass

    rows = []
    best = ("", -1.0, None)
    for name, clf in modelos.items():
        if name == "XGBoost":
            clf.fit(Xtr, ytr_i)
            pred_i = clf.predict(Xte)
            pred = le_y.inverse_transform(pred_i)
        else:
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
        f1 = float(f1_score(yte, pred, average="macro"))
        rows.append({
            "modelo": name,
            "accuracy": float(accuracy_score(yte, pred)),
            "balanced_acc": float(balanced_accuracy_score(yte, pred)),
            "f1_macro": f1,
            "n_train": len(Xtr),
            "n_test": len(Xte),
        })
        if f1 > best[1]:
            best = (name, f1, pred)
    return pd.DataFrame(rows), best, yte


def save_manifest(sistema: str, out_dir: Path, **kwargs) -> None:
    payload = {"sistema": sistema, **{k: (str(v) if isinstance(v, Path) else v) for k, v in kwargs.items()}}
    (Path(out_dir) / "manifiesto_modelamiento.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def mejor_modelo_wmape(pivot: pd.DataFrame) -> str:
    """Nombre del modelo con menor WMAPE (excluye pinball)."""
    sub = pivot.copy()
    if "WMAPE" not in sub.columns:
        return "chronos"
    return str(sub["WMAPE"].idxmin())
