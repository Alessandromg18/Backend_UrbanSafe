"""Evaluación en test (2025-11-01 a 2025-12-31) de la ocupación v2 en los cuatro sistemas.

Uso (desde la raíz del backend)::

    python scripts/evaluar_ocupacion.py

"Verdad": la ocupación que la misma cadena calcula con las subidas **observadas** (por estación
en Metro L1 y troncal; por paradero en corredores; por sentido en alimentadores). Se compara con:

- LightGBM (modelo de producción) -> reparto a paraderos/sentidos -> cadena de ocupación;
- baseline media_4sem por la misma cadena.

Así el error incluye el pronóstico de demanda y el reparto a paraderos. Salida:
``models/evaluacion_ocupacion.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "model")]

import model_atu_v2 as V  # noqa: E402
from inferencia import ModeloSistema, panel_desde_contexto  # noqa: E402

CLASES = ["Asientos", "De_pie", "Saturado"]
DIAS = pd.date_range("2025-11-01", "2025-12-31", freq="D")


def observado_unidades(sistema: str, capa, dia) -> dict | None:
    ruta = RAIZ / "data" / "eval" / f"{sistema}_paradas_test.parquet"
    if not ruta.exists():
        return None
    df = _test(sistema)
    x = df[df["fecha"] == pd.Timestamp(dia)]
    out = {}
    for serie, r in capa.rutas.items():
        arr = np.zeros((len(r["unidades"]), 24))
        sub = x[x["serie"] == serie]
        pos = {u: i for i, u in enumerate(r["unidades"])}
        for u, h, v in zip(sub["unidad"], sub["hora"], sub["validaciones"]):
            if u in pos:
                arr[pos[u], int(h)] += v
        out[serie] = arr
    return out


_CACHE = {}


def _test(sistema):
    if sistema not in _CACHE:
        df = pd.read_parquet(RAIZ / "data" / "eval" / f"{sistema}_paradas_test.parquet")
        df["fecha"] = pd.to_datetime(df["fecha"])
        _CACHE[sistema] = df
    return _CACHE[sistema]


def metricas(obs_c, pred_c, obs_l, pred_l) -> dict:
    yo, yp = np.concatenate(obs_c).astype(str), np.concatenate(pred_c).astype(str)
    lo, lp = np.concatenate(obs_l).astype(float), np.concatenate(pred_l).astype(float)
    ok = np.isfinite(lo) & np.isfinite(lp)
    return {"n": int(len(yo)),
            "accuracy": round(accuracy_score(yo, yp), 4),
            "f1_macro": round(f1_score(yo, yp, labels=CLASES, average="macro", zero_division=0), 4),
            "f1_weighted": round(f1_score(yo, yp, labels=CLASES, average="weighted", zero_division=0), 4),
            "balanced_accuracy": round(balanced_accuracy_score(yo, yp), 4),
            "MAE_pasajeros_vehiculo": round(float(np.abs(lp[ok] - lo[ok]).mean()), 2),
            "WMAPE_carga": round(float(np.abs(lp[ok] - lo[ok]).sum() / max(lo[ok].sum(), 1e-9)), 4)}


def evaluar(sistema: str) -> dict:
    m = ModeloSistema.cargar(sistema, verificar_hash=False)
    capa = m.linea
    ctx = pd.read_parquet(RAIZ / "data" / "context" / f"{sistema}_2025.parquet")
    acc = {k: {"c": [], "l": []} for k in ("obs", "lgb", "base")}
    acc_p = {k: {"c": [], "l": []} for k in ("obs", "lgb", "base")}
    dist = []
    t0 = time.time()
    for d in DIAS:
        P, p = panel_desde_contexto(sistema, ctx[m.series], d)
        pt, _, _ = m.pronostico_puntual(P, p, "ligero")
        base, _ = V.f_media_4sem(P, p)
        Yobs = ctx.loc[d:d + pd.Timedelta(hours=23), m.series].to_numpy(float).T
        bucket = V.bucket_dia(d)
        obs_u = observado_unidades(sistema, capa, d) if hasattr(capa, "rutas") else None
        res = {"obs": capa.aplicar(Yobs, m.series, bucket, obs_u) if obs_u is not None
               else capa.aplicar(Yobs, m.series, bucket),
               "lgb": capa.aplicar(pt, m.series, bucket),
               "base": capa.aplicar(base, m.series, bucket)}
        for slot in ("asc", "desc"):
            serv = res["obs"].clase[slot] != "Sin_servicio"
            for k, r in res.items():
                acc[k]["c"].append(r.clase[slot][serv])
                acc[k]["l"].append(r.carga_tren[slot][serv])
            dist.append(res["obs"].clase[slot][serv])
        if res["obs"].paradas:
            for clave, po in res["obs"].paradas.items():
                serv = po["clase"] != "Sin_servicio"
                for k in ("obs", "lgb", "base"):
                    pr = res[k].paradas[clave]
                    acc_p[k]["c"].append(pr["clase"][serv])
                    acc_p[k]["l"].append(pr["carga_tren"][serv])
    out = {
        "nivel_serie": {
            "lightgbm": metricas(acc["obs"]["c"], acc["lgb"]["c"], acc["obs"]["l"], acc["lgb"]["l"]),
            "media_4sem": metricas(acc["obs"]["c"], acc["base"]["c"], acc["obs"]["l"], acc["base"]["l"]),
        },
        "distribucion_clases_observadas": pd.Series(np.concatenate(dist).astype(str)).value_counts(normalize=True).round(4).to_dict(),
        "segundos": round(time.time() - t0, 1),
    }
    if acc_p["obs"]["c"]:
        out["nivel_paradero"] = {
            "lightgbm": metricas(acc_p["obs"]["c"], acc_p["lgb"]["c"], acc_p["obs"]["l"], acc_p["lgb"]["l"]),
            "media_4sem": metricas(acc_p["obs"]["c"], acc_p["base"]["c"], acc_p["obs"]["l"], acc_p["base"]["l"]),
        }
    lgb = out.get("nivel_paradero", out["nivel_serie"])["lightgbm"]
    out["RNF_01"] = {"umbral": "accuracy >= 0.75 y F1-macro >= 0.72",
                     "cumple": bool(lgb["accuracy"] >= 0.75 and lgb["f1_macro"] >= 0.72)}
    return out


def comparar_v1(sistema: str) -> dict:
    """Proxy v1 (presión de abordaje, umbrales p60/p90 por franja) vs ocupación v2, ambos con las
    subidas observadas del test: cuánto cambia la clase que ve el usuario y por qué."""
    m = ModeloSistema.cargar(sistema, verificar_hash=False)
    ctx = pd.read_parquet(RAIZ / "data" / "context" / f"{sistema}_2025.parquet")
    c1, c2, horas = [], [], []
    for d in DIAS:
        Y = ctx.loc[d:d + pd.Timedelta(hours=23), m.series].to_numpy(float).T
        v1 = V.clases_desde_crowding(m.crowding, V.crowding_dia(m.crowding, m.series, Y, d))
        obs_u = observado_unidades(sistema, m.linea, d) if hasattr(m.linea, "rutas") else None
        r = (m.linea.aplicar(Y, m.series, V.bucket_dia(d), obs_u) if obs_u is not None
             else m.linea.aplicar(Y, m.series, V.bucket_dia(d)))
        asc = np.nan_to_num(r.ocupacion["asc"].astype(float), nan=-1)
        desc = np.nan_to_num(r.ocupacion["desc"].astype(float), nan=-1)
        v2 = np.where(asc >= desc, r.clase["asc"], r.clase["desc"])     # sentido más cargado
        serv = v2 != "Sin_servicio"
        c1.append(v1[serv]); c2.append(v2[serv])
        horas.append(np.broadcast_to(np.arange(24), v1.shape)[serv])
    c1, c2, h = np.concatenate(c1).astype(str), np.concatenate(c2).astype(str), np.concatenate(horas)
    tab = pd.crosstab(pd.Series(c1, name="v1"), pd.Series(c2, name="v2"), normalize="index").round(3)
    punta = np.isin(h, [7, 8, 18, 19])
    valle = np.isin(h, [11, 12, 13, 14])
    sat = lambda c, mask: round(float((c[mask] == "Saturado").mean()), 3)  # noqa: E731
    return {
        "acuerdo_clase": round(float((c1 == c2).mean()), 4),
        "v1_saturado_que_v2_ve_con_asientos": round(float(((c1 == "Saturado") & (c2 == "Asientos")).sum()
                                                          / max((c1 == "Saturado").sum(), 1)), 3),
        "saturado_en_punta": {"v1": sat(c1, punta), "v2": sat(c2, punta)},
        "saturado_en_valle": {"v1": sat(c1, valle), "v2": sat(c2, valle)},
        "distribucion": {"v1": pd.Series(c1).value_counts(normalize=True).round(3).to_dict(),
                         "v2": pd.Series(c2).value_counts(normalize=True).round(3).to_dict()},
        "matriz_v1_a_v2": {k: v for k, v in tab.to_dict(orient="index").items()},
    }


def main() -> None:
    res = {"dias_test": f"{DIAS[0].date()} a {DIAS[-1].date()} ({len(DIAS)} días)",
           "verdad": "ocupación calculada con las subidas observadas por la misma cadena"}
    for s in ("metro_l1", "troncal", "corredor", "alimentador"):
        res[s] = evaluar(s)
        res[s]["comparacion_proxy_v1"] = comparar_v1(s)
        print(s, json.dumps(res[s], ensure_ascii=False))
    (RAIZ / "models" / "evaluacion_ocupacion.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                               encoding="utf-8")


if __name__ == "__main__":
    main()
