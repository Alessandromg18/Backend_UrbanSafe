"""Reentrena el pronóstico de demanda de **alimentadores** por ruta × sentido (v1.2.0).

Uso (desde la raíz del backend; los parquet limpios están en el repo DPD_project)::

    python scripts/entrenar_alimentador_sentido.py --datos ../../DPD_project/deliveries/week10/data_processed/clean

Por qué: el modelo por ruta (23 series) se repartía después por sentido con participaciones
históricas, pero los dos sentidos tienen perfiles horarios opuestos (mañana hacia la estación de
integración, tarde desde ella) y el reparto fijo pierde la variación de cada día. Un LightGBM
global sobre las 46 series ruta × sentido baja el error en el nivel que usa la app.

Protocolo sin fuga de información (el mismo de la evaluación v2):

- n_estimators por early stopping: entrena <= 2025-09-30, valida 2025-10-01..14.
- Selección contra el modelo por ruta: rolling-origin en validación (2025-10-15..31), refit
  cada 7 días. Regla fijada antes de ver el test: el modelo más simple a menos de 1 % del mejor.
- Reporte: rolling-origin en test (2025-11-01..12-31), refit cada 7 días.
- Bandas del 80 %: conformal con los residuos fuera de muestra de los últimos 14 días (como v1).
- Modelo final: entrenado con todo 2025 (para servir 2026 con el día homólogo).

Salidas: columnas "ruta|sentido" agregadas a ``data/context/alimentador_2025.parquet``;
``models/alimentador/{lightgbm_sentido.txt, demanda_sentido.json, conformal_sentido.json}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "model")]

import model_atu_v2 as V  # noqa: E402
from inferencia import panel_desde_contexto  # noqa: E402

CARPETA = RAIZ / "models" / "alimentador"
CTX = RAIZ / "data" / "context" / "alimentador_2025.parquet"
VAL = pd.date_range("2025-10-15", "2025-10-31")
TEST = pd.date_range("2025-11-01", "2025-12-31")
DIAS_CAL = TEST[-V.VENTANA_CONFORMAL:]
TOLERANCIA_SELECCION = 0.01
PARAMS = dict(V.LGB_PARAMS)


def wmape(a, b) -> float:
    return float(np.abs(a - b).sum() / b.sum())


def panel(ctx: pd.DataFrame):
    P, p_fin = panel_desde_contexto("alimentador", ctx, "2025-12-31")
    P.Y[p_fin:] = ctx.loc["2025-12-31"].to_numpy(float)
    return P


def early_stop(P) -> int:
    import lightgbm as lgb
    p_tr = P.pos(V.FIN_TRAIN_ES + pd.Timedelta(days=1))
    p_va = P.pos(V.FIN_VAL_ES + pd.Timedelta(days=1))
    X, y = V.dataset_lgb(P, 0, p_tr)
    Xv, yv = V.dataset_lgb(P, p_tr, p_va)
    m = lgb.LGBMRegressor(n_estimators=3000, **PARAMS)
    m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="l1", feature_name=V.LGB_FEATS,
          categorical_feature=["serie"], callbacks=[lgb.early_stopping(100, verbose=False)])
    return int(m.best_iteration_ or 3000)


def rolling(P, n: int, dias) -> np.ndarray:
    mdl = V.LGBMDirecto(n_estimators=n, refit_cada=7, params=PARAMS)
    return np.stack([mdl(P, P.pos(d))[0] for d in dias])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, type=Path)
    args = ap.parse_args()

    card = json.loads((CARPETA / "model_card.json").read_text(encoding="utf-8"))
    rutas = card["series"]
    cfg = json.loads((CARPETA / "rutas_config.json").read_text(encoding="utf-8"))
    shares = dict(np.load(CARPETA / "rutas_participaciones.npz"))

    a = pd.read_parquet(args.datos / "alimentador_hora.parquet",
                        columns=["fecha", "hora", "ruta", "sentido_norm", "validaciones"])
    a = a[a["ruta"].isin(rutas)]
    a["Fecha_Hora"] = pd.to_datetime(a["fecha"]) + pd.to_timedelta(a["hora"].astype(int), unit="h")
    a["serie"] = a["ruta"].astype(str) + "|" + a["sentido_norm"].astype(str)
    idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", name="Fecha_Hora")
    Yd = a.pivot_table(index="Fecha_Hora", columns="serie", values="validaciones", aggfunc="sum")
    Yd = Yd.reindex(idx).fillna(0.0)
    # solo los sentidos que la capa de ocupación conoce, en orden ruta / sentido
    sentidos = [f"{r}|{s}" for r in rutas for s in cfg["rutas"][r]["unidades"]]
    Yd = Yd.reindex(columns=sentidos).fillna(0.0)

    ctx = pd.read_parquet(CTX)
    Yr = ctx[rutas]
    suma = Yd.T.groupby(lambda c: c.split("|")[0]).sum().T[rutas]
    dif = float(np.abs(suma.to_numpy() - Yr.to_numpy()).max())
    assert dif < 1e-6, f"la suma por sentido no reproduce la serie por ruta ({dif})"

    Pr, Pd = panel(Yr), panel(Yd)
    meta_r = json.loads((CARPETA / "lightgbm_meta.json").read_text(encoding="utf-8"))
    n_r, n_d = meta_r["n_estimators"], early_stop(Pd)

    def a_sentidos(pred_r, dias):
        out = np.zeros((len(dias), len(sentidos), 24))
        for k, d in enumerate(dias):
            b = ("LAB", "SAB", "DOM").index(V.bucket_dia(d))
            for j, c in enumerate(sentidos):
                r, s = c.split("|")
                out[k, j] = pred_r[k, rutas.index(r)] * shares[r][cfg["rutas"][r]["unidades"].index(s), b]
        return out

    obs = lambda dias: np.stack([Yd.loc[d:d + pd.Timedelta(hours=23)].to_numpy(float).T for d in dias])  # noqa: E731
    metricas = {}
    preds_test = None
    for nombre, dias in (("validacion", VAL), ("test", TEST)):
        o = obs(dias)
        pr = a_sentidos(rolling(Pr, n_r, dias), dias)
        ps = rolling(Pd, n_d, dias)
        metricas[nombre] = {"dias": f"{dias[0].date()}..{dias[-1].date()}",
                            "wmape_ruta_repartida": round(wmape(pr, o), 4),
                            "wmape_por_sentido": round(wmape(ps, o), 4),
                            "wmape_promedio": round(wmape((pr + ps) / 2, o), 4)}
        print(nombre, metricas[nombre], flush=True)
        if nombre == "test":
            preds_test = ps
    v = metricas["validacion"]
    mejor = min(v["wmape_ruta_repartida"], v["wmape_por_sentido"], v["wmape_promedio"])
    assert v["wmape_por_sentido"] <= mejor * (1 + TOLERANCIA_SELECCION), "el modelo por sentido no gana en validación"

    # Bandas del 80 % para el total por ruta (lo que muestra la API): residuos fuera de muestra
    # de los últimos 14 días del test, escalados con el nivel estacional de cada ruta (como v1)
    k_cal = np.isin(TEST, DIAS_CAL)
    pred_r = np.stack([preds_test[:, [j for j, c in enumerate(sentidos) if c.split("|")[0] == r]].sum(1)
                       for r in rutas], axis=1)                        # (D, R, 24)
    obs_r = np.stack([Yr.loc[d:d + pd.Timedelta(hours=23)].to_numpy(float).T for d in TEST])
    esc = np.stack([V.escala_local(Pr, Pr.pos(d)) for d in TEST])
    alfa = 1 - V.COBERTURA
    por_hora = {}
    for h in range(24):
        r = ((obs_r[k_cal, :, h] - pred_r[k_cal, :, h]) / esc[k_cal, :, h]).ravel()
        por_hora[str(h)] = {"q_lo": V._cuantil_conformal(-r, 1 - alfa / 2),
                            "q_hi": V._cuantil_conformal(r, 1 - alfa / 2)}
    cobertura = float(((obs_r >= pred_r - np.array([[por_hora[str(h)]["q_lo"] for h in range(24)]]) * esc)
                       & (obs_r <= pred_r + np.array([[por_hora[str(h)]["q_hi"] for h in range(24)]]) * esc))
                      [~k_cal].mean())
    conformal = {"modelo": "lightgbm_sentido (suma por ruta)", "cobertura": V.COBERTURA, "tipo": "residuo",
                 "dias_calibracion": [str(d.date()) for d in DIAS_CAL], "por_hora": por_hora,
                 "cobertura_empirica_test_sin_calibracion": round(cobertura, 3)}

    # Modelo final con todo 2025
    final = V.LGBMDirecto(n_estimators=n_d, params=PARAMS).fit(Pd, len(Pd.index))
    final.model.booster_.save_model(str(CARPETA / "lightgbm_sentido.txt"))
    demanda = {
        "version": "1.2.0",
        "descripcion": "LightGBM global sobre las series ruta × sentido; el total por ruta es la suma",
        "series_sentido": sentidos, "features": V.LGB_FEATS, "n_estimators": n_d, "parametros": PARAMS,
        "entrenado_hasta": "2025-12-31 23:00:00", "archivo": "lightgbm_sentido.txt",
        "conformal": "conformal_sentido.json",
        "protocolo": "early stopping <= 2025-09-30 / 2025-10-01..14; rolling-origin con refit cada "
                     "7 días; selección en validación con tolerancia 1 % (el más simple)",
        "metricas_nivel_ruta_sentido": metricas,
        "modelo_anterior": {"archivo": "lightgbm.txt", "n_estimators": n_r,
                            "nota": "por ruta, repartido por sentido con participaciones históricas"},
    }
    (CARPETA / "demanda_sentido.json").write_text(json.dumps(demanda, ensure_ascii=False, indent=2), encoding="utf-8")
    (CARPETA / "conformal_sentido.json").write_text(json.dumps(conformal, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat([ctx[rutas], Yd], axis=1).to_parquet(CTX)
    print("n_estimators", n_d, "| cobertura 80 % en test (sin días de calibración):", round(cobertura, 3))


if __name__ == "__main__":
    main()
