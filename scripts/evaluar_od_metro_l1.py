"""Evaluación de la ocupación a bordo estimada con matriz O-D (Metro L1).

Uso (desde la raíz del backend)::

    python scripts/evaluar_od_metro_l1.py

Salidas: ``models/metro_l1/od_evaluacion.json`` y ``docs/EVALUACION_OD_METRO_L1.md``.

1. Recuperación sintética: matrices O-D "verdaderas" con supuestos violados a propósito
   (atracciones con ruido, beta distinto de 0); el estimador solo ve las subidas.
2. Sensibilidad a beta y simetría de la matriz diaria con datos reales 2025.
3. Contraste con datos públicos (Ositrán, horario oficial, capacidad).
4. Pronóstico en test (2025-11-01 a 2025-12-31): LightGBM reentrenado sin datos del futuro
   (rolling-origin, refit cada 7 días) -> ocupación vs ocupación calculada
   con las subidas observadas; baseline media_4sem por la misma cadena (RNF-01).
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
import od_matrix as OD  # noqa: E402
from inferencia import ModeloSistema, panel_desde_contexto  # noqa: E402
from ocupacion import cargar_capa  # noqa: E402

SIS = "metro_l1"
CARPETA = RAIZ / "models" / SIS
CTX = pd.read_parquet(RAIZ / "data" / "context" / f"{SIS}_2025.parquet")
LINEA = cargar_capa(CARPETA)
EST = LINEA.estaciones
CLASES = ["Asientos", "De_pie", "Saturado"]
rng = np.random.default_rng(2025)


def dia_matriz(d) -> np.ndarray:
    d = pd.Timestamp(d)
    return CTX.loc[d:d + pd.Timedelta(hours=23), EST].to_numpy(float).T      # (S, 24)


def dias_tipo(desde, hasta, bucket):
    out = []
    for d in pd.date_range(desde, hasta, freq="D"):
        if V.bucket_dia(d) == bucket and V.tipo_evento(d) == "normal":
            out.append(d)
    return out


def clases_de(carga_tren, servicio):
    return LINEA.clases(carga_tren, servicio)


# ── 1. Recuperación sintética ──────────────────────────────────────────────
def sintetico(n_sim: int = 40, sigma: float = 0.35) -> dict:
    dias = dias_tipo("2025-03-01", "2025-10-31", "LAB")
    th = LINEA.trenes_hora["LAB"]
    servicio = np.broadcast_to(th > 0, (len(EST), 24))
    res = {"od_retorno": [], "bajadas_uniformes": []}
    for _ in range(n_sim):
        O = dia_matriz(dias[rng.integers(len(dias))])
        beta_v = rng.uniform(0.0, 0.06)
        # atracción verdadera = espejo con ruido lognormal por estación y bloque
        Dv = np.zeros_like(O)
        D_espejo = OD.atracciones_espejo(O, LINEA.bloques, LINEA.espejo)
        for horas in LINEA.bloques.values():
            ruido = rng.lognormal(0, sigma, len(EST))
            for h in horas:
                w = D_espejo[:, h] * ruido
                Dv[:, h] = O[:, h].sum() * w / w.sum() if w.sum() > 0 else 0
        Tv = OD.od_dia(O, Dv, LINEA.C, beta_v, LINEA.mascara)
        Lv = np.stack(OD.carga_al_salir(Tv))                     # (2, 24, S)
        estimadores = {
            "od_retorno": OD.atracciones_espejo(O, LINEA.bloques, LINEA.espejo),
            "bajadas_uniformes": np.broadcast_to(O.sum(0) / len(EST), O.shape).copy(),
        }
        for nombre, De in estimadores.items():
            Te = OD.od_dia(O, De, LINEA.C, LINEA.beta, LINEA.mascara)
            Le = np.stack(OD.carga_al_salir(Te))
            m = np.broadcast_to(th > 0, Lv.shape[:2])[..., None] & np.ones_like(Lv, bool)
            wape = np.abs(Le - Lv)[m].sum() / Lv[m].sum()
            cv = np.concatenate([clases_de(Lv[k].T / np.where(th > 0, th, 1), servicio)[servicio] for k in (0, 1)])
            ce = np.concatenate([clases_de(Le[k].T / np.where(th > 0, th, 1), servicio)[servicio] for k in (0, 1)])
            pico_v = Lv.max(axis=2)[:, [7, 18]] / th[[7, 18]]
            pico_e = Le.max(axis=2)[:, [7, 18]] / th[[7, 18]]
            res[nombre].append({
                "wape_carga": wape,
                "wape_bajadas": np.abs(De - Dv).sum() / Dv.sum(),
                "acc": accuracy_score(cv, ce),
                "f1_macro": f1_score(cv, ce, labels=CLASES, average="macro", zero_division=0),
                "err_rel_pico": float(np.mean(np.abs(pico_e - pico_v) / pico_v)),
            })
    resumen = {}
    for k, filas in res.items():
        df = pd.DataFrame(filas)
        resumen[k] = {c: {"media": round(df[c].mean(), 4), "p10": round(df[c].quantile(.1), 4),
                          "p90": round(df[c].quantile(.9), 4)} for c in df.columns}
    return {"n_sim": n_sim, "sigma_ruido_atraccion": sigma, "beta_verdadero": "U(0, 0.06) por min",
            "estimador_beta": LINEA.beta, "resultados": resumen}


# ── 2. Sensibilidad a beta (datos reales) ──────────────────────────────────
def sensibilidad() -> list[dict]:
    O = np.mean([dia_matriz(d) for d in dias_tipo("2025-03-01", "2025-10-31", "LAB")], axis=0)
    D = OD.atracciones_espejo(O, LINEA.bloques, LINEA.espejo)
    th = LINEA.trenes_hora["LAB"]
    filas = []
    for beta in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]:
        T = OD.od_dia(O, D, LINEA.C, beta, LINEA.mascara)
        Td = T.sum(0)
        a, d = OD.carga_al_salir(T)
        filas.append({
            "beta_por_min": beta,
            "viaje_medio_min": round(OD.costo_medio(Td, LINEA.C), 1),
            "viaje_medio_km": round(OD.costo_medio(Td, LINEA.C * LINEA.cfg["velocidad_comercial_kmh"] / 60), 1),
            "asimetria_diaria": round(float(np.abs(Td - Td.T).sum() / Td.sum() / 2), 4),
            "pico_am_por_tren": round(float(max(a[7].max(), d[7].max()) / th[7])),
            "pico_pm_por_tren": round(float(max(a[18].max(), d[18].max()) / th[18])),
        })
    return filas


# ── 3. Contraste con el mundo real ─────────────────────────────────────────
def contraste() -> dict:
    lab = dias_tipo("2025-03-01", "2025-10-31", "LAB")
    O = np.mean([dia_matriz(d) for d in lab], axis=0)
    r = LINEA.aplicar(O, EST, "LAB")
    out = {"subidas_dia_laborable": round(float(O.sum())),
           "subidas_2025_total": round(float(CTX[EST].to_numpy().sum())),
           "referencia_ositran": "191,1 millones de pasajeros en 2024 (Informe de Desempeño 2024)",
           "longitud_km": LINEA.cfg["longitud_km"],
           "referencia_longitud": "34 km (Ositrán), 34,6 km (Wikipedia); Grau en km 22,1 (Ositrán: tramo 1)",
           "km_miguel_grau": LINEA.cfg["km"][EST.index("Miguel Grau")],
           "tramos_criticos": {}}
    for h, et in ((7, "07:00"), (8, "08:00"), (18, "18:00")):
        for s in ("asc", "desc"):
            ct = r.carga_tren[s][:, h]
            k = int(np.nanargmax(ct))
            sig = k + 1 if s == "asc" else k - 1
            out["tramos_criticos"][f"{et} {LINEA.sentidos[s]}"] = {
                "tramo": f"{EST[k]} -> {EST[sig]}", "pasajeros_por_tren": round(float(ct[k])),
                "ocupacion": round(float(ct[k] / LINEA.capacidad), 3), "clase": str(r.clase[s][k, h]),
                "trenes_hora": float(LINEA.trenes_hora['LAB'][h])}
    return out


# ── 4. Pronóstico en test ──────────────────────────────────────────────────
def evaluar_pronostico() -> dict:
    m = ModeloSistema.cargar(SIS, verificar_hash=False)
    dias = pd.date_range("2025-11-01", "2025-12-31", freq="D")
    from evaluar_ocupacion import pronosticos_fuera_de_muestra
    OOS = pronosticos_fuera_de_muestra(SIS, dias)
    y_obs, y_lgb, y_base = {s: [] for s in ("asc", "desc")}, {s: [] for s in ("asc", "desc")}, {s: [] for s in ("asc", "desc")}
    c_obs, c_lgb, c_base = [], [], []
    rec_obs, rec_lgb = [], []
    tiempos = []
    for d in dias:
        P, p = panel_desde_contexto(SIS, CTX[m.series], d)
        t = time.perf_counter()
        pt = OOS[pd.Timestamp(d)]          # LightGBM reentrenado sin datos del futuro
        base, _ = V.f_media_4sem(P, p)
        obs = CTX.loc[d:d + pd.Timedelta(hours=23), m.series].to_numpy(float).T
        so, sl, sb = (m.salida_operativa(m.series, Y, d) for Y in (obs, pt, base))
        tiempos.append(time.perf_counter() - t)
        for s in ("asc", "desc"):
            serv = so[f"aforo_{s}"] != "Sin_servicio"
            y_obs[s].append(so[f"carga_{s}"][serv]); y_lgb[s].append(sl[f"carga_{s}"][serv])
            y_base[s].append(sb[f"carga_{s}"][serv])
            c_obs.append(so[f"aforo_{s}"][serv]); c_lgb.append(sl[f"aforo_{s}"][serv]); c_base.append(sb[f"aforo_{s}"][serv])
            rec_obs.append(so[f"recomendacion_{s}"][serv]); rec_lgb.append(sl[f"recomendacion_{s}"][serv])

    def met_carga(pred):
        yo = np.concatenate([np.concatenate(y_obs[s]) for s in ("asc", "desc")])
        yp = np.concatenate([np.concatenate(pred[s]) for s in ("asc", "desc")])
        return {"MAE_pasajeros_tren": round(float(np.abs(yp - yo).mean()), 1),
                "WMAPE": round(float(np.abs(yp - yo).sum() / yo.sum()), 4)}

    def met_clase(pred):
        yo, yp = np.concatenate(c_obs).astype(str), np.concatenate(pred).astype(str)
        return {"accuracy": round(accuracy_score(yo, yp), 4),
                "f1_macro": round(f1_score(yo, yp, labels=CLASES, average="macro", zero_division=0), 4),
                "f1_weighted": round(f1_score(yo, yp, labels=CLASES, average="weighted", zero_division=0), 4),
                "balanced_accuracy": round(balanced_accuracy_score(yo, yp), 4)}

    ro, rl = np.concatenate(rec_obs).astype(str), np.concatenate(rec_lgb).astype(str)
    dist = pd.Series(np.concatenate(c_obs).astype(str)).value_counts(normalize=True).round(4).to_dict()
    return {
        "dias_test": f"{dias[0].date()} a {dias[-1].date()} ({len(dias)} días)",
        "verdad": "ocupación calculada con las subidas observadas por la misma cadena O-D",
        "lightgbm": {**met_carga(y_lgb), **met_clase(c_lgb)},
        "media_4sem": {**met_carga(y_base), **met_clase(c_base)},
        "distribucion_clases_observadas": dist,
        "recomendacion_acuerdo": round(float((ro == rl).mean()), 4),
        "recomendacion_siempre_abordar": round(float((ro == "abordar").mean()), 4),
        "segundos_por_dia_pronostico_mas_od": round(float(np.median(tiempos)), 3),
        "RNF_01": {"umbral": "accuracy >= 0.75 y F1-macro >= 0.72"},
    }


def main() -> None:
    t0 = time.time()
    out = {"sistema": SIS, "metodo": LINEA.cfg["od"]["metodo"], "beta": LINEA.beta,
           "sintetico": sintetico(), "sensibilidad_beta": sensibilidad(),
           "contraste_real": contraste(), "pronostico_test": evaluar_pronostico()}
    pt = out["pronostico_test"]["lightgbm"]
    out["pronostico_test"]["RNF_01"]["cumple"] = bool(pt["accuracy"] >= 0.75 and pt["f1_macro"] >= 0.72)
    (CARPETA / "od_evaluacion.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
