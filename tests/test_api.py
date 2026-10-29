import time

import numpy as np
import pytest


@pytest.fixture(scope="module")
def client():
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


def test_prediccion_metro_hora_con_sentido(client):
    r = client.get("/api/predictions/metro_l1/Gamarra?dia=2026-10-29&hour=7&hacia=Bayovar")
    assert r.status_code == 200
    d = r.get_json()
    assert d["datetime"] == "2026-10-29 07:00"                 # fecha objetivo real, no 2025
    assert d["referencia"]["fecha_referencia_datos"] == "2025-10-30"
    assert d["sentido"] == "asc" and d["metodo_aforo"] == "ocupacion_od_v2"
    assert d["crowding"]["class"] in ("Asientos", "De_pie", "Saturado")
    assert 0 < d["crowding"]["index"] < 3
    assert d["eta"]["next"] == pytest.approx(60 / 20 / 2, abs=0.6)   # 3 min de intervalo en punta


def test_forecast_compatible_con_frontend(client):
    r = client.get("/api/predictions/metro_l1/Angamos/forecast?dia=2026-10-29&hours=24")
    assert r.status_code == 200
    preds = r.get_json()["predictions"]
    assert len(preds) == 24
    for k in ("datetime", "value", "crowding", "class", "eta_min", "eta2_min", "recommendation"):
        assert k in preds[0]
    noche = [p for p in preds if p["datetime"].endswith("03:00")][0]
    assert noche["class"] == "Sin_servicio" and noche["recommendation"] == "sin_servicio"


def test_serie_sin_datos_no_inventa(client):
    r = client.get("/api/predictions/metro_l1/Naranjal?dia=2026-10-29")
    assert r.status_code == 404


def test_nombres_sin_tildes(client):
    a = client.get("/api/predictions/metro_l1/Bayovar?dia=2026-10-29&hour=8").get_json()
    b = client.get("/api/predictions/metro_l1/Bay%C3%B3var?dia=2026-10-29&hour=8").get_json()
    assert a["series"] == b["series"] == "Bayóvar"


def test_estacion_por_id_una_llamada(client):
    est = client.get("/api/stations?system=metro_l1").get_json()
    assert all(e["model_series"] for e in est)
    gamarra = next(e for e in est if e["name"] == "Gamarra")
    vmt = next(e for e in est if e["name"] == "Villa Maria")
    r = client.get(f"/api/predictions/estacion/{gamarra['id']}?dia=2026-10-29&hour=18&hacia_id={vmt['id']}")
    assert r.status_code == 200
    d = r.get_json()
    assert d["sentido"] == "desc" and len(d["forecast"]) == 24
    assert d["current"]["datetime"] == "2026-10-29 18:00"


def test_paraderos_sin_datos_devuelven_null():
    from routes.stations import load_all_stations_into_memory
    est = load_all_stations_into_memory()
    corredor_sin_modelo = [e for e in est if e["system"] == "corredores"
                           and e["raw_properties"].get("ruta") == "306"]
    assert corredor_sin_modelo and all(e["model_series"] is None for e in corredor_sin_modelo)


def test_latencia_p95_en_caliente(client):
    client.get("/api/predictions/metro_l1/Gamarra/forecast?dia=2026-10-29")   # calienta caché
    t = []
    for est in ["Gamarra", "Angamos", "La Cultura", "Bayovar", "Atocongo"] * 4:
        t0 = time.perf_counter()
        assert client.get(f"/api/predictions/metro_l1/{est}?dia=2026-10-29&hour=8").status_code == 200
        t.append(time.perf_counter() - t0)
    assert np.percentile(t, 95) <= 2.0      # AC-04
