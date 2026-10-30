import json
from pathlib import Path

import numpy as np
import pytest

from model.ocupacion import RutasParaderoOD, cargar_capa

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_perfil_ruta_conserva_pasajeros():
    rng = np.random.default_rng(0)
    O = rng.uniform(0, 50, (12, 24))
    O[-1] = 0                                   # en la última parada nadie sube
    W = rng.uniform(0, 1, (12, 24))
    baj, carga = RutasParaderoOD.perfil(None, O, W)
    np.testing.assert_allclose(baj.sum(0), O.sum(0), rtol=1e-9)   # todo el que sube, baja
    assert np.all(carga >= 0)
    np.testing.assert_allclose(carga[-1], 0, atol=1e-6)
    assert np.all(baj[0] < 1e-9)                # nadie baja en la primera parada


@pytest.mark.parametrize("sistema,tipo", [("metro_l1", "LineaOD"), ("troncal", "LineaOD"),
                                          ("corredor", "RutasParaderoOD"), ("alimentador", "RutasCargaCritica")])
def test_capas_cargan(sistema, tipo):
    assert type(cargar_capa(MODELS / sistema)).__name__ == tipo


def test_densidad_a_capacidad_coherente():
    # la capacidad nominal de tren y bus troncal corresponde a ~6 pasajeros/m2 de pie
    for s in ("metro_l1", "troncal"):
        v = json.loads((MODELS / s / "linea_config.json").read_text(encoding="utf-8"))["vehiculo"]
        assert 5.5 <= v["densidad_implicita_capacidad_pax_m2"] <= 6.5


def test_orden_corredores_coherente():
    cfg = json.loads((MODELS / "corredor" / "rutas_config.json").read_text(encoding="utf-8"))
    for serie, r in cfg["rutas"].items():
        largos = [s["km"][-1] for s in r["sentidos"].values()]
        assert max(largos) / min(largos) < 1.35, serie          # ida y vuelta de largo parecido
        ini = [s["nombres"][0] for s in r["sentidos"].values()]
        assert len(set(ini)) == len(ini), serie                 # cada sentido empieza en otro lado
