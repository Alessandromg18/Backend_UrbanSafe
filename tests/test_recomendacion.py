import numpy as np
import pytest

from model import recomendacion as R
from model.ocupacion import p_no_subir


def test_multiplicador_whelan_crockett():
    # sentado sin gente de pie = 1,00; de pie a 6 pax/m2 = 2,04 (Li y Hensher, 2013)
    assert R.multiplicador(30, asientos=36, area_pie=7.3) == pytest.approx(1.0)
    assert R.multiplicador(36 + 6 * 7.3, asientos=36, area_pie=7.3) == pytest.approx(2.04)
    assert R.multiplicador(36.1, asientos=36, area_pie=7.3) == pytest.approx(1.53, abs=0.01)


def test_costo_generalizado_pondera_espera_y_caminata():
    # 2 min de espera y 150 m de caminata pesan el doble que el tiempo a bordo
    assert R.costo(2.0, 10.0, 1.0, caminata_m=150.0) == pytest.approx(2 * 2 + 10 + 2 * 2)


def test_p_no_subir_por_irregularidad():
    # servicio regular (Metro) bajo capacidad: siempre sube
    assert p_no_subir(np.array([900.0]), 1200, np.array([50.0]), cv=0.0)[0] == 0.0
    # bus irregular (cv 0,5) con carga media = capacidad: ~63 % de los pasajeros ve su bus lleno
    assert p_no_subir(np.array([80.0]), 80, np.array([10.0]), cv=0.5)[0] == pytest.approx(0.629, abs=0.01)
    # más irregularidad, más riesgo con la misma carga media
    assert (p_no_subir(np.array([64.0]), 80, np.array([10.0]), cv=0.5)[0]
            > p_no_subir(np.array([64.0]), 80, np.array([10.0]), cv=0.3)[0])


def test_decidir():
    clase = np.array(["De_pie", "Saturado", "De_pie", "Sin_servicio"], dtype=object)
    pns = np.array([0.1, 0.7, 0.1, np.nan])
    aqui = np.array([30.0, 30.0, 30.0, np.nan])
    alt = np.array([np.inf, np.inf, 25.0, np.inf])
    assert list(R.decidir(clase, pns, aqui, alt)) == ["abordar", "esperar_siguiente", "cambiar_paradero", "sin_servicio"]
