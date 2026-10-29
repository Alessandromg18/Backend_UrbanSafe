import numpy as np
import pytest

from model import od_matrix as OD

# Ortúzar y Willumsen (2008), Tabla 5.6 (problema) y Tabla 5.7 (solución tras 3 iteraciones)
T56 = np.array([[5, 50, 100, 200], [50, 5, 100, 300], [50, 100, 5, 100], [100, 200, 250, 20]], float)
O56 = np.array([400, 460, 400, 702], float)
D56 = np.array([260, 400, 500, 802], float)
T57 = np.array([[5.25, 44.12, 98.24, 254.25], [45.30, 3.81, 84.78, 329.11],
                [77.04, 129.50, 7.21, 186.58], [132.41, 222.57, 309.77, 32.07]])


def test_furness_reproduce_tabla_5_7():
    T, info = OD.furness(T56, O56, D56, tol=0.0, max_iter=3)
    assert info.iteraciones == 3
    np.testing.assert_allclose(T, T57, atol=0.02)
    np.testing.assert_allclose(T.sum(0), D56, atol=1e-9)      # la última corrección es por columnas


def test_furness_converge_y_respeta_totales():
    T, info = OD.furness(T56, O56, D56, tol=1e-9)
    assert info.convergio
    np.testing.assert_allclose(T.sum(1), O56, rtol=1e-8)
    np.testing.assert_allclose(T.sum(0), D56, rtol=1e-8)


def test_furness_conserva_ceros_estructurales():
    semilla = np.ones((4, 4)) - np.eye(4)
    T, _ = OD.furness(semilla, O56, D56)
    assert np.all(np.diag(T) == 0)


def test_gravedad_beta_reduce_coste_medio():
    km = np.array([0, 2, 5, 9, 14], float)
    C = np.abs(km[:, None] - km[None, :])
    M = ~np.eye(5, dtype=bool)
    O = np.array([100, 80, 60, 40, 20], float)
    D = O[::-1].copy()
    c0 = OD.costo_medio(OD.gravedad(O, D, C, 0.0, M)[0], C)
    c1 = OD.costo_medio(OD.gravedad(O, D, C, 0.2, M)[0], C)
    assert c1 < c0


def test_hyman_alcanza_coste_objetivo():
    km = np.linspace(0, 30, 12)
    C = np.abs(km[:, None] - km[None, :])
    M = ~np.eye(12, dtype=bool)
    rng = np.random.default_rng(0)
    O, D = rng.uniform(50, 150, 12), rng.uniform(50, 150, 12)
    D *= O.sum() / D.sum()
    beta, _ = OD.calibrar_hyman(O, D, C, M, c_objetivo=8.0)
    assert OD.costo_medio(OD.gravedad(O, D, C, beta, M)[0], C) == pytest.approx(8.0, rel=1e-2)


def test_carga_por_tramo_ejemplo_a_mano():
    # 3 estaciones: 10 viajes 0->2, 5 viajes 0->1, 4 viajes 2->0
    T = np.zeros((3, 3))
    T[0, 2], T[0, 1], T[2, 0] = 10, 5, 4
    asc, desc = OD.carga_por_tramo(T)
    np.testing.assert_allclose(asc, [15, 10])     # tramo 0-1: 15 a bordo; tramo 1-2: 10
    np.testing.assert_allclose(desc, [4, 4])
    sal_a, sal_d = OD.carga_al_salir(T)
    np.testing.assert_allclose(sal_a, [15, 10, 0])
    np.testing.assert_allclose(sal_d, [0, 4, 4])


def test_conservacion_carga_igual_pasajero_tramo():
    rng = np.random.default_rng(1)
    T = rng.uniform(0, 10, (6, 6)) * (1 - np.eye(6))
    asc, desc = OD.carga_por_tramo(T)
    # suma de cargas por tramo = sum T_ij · |j - i| (tramos recorridos por cada viaje)
    i, j = np.indices(T.shape)
    np.testing.assert_allclose(asc.sum() + desc.sum(), (T * np.abs(j - i)).sum())


def test_atracciones_espejo_balancea_cada_hora():
    rng = np.random.default_rng(2)
    O = rng.uniform(0, 100, (5, 24))
    bloques = {"am": list(range(5, 10)), "md": list(range(10, 16)),
               "pm": list(range(16, 22)), "noche": [22, 23, 0, 1, 2, 3, 4]}
    espejo = {"am": "pm", "md": "md", "pm": "am", "noche": "dia"}
    D = OD.atracciones_espejo(O, bloques, espejo)
    np.testing.assert_allclose(D.sum(0), O.sum(0))
    # en la mañana, las bajadas se reparten como las subidas de la tarde
    s_pm = O[:, 16:22].sum(1) / O[:, 16:22].sum()
    np.testing.assert_allclose(D[:, 7] / D[:, 7].sum(), s_pm)
