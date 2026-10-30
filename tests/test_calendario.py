import datetime as dt

from services import calendario as C


def test_pascua_y_semana_santa():
    assert C.pascua(2025) == dt.date(2025, 4, 20)
    assert C.pascua(2026) == dt.date(2026, 4, 5)
    assert C.pascua(2027) == dt.date(2027, 3, 28)
    f26 = C.feriados(2026)
    assert f26[dt.date(2026, 4, 2)] == "Jueves Santo"
    assert f26[dt.date(2026, 4, 3)] == "Viernes Santo"
    assert len(C.feriados(2025)) == 16


def test_coincide_con_feriados_del_modelo():
    from model import model_atu_v2 as V
    modelo = {dt.date.fromisoformat(d) for d in V.FERIADOS}
    calc = set(C.feriados(2025)) | set(C.feriados(2026))
    assert modelo == calc


def test_dia_normal_mismo_dia_de_semana():
    r = C.dia_referencia(dt.date(2026, 10, 29))     # jueves
    assert r.referencia == dt.date(2025, 10, 30)
    assert r.referencia.weekday() == dt.date(2026, 10, 29).weekday()
    assert r.tipo == "normal"


def test_feriado_usa_mismo_feriado():
    r = C.dia_referencia(dt.date(2026, 7, 28))
    assert r.referencia == dt.date(2025, 7, 28) and r.tipo == "feriado"
    r = C.dia_referencia(dt.date(2026, 4, 3))       # Viernes Santo 2026
    assert r.referencia == dt.date(2025, 4, 18)
    r = C.dia_referencia(dt.date(2027, 1, 1))       # Año Nuevo: 2025-01-01 no tiene 28 días previos
    assert r.referencia in C.feriados(2025) and r.referencia >= C.PRIMER_DIA_CON_CONTEXTO


def test_evita_paros_y_feriados_como_referencia():
    # 2026-10-01 es jueves; 364 días antes cae 2025-10-02 (paro) -> se desplaza una semana
    r = C.dia_referencia(dt.date(2026, 10, 1))
    assert C.tipo_dia(r.referencia)[0] == "normal"
    assert r.referencia.weekday() == 3
    # 2026-10-07 (miércoles) -> 2025-10-08 es feriado (Angamos) -> otra semana
    r = C.dia_referencia(dt.date(2026, 10, 7))
    assert C.tipo_dia(r.referencia)[0] == "normal" and r.referencia.weekday() == 2


def test_cualquier_dia_de_2026_y_2027_tiene_referencia_valida():
    d = dt.date(2026, 1, 1)
    while d <= dt.date(2027, 12, 31):
        r = C.dia_referencia(d)
        assert C.PRIMER_DIA_CON_CONTEXTO <= r.referencia <= C.ULTIMO_DIA_DATOS
        if r.tipo == "normal":
            assert r.referencia.weekday() == d.weekday()
        d += dt.timedelta(days=1)
