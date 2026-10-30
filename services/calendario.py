"""Calendario de Lima: hora local, feriados nacionales y día de referencia con datos ATU.

Los datos de validaciones que entregó la ATU cubren 2025. Para pronosticar un día posterior
(p. ej. 2026-10-29) el modelo necesita los 28 días previos de demanda real, que no existen. Se
usa el **día homólogo** de 2025: mismo día de la semana, misma época del año (52 semanas antes)
y mismo tipo de día (feriado ↔ el mismo feriado, víspera ↔ víspera, día normal ↔ día normal
sin paro). Se pronostica ese día con su contexto real y la respuesta declara la referencia.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

# Perú no tiene horario de verano: UTC-5 todo el año
TZ_LIMA = dt.timezone(dt.timedelta(hours=-5), "America/Lima")

ANIO_DATOS = 2025
PRIMER_DIA_CON_CONTEXTO = dt.date(2025, 1, 29)   # 28 días de historia desde el 2025-01-01
ULTIMO_DIA_DATOS = dt.date(2025, 12, 31)

# Feriados nacionales de fecha fija (Ley 31.788 y calendario MTPE)
FERIADOS_FIJOS = {
    (1, 1): "Año Nuevo", (5, 1): "Día del Trabajo",
    (6, 7): "Batalla de Arica y Día de la Bandera", (6, 29): "San Pedro y San Pablo",
    (7, 23): "Día de la Fuerza Aérea del Perú", (7, 28): "Fiestas Patrias",
    (7, 29): "Fiestas Patrias", (8, 6): "Batalla de Junín", (8, 30): "Santa Rosa de Lima",
    (10, 8): "Combate de Angamos", (11, 1): "Todos los Santos",
    (12, 8): "Inmaculada Concepción", (12, 9): "Batalla de Ayacucho", (12, 25): "Navidad",
}
VISPERAS = {(12, 24): "Nochebuena", (12, 31): "Fin de año"}

# Paros de transportistas de 2025 (ver model_atu_v2.PAROS): no sirven como día típico
PAROS_2025 = {dt.date(2025, 2, 6), dt.date(2025, 4, 7), dt.date(2025, 4, 10), dt.date(2025, 8, 21),
              dt.date(2025, 10, 2), dt.date(2025, 10, 6), dt.date(2025, 10, 15), dt.date(2025, 11, 4)}


def ahora_lima() -> dt.datetime:
    return dt.datetime.now(TZ_LIMA)


def pascua(anio: int) -> dt.date:
    """Domingo de Pascua (algoritmo anónimo gregoriano, Meeus/Jones/Butcher)."""
    a, b, c = anio % 19, anio // 100, anio % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(anio, mes, dia)


def feriados(anio: int) -> dict[dt.date, str]:
    out = {dt.date(anio, m, d): n for (m, d), n in FERIADOS_FIJOS.items()}
    p = pascua(anio)
    out[p - dt.timedelta(days=3)] = "Jueves Santo"
    out[p - dt.timedelta(days=2)] = "Viernes Santo"
    return out


def tipo_dia(fecha: dt.date) -> tuple[str, str | None]:
    """('feriado' | 'vispera' | 'paro' | 'normal', nombre)."""
    fer = feriados(fecha.year)
    if fecha in fer:
        return "feriado", fer[fecha]
    if (fecha.month, fecha.day) in VISPERAS:
        return "vispera", VISPERAS[(fecha.month, fecha.day)]
    if fecha in PAROS_2025:
        return "paro", "Paro de transportistas"
    return "normal", None


@dataclass(frozen=True)
class Referencia:
    objetivo: dt.date
    referencia: dt.date
    tipo: str
    nombre: str | None
    criterio: str

    def to_dict(self) -> dict:
        return {"fecha_objetivo": self.objetivo.isoformat(),
                "fecha_referencia_datos": self.referencia.isoformat(),
                "tipo_dia": self.tipo, "evento": self.nombre, "criterio": self.criterio}


def _en_rango(d: dt.date) -> bool:
    return PRIMER_DIA_CON_CONTEXTO <= d <= ULTIMO_DIA_DATOS


def dia_referencia(objetivo: dt.date) -> Referencia:
    """Día de 2025 con datos reales que representa a ``objetivo``."""
    tipo, nombre = tipo_dia(objetivo)
    if _en_rango(objetivo) and tipo != "paro":
        return Referencia(objetivo, objetivo, tipo, nombre, "fecha con datos observados")

    if tipo == "feriado":
        fer25 = {n: f for f, n in feriados(ANIO_DATOS).items()}
        # Fiestas Patrias tiene dos días: se empareja por fecha (28 ↔ 28, 29 ↔ 29)
        candidatos = [f for f, n in feriados(ANIO_DATOS).items() if n == nombre]
        ref = next((f for f in candidatos if (f.month, f.day) == (objetivo.month, objetivo.day)),
                   fer25.get(nombre))
        if ref is not None and _en_rango(ref):
            return Referencia(objetivo, ref, tipo, nombre, f"mismo feriado en {ANIO_DATOS}")
        # p. ej. Año Nuevo: el 2025-01-01 no tiene 28 días de historia -> otro feriado de 2025,
        # preferentemente del mismo día de la semana y cercano en el año
        otros = [f for f in feriados(ANIO_DATOS) if _en_rango(f)]
        doy = objetivo.timetuple().tm_yday
        ref = min(otros, key=lambda f: (f.weekday() != objetivo.weekday(),
                                        min(abs(f.timetuple().tm_yday - doy),
                                            365 - abs(f.timetuple().tm_yday - doy))))
        return Referencia(objetivo, ref, tipo, nombre,
                          f"feriado de {ANIO_DATOS} más parecido ({feriados(ANIO_DATOS)[ref]})")
    if tipo == "vispera":
        ref = dt.date(ANIO_DATOS, objetivo.month, objetivo.day)
        return Referencia(objetivo, ref, tipo, nombre, f"misma víspera en {ANIO_DATOS}")

    # Día normal: 52 semanas (364 días) hacia el año con datos -> mismo día de la semana
    semanas = round((objetivo - dt.date(ANIO_DATOS, objetivo.month, min(objetivo.day, 28))).days / 364)
    ref = objetivo - dt.timedelta(weeks=52 * semanas)
    while ref > ULTIMO_DIA_DATOS:
        ref -= dt.timedelta(weeks=52)
    while ref < dt.date(ANIO_DATOS, 1, 1):
        ref += dt.timedelta(weeks=52)
    # Evitar referencias atípicas (feriado, víspera, paro) o sin 28 días de historia:
    # se desplaza ±1, ±2, ... semanas manteniendo el día de la semana
    for k in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
        cand = ref + dt.timedelta(weeks=k)
        if _en_rango(cand) and tipo_dia(cand)[0] == "normal":
            desplazo = "" if k == 0 else f" (desplazado {k:+d} semana(s) para evitar un día atípico)"
            return Referencia(objetivo, cand, tipo, nombre,
                              f"día homólogo: mismo día de la semana, 52 semanas antes{desplazo}")
    raise ValueError(f"no hay día de referencia para {objetivo}")
