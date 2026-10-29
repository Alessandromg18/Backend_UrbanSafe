"""Construye la configuración de ocupación por ruta de **corredores** y **alimentadores**.

Uso (desde la raíz del backend; los parquet limpios están en el repo DPD_project)::

    python scripts/construir_rutas_bus.py --datos ../../DPD_project/deliveries/week10/data_processed/clean

Salidas:
- ``models/corredor/rutas_config.json`` + ``rutas_participaciones.npz``
- ``models/alimentador/rutas_config.json`` + ``rutas_participaciones.npz``
- ``data/eval/<sistema>_paradas_test.parquet`` (subidas observadas por paradero, nov-dic 2025,
  solo para ``scripts/evaluar_od_rutas.py``)

Las participaciones (subidas del paradero / subidas de la ruta, por tipo de día y hora) se
estiman con ene-oct 2025, sin los días de test ni paros.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from services import calendario  # noqa: E402

BUCKETS = ("LAB", "SAB", "DOM")
FIN_TRAIN = pd.Timestamp("2025-10-31")
INI_TEST = pd.Timestamp("2025-11-01")

FUENTES = {
    "infobae_rojo_2026": "https://www.infobae.com/peru/2026/10/16/corredor-rojo-rutas-201-204-206-y-209-paraderos-y-mapa-completo/",
    "moovit_201": "https://moovitapp.com/index/en/public_transit-line-201-Lima-1102-858864-774953-1",
    "moovit_301": "https://moovitapp.com/index/en/public_transit-line-301-Lima-1102-858863-774948-0",
    "moovit_305": "https://moovitapp.com/index/en/public_transit-line-305-Lima-1102-858863-82303275-0",
    "titan_urbano_12m": "https://transportepesadognv.calidda.com.pe/wp-content/uploads/2025/06/ft_modasa_titan_12m.pdf",
    "titan_urbano_gnv": "http://www.busesycamiones.pe/carrocerias/596-carroceria-modasa-titan-urbano-gnv",
    "alimentadores_buses": "https://www.infobae.com/peru/2023/03/14/alimentadores-del-metropolitano-diferencias-entre-los-buses-y-cuales-son-las-rutas/",
    "alimentadores_2010": "https://andina.pe/agencia/noticia-buses-alimentadores-de-metropolitano-solo-trasladaran-pasajeros-hasta-las-estaciones-comas-y-chorrillos-274903.aspx",
}

# ── Oferta: flota publicada repartida entre rutas (método del punto de carga máxima) ──
# No hay intervalos oficiales por ruta y hora: los publicados (5-7 min la 201, 10 min la 204) no
# alcanzan para la demanda medida en punta (la 201 sube ~6000 pasajeros/h), y el operador del
# Corredor Rojo declaró "unidades cada 2 minutos en hora punta" (El Comercio). Se reparte la flota
# en servicio de cada corredor entre sus rutas con la regla clásica de diseño de frecuencias
# (Ceder, *Public Transit Planning and Operation*): f_r = L_max,r / k, con L_max,r la carga en el
# punto de máxima carga en la hora punta y k común, y sum_r f_r · T_ciclo,r = flota.
FLOTAS = {
    "corredor": {
        "2": (225, "Corredor Rojo: 225 buses operando + 20 de reserva (El Comercio)"),
        "3": (round(157 * 225 / 294), "Corredor Azul: 125 buses + 32 midibuses (Wikipedia) x 225/294 en servicio (relación del Rojo)"),
        "4": (152, "Corredor Morado: 152 buses programados en día laborable (ATU, mar. 2024)"),
    },
    "alimentador": {"*": (round(223 * 225 / 294), "Alimentadores: 223 buses (plan MTC de renovación) x 225/294 en servicio")},
}
VELOCIDAD_KMH = {"corredor": 26.0, "alimentador": 20.0}   # 201: 31,4 km en ~72 min; alimentador: supuesto
FACTOR_CICLO = 1.10                                      # +10 % de regulación en terminales (supuesto)
# Perfil horario de frecuencias relativo a la punta (Metropolitano, Infobae oct. 2026: punta 2-4
# min, valle 5-7, primera y última hora 8-12 -> 1 : 0,5 : 0,3) y por tipo de día (intervalos
# publicados de la 201, 204 y 301: sábado ~0,8 y domingo ~0,65 de la frecuencia laborable)
RATIO_HORA = {h: 1.0 for h in (6, 7, 8, 17, 18, 19)} | {h: 0.5 for h in (*range(9, 17), 20, 21)} | {5: 0.3, 22: 0.3}
RATIO_DIA = {"LAB": 1.0, "SAB": 0.8, "DOM": 0.65}

ALIMENTADORES_SUR = {"Cedros de Villa", "Villa El Salvador", "Las Américas", "Los Próceres"}


def vehiculo(capacidad: float, asientos: float, fuente: str) -> dict:
    # Densidad a capacidad nominal ≈ 6 pie/m2 (verificado en Metro L1: 6,15 y Metropolitano:
    # 6,1). "Saturado" > 4 pie/m2, el mismo criterio que en Metro L1.
    de_pie = capacidad - asientos
    return {"capacidad": capacidad, "asientos": asientos, "fuente": fuente,
            "densidad_capacidad_pax_m2": 6.0,
            "umbral_saturado_pasajeros": round(asientos + 4.0 * de_pie / 6.0, 1)}


VEH_CORREDOR = vehiculo(80, 36, "Modasa Titán Urbano 12 m: 80 pasajeros (ficha GNV); 36 asientos (ficha 2025)")
VEH_ALIM_NORTE = vehiculo(40, 18, "Bus naranja 8,5 m: 40 pasajeros (Infobae 2023); asientos supuestos (45 %)")
VEH_ALIM_SUR = vehiculo(80, 36, "Bus amarillo 12 m: 80 pasajeros (Infobae 2023, ruta sur); asientos como el 12 m")


def perfil(f_punta: float) -> dict:
    return {b: [round(f_punta * RATIO_DIA[b] * RATIO_HORA.get(h, 0.0), 3) for h in range(24)]
            for b in BUCKETS}


def dia_laborable_medio(sistema: str, series: list) -> np.ndarray:
    """Subidas medias (R, 24) de un día laborable normal (feb-oct 2025)."""
    ctx = pd.read_parquet(RAIZ / "data" / "context" / f"{sistema}_2025.parquet")
    dias = [d for d in pd.date_range("2025-02-03", FIN_TRAIN, freq="D")
            if d.weekday() < 5 and calendario.tipo_dia(d.date())[0] == "normal"]
    return np.mean([ctx.loc[d:d + pd.Timedelta(hours=23), series].to_numpy(float).T for d in dias], axis=0)


def asignar_oferta(sistema: str, cfg: dict, npz: dict, largo_km: dict, grupo_de, frac_modelada: dict) -> list:
    """Buses por hora de cada ruta con la flota publicada del corredor (ver FLOTAS)."""
    sys.path.insert(0, str(RAIZ / "model"))
    from ocupacion import RutasCargaCritica, RutasParaderoOD
    for r in cfg["rutas"].values():
        r["buses_hora"] = {b: [1.0] * 24 for b in BUCKETS}     # 1 vehículo/h -> carga = pasajeros/h
    capa = (RutasParaderoOD if cfg["tipo"] == "rutas_paradero" else RutasCargaCritica)(cfg, npz)
    series = list(cfg["rutas"])
    res = capa.aplicar(dia_laborable_medio(sistema, series), series, "LAB")
    L = np.maximum(res.carga_tren["asc"], res.carga_tren["desc"]).astype(float)   # (R, 24) pax/h
    grupos: dict[str, list[int]] = {}
    for i, s in enumerate(series):
        grupos.setdefault(grupo_de(s), []).append(i)
    resumen = []
    for g, idx in grupos.items():
        flota, fuente = FLOTAS[sistema][g]
        h_punta = int(np.argmax(L[idx].sum(0)))
        lmax = L[idx, h_punta]
        ciclo_h = np.array([2 * largo_km[series[i]] / VELOCIDAD_KMH[sistema] * FACTOR_CICLO for i in idx])
        flota_mod = flota * frac_modelada.get(g, 1.0)
        k = float((lmax * ciclo_h).sum() / flota_mod)          # pasajeros por bus en el punto crítico
        for j, i in enumerate(idx):
            f = float(np.clip(lmax[j] / k, 3.0, 60.0))
            r = cfg["rutas"][series[i]]
            r["buses_hora"] = perfil(f)
            r["oferta"] = {"buses_hora_punta": round(f, 2), "intervalo_punta_min": round(60 / f, 2),
                           "carga_max_punta_pax_h": round(float(lmax[j])), "hora_punta": h_punta,
                           "largo_km": round(largo_km[series[i]], 1), "ciclo_h": round(float(ciclo_h[j]), 2),
                           "flota_grupo": flota, "fraccion_flota_rutas_modeladas": round(frac_modelada.get(g, 1.0), 3),
                           "fuente_flota": fuente}
            resumen.append((series[i], round(60 / f, 1), round(float(lmax[j])), round(k, 1)))
    cfg["oferta"] = {
        "metodo": "flota publicada / tiempo de ciclo, repartida por carga máxima en punta (Ceder)",
        "velocidad_comercial_kmh": VELOCIDAD_KMH[sistema], "factor_ciclo": FACTOR_CICLO,
        "ratio_hora": RATIO_HORA, "ratio_dia": RATIO_DIA,
        "fuente_ratios": "Infobae oct. 2026 (Metropolitano: punta 2-4, valle 5-7, extremos 8-12 min); "
                         "intervalos publicados 201/204/301 (sábado y domingo)",
    }
    return resumen


def bucket(fecha: pd.Timestamp) -> str:
    d = fecha.date()
    if calendario.tipo_dia(d)[0] == "feriado" or d.weekday() == 6:
        return "DOM"
    return "SAB" if d.weekday() == 5 else "LAB"


def participaciones(df: pd.DataFrame, clave: str, series: list) -> tuple[dict, dict]:
    """share[serie] (n_unidades, 3, 24) con unidades = valores de ``clave`` dentro de la serie."""
    df = df[df["fecha"] <= FIN_TRAIN].copy()
    df = df[~df["fecha"].dt.date.isin(calendario.PAROS_2025)]
    fechas = pd.Series(df["fecha"].unique())
    bmap = dict(zip(fechas, [bucket(pd.Timestamp(f)) for f in fechas]))
    df["bucket"] = df["fecha"].map(bmap)
    agg = df.groupby(["serie", clave, "bucket", "hora"], observed=True)["validaciones"].sum()
    shares, unidades = {}, {}
    for s in series:
        sub = agg.loc[s] if s in agg.index.get_level_values(0) else None
        if sub is None:
            continue
        u = sorted(sub.index.get_level_values(0).unique())
        arr = np.zeros((len(u), 3, 24))
        for (unidad, b, h), v in sub.items():
            arr[u.index(unidad), BUCKETS.index(b), int(h)] = v
        tot = arr.sum(0, keepdims=True)
        shares[s] = np.divide(arr, tot, out=np.zeros_like(arr), where=tot > 0)
        unidades[s] = u
    return shares, unidades


def haversine_km(a, b) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    x = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(x))


def ordenar(paradas: list[tuple[str, int, tuple]], sufijo: str) -> list[str]:
    """Orden de recorrido: secuencia principal del catálogo ATU (bloque más largo de
    ``n_paradero`` consecutivos) + inserción más barata de las paradas agregadas después."""
    paradas = sorted(paradas, key=lambda p: p[1])
    bloques, actual = [], [paradas[0]]
    for p in paradas[1:]:
        if p[1] - actual[-1][1] <= 5:
            actual.append(p)
        else:
            bloques.append(actual)
            actual = [p]
    bloques.append(actual)
    base = max(bloques, key=len)
    ruta = list(base)
    for p in [q for q in paradas if q not in base]:
        mejor, pos = math.inf, 0
        for k in range(len(ruta) + 1):
            if k == 0:
                costo = haversine_km(p[2], ruta[0][2])
            elif k == len(ruta):
                costo = haversine_km(ruta[-1][2], p[2])
            else:
                costo = (haversine_km(ruta[k - 1][2], p[2]) + haversine_km(p[2], ruta[k][2])
                         - haversine_km(ruta[k - 1][2], ruta[k][2]))
            if costo < mejor:
                mejor, pos = costo, k
        ruta.insert(pos, p)
    ruta = dos_opt(ruta)
    # Orientar con el sufijo del código ATU (NS = norte->sur, SN, EO = este->oeste, OE);
    # verificado con la secuencia del catálogo en las rutas 201 y 305
    (lon0, lat0), (lon1, lat1) = ruta[0][2], ruta[-1][2]
    al_reves = {"NS": lat0 < lat1, "SN": lat0 > lat1, "EO": lon0 < lon1, "OE": lon0 > lon1}.get(sufijo, False)
    if al_reves:
        ruta = ruta[::-1]
    return [p[0] for p in ruta]


def largo(ruta) -> float:
    return sum(haversine_km(a[2], b[2]) for a, b in zip(ruta, ruta[1:]))


def dos_opt(ruta: list) -> list:
    """Mejora 2-opt de un camino abierto: invierte subtramos mientras se acorte el recorrido."""
    mejoro = True
    while mejoro:
        mejoro = False
        for i in range(-1, len(ruta) - 2):          # i = -1 permite invertir el inicio
            for j in range(i + 2, len(ruta)):
                nuevo = ruta[:i + 1] + ruta[i + 1:j + 1][::-1] + ruta[j + 1:]
                if largo(nuevo) < largo(ruta) - 1e-9:
                    ruta, mejoro = nuevo, True
    return ruta


def corredor(datos: Path) -> None:
    card = json.loads((RAIZ / "models" / "corredor" / "model_card.json").read_text(encoding="utf-8"))
    series = card["series"]
    df = pd.read_parquet(datos / "corredores_hora.parquet",
                         columns=["fecha", "ruta", "cod_paradero", "paradero", "hora", "validaciones"])
    # fracción de la demanda de cada corredor (primer dígito de la ruta) que va en rutas modeladas
    vol = df.groupby("ruta")["validaciones"].sum()
    grp = vol.groupby(vol.index.str[0]).sum()
    frac = {g: float(vol[vol.index.isin(series) & (vol.index.str[0] == g)].sum() / grp[g]) for g in grp.index}
    df = df[df["ruta"].isin(series)].rename(columns={"ruta": "serie"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    shares, unidades = participaciones(df, "cod_paradero", series)
    nombres = df.drop_duplicates(["serie", "cod_paradero"]).set_index(["serie", "cod_paradero"])["paradero"]

    geo = json.loads((RAIZ / "data" / "geo" / "puntos_corredores.geojson").read_text(encoding="utf-8"))
    coord, npar = {}, {}
    for f in geo["features"]:
        p = f["properties"]
        coord[p["cod_paradero"]] = tuple(f["geometry"]["coordinates"])
        npar[(p["ruta"], p["cod_paradero"])] = int(p["n_paradero"])

    rutas, npz, reporte = {}, {}, []
    for s in series:
        u = unidades[s]
        share_tot = shares[s].sum(axis=(1, 2))
        sentidos = {}
        perdida = 0.0
        for suf in sorted({c[-2:] for c in u}):
            cands = [(c, npar.get((s, c), 10_000), coord[c]) for c in u if c[-2:] == suf and c in coord]
            perdida += sum(share_tot[u.index(c)] for c in u if c[-2:] == suf and c not in coord)
            if len(cands) < 2:
                perdida += sum(share_tot[u.index(c)] for c, _, _ in cands)
                continue
            orden = ordenar(cands, suf)
            km = [0.0]
            for a, b in zip(orden, orden[1:]):
                km.append(km[-1] + haversine_km(coord[a], coord[b]))
            sentidos[suf] = {"paradas": orden, "km": [round(x, 3) for x in km],
                             "nombres": [str(nombres.get((s, c), c)) for c in orden],
                             "descripcion": f"hacia {nombres.get((s, orden[-1]), orden[-1])}"}
        # pareja (paradero del sentido opuesto más cercano) para el viaje de retorno
        sufs = list(sentidos)
        for suf in sufs:
            otros = [x for x in sufs if x != suf]
            if not otros:
                sentidos[suf]["pareja"] = sentidos[suf]["paradas"]
                continue
            opp = sentidos[otros[0]]["paradas"]
            sentidos[suf]["pareja"] = [min(opp, key=lambda q: haversine_km(coord[c], coord[q]))
                                       for c in sentidos[suf]["paradas"]]
        rutas[s] = {"unidades": u, "sentidos": sentidos,
                    "demanda_sin_ubicar": round(float(perdida) / 72, 4)}
        npz[s] = shares[s]
        reporte.append((s, len(u), {k: len(v["paradas"]) for k, v in sentidos.items()},
                        rutas[s]["demanda_sin_ubicar"]))
    cfg = {"sistema": "corredor", "version": "2.0.0", "tipo": "rutas_paradero",
           "metodo": "gravedad acotada a orígenes por ruta y sentido (Ortúzar ec. 5.16); bajadas por "
                     "viaje de retorno con el paradero del sentido opuesto; beta = 0 (ME2)",
           "vehiculo": VEH_CORREDOR, "rutas": rutas, "fuentes": FUENTES,
           "od": {"bloques": {"am": list(range(5, 10)), "md": list(range(10, 16)),
                              "pm": list(range(16, 22)), "noche": [22, 23, 0, 1, 2, 3, 4]},
                  "espejo": {"am": "pm", "md": "md", "pm": "am", "noche": "dia"}}}
    largo = {s: float(np.mean([v["km"][-1] for v in r["sentidos"].values()])) for s, r in rutas.items()}
    for fila in asignar_oferta("corredor", cfg, npz, largo, lambda s: s[0], frac):
        print("oferta corredor (ruta, intervalo punta min, carga max pax/h, pax/bus diseño):", fila)
    escribir("corredor", cfg, npz)
    test = df[df["fecha"] >= INI_TEST].groupby(["fecha", "hora", "serie", "cod_paradero"])["validaciones"].sum()
    guardar_test("corredor", test.reset_index().rename(columns={"cod_paradero": "unidad"}))
    for r in reporte:
        print("corredor", *r)


def alimentador(datos: Path) -> None:
    card = json.loads((RAIZ / "models" / "alimentador" / "model_card.json").read_text(encoding="utf-8"))
    series = card["series"]
    df = pd.read_parquet(datos / "alimentador_hora.parquet",
                         columns=["fecha", "ruta", "sentido_norm", "paradero", "hora", "validaciones"])
    vol = df.groupby("ruta")["validaciones"].sum()
    frac = {"*": float(vol[vol.index.isin(series)].sum() / vol.sum())}
    df = df[df["ruta"].isin(series)].rename(columns={"ruta": "serie", "sentido_norm": "sentido"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    shares, unidades = participaciones(df, "sentido", series)
    # sentido que sale del terminal: el que concentra subidas en la estación de integración
    term = df["paradero"].str.contains("Estaci|Terminal|Embarque", case=False, na=False)
    tot = df.groupby(["serie", "sentido"])["validaciones"].sum()
    en_term = df[term].groupby(["serie", "sentido"])["validaciones"].sum().reindex(tot.index).fillna(0)
    frac_term = (en_term / tot).fillna(0)
    rutas, npz = {}, {}
    for s in series:
        sentidos = {}
        for sen in unidades[s]:
            ft = float(frac_term.get((s, sen), 0.0))
            desc = ("saliendo de la estación de integración" if ft >= 0.5 else
                    "hacia la estación de integración" if any(frac_term.get((s, o), 0) >= 0.5
                                                               for o in unidades[s] if o != sen)
                    else f"sentido {sen}")
            sentidos[sen] = {"descripcion": desc, "fraccion_subidas_en_terminal": round(ft, 3)}
        veh = VEH_ALIM_SUR if s in ALIMENTADORES_SUR else VEH_ALIM_NORTE
        rutas[s] = {"unidades": unidades[s], "sentidos": sentidos, "vehiculo": veh}
        npz[s] = shares[s]
    cfg = {"sistema": "alimentador", "version": "2.0.0", "tipo": "rutas_carga_critica",
           "metodo": "carga máxima del bus = subidas del sentido por hora / buses por hora: en un "
                     "alimentador casi todos los viajes empiezan o terminan en la estación de "
                     "integración, así que el tramo junto al terminal lleva a todos los pasajeros "
                     "del sentido (cota superior cuando hay viajes locales)",
           "rutas": rutas, "fuentes": FUENTES}
    for fila in asignar_oferta("alimentador", cfg, npz, largo_alimentadores(series), lambda s: "*", frac):
        print("oferta alimentador (ruta, intervalo punta min, carga max pax/h, pax/bus diseño):", fila)
    escribir("alimentador", cfg, npz)
    test = df[df["fecha"] >= INI_TEST].groupby(["fecha", "hora", "serie", "sentido"])["validaciones"].sum()
    guardar_test("alimentador", test.reset_index().rename(columns={"sentido": "unidad"}))
    print("alimentador: fracción de subidas en terminal por sentido")
    print(frac_term.unstack().round(2).to_string())


def largo_alimentadores(series: list) -> dict:
    """Largo (km) de cada alimentador: trazado ATU/OSRM del servicio, promedio de sus sentidos."""
    mapeo = json.loads((RAIZ / "data" / "geo" / "mapeo_alimentadores.json").read_text(encoding="utf-8"))["servicio_a_serie"]
    geo = json.loads((RAIZ / "data" / "geo" / "trazados_alimentadores_atu.geojson").read_text(encoding="utf-8"))
    por_serie: dict[str, list[float]] = {}
    for f in geo["features"]:
        serie = mapeo.get(f["properties"]["ruta"])
        c = f["geometry"]["coordinates"]
        if serie:
            por_serie.setdefault(serie, []).append(sum(haversine_km(a, b) for a, b in zip(c, c[1:])))
    media = float(np.mean([np.mean(v) for v in por_serie.values()]))
    return {s: float(np.mean(por_serie[s])) if s in por_serie else media for s in series}


def escribir(sistema: str, cfg: dict, npz: dict) -> None:
    carpeta = RAIZ / "models" / sistema
    (carpeta / "rutas_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(carpeta / "rutas_participaciones.npz", **{k: v.astype(np.float32) for k, v in npz.items()})


def guardar_test(sistema: str, df: pd.DataFrame) -> None:
    salida = RAIZ / "data" / "eval"
    salida.mkdir(parents=True, exist_ok=True)
    df.to_parquet(salida / f"{sistema}_paradas_test.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, type=Path, help="carpeta data_processed/clean del repo DPD")
    a = ap.parse_args()
    corredor(a.datos)
    alimentador(a.datos)


if __name__ == "__main__":
    main()
