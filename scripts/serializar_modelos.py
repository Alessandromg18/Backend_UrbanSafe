"""Serializa una versión de los modelos (aforo v2 con matriz O-D) y su verificación de recarga.

Uso (desde la raíz del backend, después de construir las capas y correr las evaluaciones)::

    python scripts/serializar_modelos.py 1.1.1 "corrección de coordenada de Bulevar"

- Sube la versión de las capas (``linea_config.json`` / ``rutas_config.json``) y de los
  ``model_card.json`` y agrega a cada card la sección ``aforo_v2`` (método, archivos y
  métricas de test).
- Genera salidas de referencia (día 2025-12-15, LightGBM) en ``models/<sistema>/referencia_salida_v2*``
  (borra las de versiones anteriores).
- Escribe ``models/manifest.json`` con sha256 y tamaño de cada artefacto de ``models/`` y de los
  datos de entrada de la inferencia (``data/context``, ``data/geo/mapeo_alimentadores.json``).

``ModeloSistema.cargar(verificar_hash=True)`` compara contra este manifiesto y
``tests/test_serializacion.py`` verifica que la recarga reproduzca las salidas de referencia.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ / "model")]

from inferencia import ModeloSistema, sha256  # noqa: E402

VERSION = sys.argv[1] if len(sys.argv) > 1 else "1.1.0"
CAMBIOS = sys.argv[2] if len(sys.argv) > 2 else ""
SISTEMAS = ("metro_l1", "troncal", "corredor", "alimentador")
DIA_REFERENCIA = "2025-12-15"
MODELS = RAIZ / "models"


def leer(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def escribir(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1 if p.name == "rutas_config.json" else 2),
                 encoding="utf-8")


def actualizar_cards() -> None:
    ev = leer(MODELS / "evaluacion_ocupacion.json")
    for s in SISTEMAS:
        carpeta = MODELS / s
        capa = "linea_config.json" if (carpeta / "linea_config.json").exists() else "rutas_config.json"
        cfg = leer(carpeta / capa)
        cfg["version"] = VERSION
        escribir(carpeta / capa, cfg)
        card = leer(carpeta / "model_card.json")
        card["version"] = VERSION
        e = ev[s]
        nivel = "nivel_paradero" if "nivel_paradero" in e else "nivel_serie"
        card["aforo_v2"] = {
            "version": VERSION,
            "metodo": cfg.get("metodo") or cfg["od"]["metodo"],
            "capa": capa,
            "artefactos": [capa] + (["rutas_participaciones.npz"] if capa == "rutas_config.json" else []),
            "modelo_demanda_produccion": ("lightgbm por ruta × sentido (demanda_sentido.json, lightgbm_sentido.txt)"
                                          if (carpeta / "demanda_sentido.json").exists() else "lightgbm (modo ligero)"),
            "salida": "ocupacion_{asc,desc}, carga_{asc,desc}, aforo_{asc,desc}, eta_{asc,desc}, "
                      "sentido_critico; crowding_index y aforo = sentido más cargado (compatibles con v1)",
            "evaluacion_test": {"dias": ev["dias_test"], "nivel": nivel,
                                "protocolo": "fuera de muestra: LightGBM reentrenado solo con días previos, refit cada 7 días",
                                **e[nivel],
                                "RNF_01": e["RNF_01"]},
            "comparacion_proxy_v1": {k: e["comparacion_proxy_v1"][k] for k in
                                     ("acuerdo_clase", "saturado_en_punta", "saturado_en_valle", "distribucion")},
            "documentacion": "docs/MODELO_OCUPACION.md",
        }
        escribir(carpeta / "model_card.json", card)


def referencias() -> None:
    from services import model_service as MS
    for s in SISTEMAS:
        m = ModeloSistema.cargar(s, verificar_hash=False)
        df, par = m.pronosticar(MS.contexto(s), DIA_REFERENCIA, modo="ligero", con_paradas=True)
        for viejo in (MODELS / s).glob("referencia_v1_1_0*"):
            viejo.unlink()
        df.to_csv(MODELS / s / "referencia_salida_v2.csv", index=False)
        if par is not None:
            par.to_parquet(MODELS / s / "referencia_salida_v2_paradas.parquet", index=False)


def manifiesto() -> None:
    def entrada(p: Path) -> dict:
        return {"sha256": sha256(p), "bytes": p.stat().st_size}

    artefactos = {}
    for s in SISTEMAS:
        for p in sorted((MODELS / s).iterdir()):
            if p.is_file():
                artefactos[f"{s}/{p.name}"] = entrada(p)
    datos = {str(p.relative_to(RAIZ)).replace("\\", "/"): entrada(p)
             for p in sorted((RAIZ / "data" / "context").glob("*_2025.parquet"))}
    datos["data/geo/mapeo_alimentadores.json"] = entrada(RAIZ / "data" / "geo" / "mapeo_alimentadores.json")
    try:
        base = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        base = None
    import lightgbm
    import numpy
    import sklearn
    man = {
        "version": VERSION,
        "fecha": dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(timespec="seconds"),
        "commit_base": base,
        "cambios": CAMBIOS or ("aforo v2: ocupación a bordo con matriz O-D en los 4 sistemas; LightGBM "
                               "en producción; día homólogo 2025; orden_upstream corregido (bug de nombres)"),
        "historial": historial(),
        "dia_referencia_verificacion": DIA_REFERENCIA,
        "entorno": {"python": sys.version.split()[0], "lightgbm": lightgbm.__version__,
                    "numpy": numpy.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "artefactos": artefactos,
        "datos": datos,
    }
    (MODELS / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest.json: {len(artefactos)} artefactos + {len(datos)} archivos de datos")


def historial() -> list:
    previo = MODELS / "manifest.json"
    if not previo.exists():
        return []
    m = leer(previo)
    h = m.get("historial", [])
    if m.get("version") != VERSION:
        h = h + [{"version": m.get("version"), "fecha": m.get("fecha"), "cambios": m.get("cambios")}]
    return h


def main() -> None:
    actualizar_cards()
    referencias()
    manifiesto()


if __name__ == "__main__":
    main()
