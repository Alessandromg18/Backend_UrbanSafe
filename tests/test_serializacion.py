"""Verificación de la serialización v1.1.0: integridad (sha256) y recarga reproducible."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.inferencia import ModeloSistema, sha256

RAIZ = Path(__file__).resolve().parents[1]
MODELS = RAIZ / "models"
MAN = json.loads((MODELS / "manifest.json").read_text(encoding="utf-8"))
SISTEMAS = ("metro_l1", "troncal", "corredor", "alimentador")


def test_version_coherente():
    assert MAN["version"] == "1.1.0"
    for s in SISTEMAS:
        card = json.loads((MODELS / s / "model_card.json").read_text(encoding="utf-8"))
        assert card["version"] == "1.1.0" and card["aforo_v2"]["version"] == "1.1.0"


def test_hashes_de_datos_de_entrada():
    for rel, meta in MAN["datos"].items():
        assert sha256(RAIZ / rel) == meta["sha256"], rel


@pytest.mark.parametrize("sistema", SISTEMAS)
def test_recarga_verifica_hash_y_reproduce_referencia(sistema):
    from services import model_service as MS
    m = ModeloSistema.cargar(sistema, verificar_hash=True)       # falla si un artefacto cambió
    df, par = m.pronosticar(MS.contexto(sistema), MAN["dia_referencia_verificacion"], modo="ligero",
                            con_paradas=True)
    ref = pd.read_csv(MODELS / sistema / "referencia_v1_1_0.csv")
    assert list(df.columns) == list(ref.columns) and len(df) == len(ref)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(ref[c]):
            np.testing.assert_allclose(df[c].to_numpy(float), ref[c].to_numpy(float),
                                       rtol=1e-6, atol=1e-6, equal_nan=True, err_msg=c)
        elif c != "fecha_hora":
            assert (df[c].astype(str).fillna("") == ref[c].astype(str).fillna("")).all(), c
    if par is not None:
        rp = pd.read_parquet(MODELS / sistema / "referencia_v1_1_0_paradas.parquet")
        np.testing.assert_allclose(par["carga"].to_numpy(float), rp["carga"].to_numpy(float), rtol=1e-6, atol=1e-6)
        assert (par["aforo"].astype(str).to_numpy() == rp["aforo"].astype(str).to_numpy()).all()
