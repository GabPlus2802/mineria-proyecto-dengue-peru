"""Pruebas de modelos: clustering, clasificacion y pronostico."""

import numpy as np
import pandas as pd
import pytest

import config
from src import clustering, forecasting, modeling


# ---------------------------------------------------------------------------
# Datos sinteticos con estructura minima de un dataset maestro
# ---------------------------------------------------------------------------
@pytest.fixture
def maestro_sintetico():
    rng = np.random.RandomState(0)
    filas = []
    for ubigeo, dep, base in [("160101", "LORETO", 10), ("200101", "PIURA", 3),
                              ("150101", "LIMA", 1)]:
        for i in range(160):
            ano = 2019 + i // 52
            semana = (i % 52) + 1
            casos = max(0, int(base + 5 * np.sin(i / 8) + rng.randint(-2, 3)))
            filas.append([ubigeo, dep, "DIST", ano, semana, i, casos])
    df = pd.DataFrame(filas, columns=[
        "ubigeo", "departamento", "distrito", "ano", "semana", "week_id", "casos"])
    df["fecha"] = pd.to_datetime("2019-01-01") + pd.to_timedelta(df["week_id"] * 7, "D")

    # features requeridas por el modelado
    g = df.groupby("ubigeo")["casos"]
    df["casos_lag_1"] = g.shift(1)
    df["casos_lag_2"] = g.shift(2)
    df["casos_lag_4"] = g.shift(4)
    prev = g.shift(1)
    df["promedio_movil_4"] = prev.rolling(4).mean()
    df["promedio_movil_8"] = prev.rolling(8).mean()
    df["desviacion_movil_4"] = prev.rolling(4).std()
    df["crecimiento_semanal"] = (df["casos"] - df["casos_lag_1"]) / (df["casos_lag_1"] + 1)
    df["semana_sen"] = np.sin(2 * np.pi * df["semana"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana"] / 52)

    # split y objetivo
    df["split"] = "train"
    df.loc[df["ano"] == 2021, "split"] = "val"
    df.loc[df["ano"] == 2022, "split"] = "test"
    umbral = df[df["split"] == "train"].groupby("ubigeo")["casos"].quantile(0.75)
    df = df.merge(umbral.rename("umbral_incidencia"), on="ubigeo")
    df["casos_siguiente"] = df.groupby("ubigeo")["casos"].shift(-1)
    df["alta_incidencia_siguiente_semana"] = (
        df["casos_siguiente"] > df["umbral_incidencia"]).astype("Int64")
    df.loc[df["casos_siguiente"].isna(), "alta_incidencia_siguiente_semana"] = pd.NA
    return df


# ---------------------------------------------------------------------------
# Clasificacion
# ---------------------------------------------------------------------------
def test_modelos_se_entrenan(maestro_sintetico):
    data = modeling.get_modeling_frame(maestro_sintetico)
    models, info = modeling.train_models(data)
    assert set(models) == {"random_forest", "xgboost"}
    assert "balanceo_aplicado" in info


def test_probabilidades_entre_0_y_1(maestro_sintetico):
    data = modeling.get_modeling_frame(maestro_sintetico)
    models, _ = modeling.train_models(data)
    for modelo in models.values():
        proba = modelo.predict_proba(data["X_test"])[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()


def test_metricas_se_calculan(maestro_sintetico):
    data = modeling.get_modeling_frame(maestro_sintetico)
    models, _ = modeling.train_models(data)
    tabla = modeling.metrics_table(models, data["X_test"], data["y_test"])
    for col in ["accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]:
        assert col in tabla.columns


def test_guardar_y_cargar_modelo(maestro_sintetico, tmp_path):
    import joblib
    data = modeling.get_modeling_frame(maestro_sintetico)
    models, _ = modeling.train_models(data)
    ruta = tmp_path / "rf.joblib"
    joblib.dump(models["random_forest"], ruta)
    cargado = joblib.load(ruta)
    assert cargado.predict_proba(data["X_test"]).shape[1] == 2


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_kmeans_respeta_numero_clusters(maestro_sintetico):
    perfil = clustering.district_profile(maestro_sintetico)
    res = clustering.run_kmeans(perfil, k=2)
    assert res["k"] == 2
    assert perfil.shape[0] == maestro_sintetico["ubigeo"].nunique()
    assert res["perfil"]["cluster"].nunique() == 2


# ---------------------------------------------------------------------------
# Pronostico
# ---------------------------------------------------------------------------
def test_safe_mape_excluye_ceros():
    y_true = np.array([0, 10, 20])
    y_pred = np.array([5, 11, 18])
    # solo se consideran los indices 1 y 2
    esperado = np.mean([abs(10 - 11) / 10, abs(20 - 18) / 20]) * 100
    assert forecasting.safe_mape(y_true, y_pred) == pytest.approx(esperado)


def test_safe_mape_todo_cero_es_nan():
    assert np.isnan(forecasting.safe_mape([0, 0], [1, 2]))


def test_pronostico_genera_periodos(maestro_sintetico):
    serie = forecasting.build_series(maestro_sintetico, nivel="nacional")
    futuro = forecasting.forecast_future(serie, periods=config.FORECAST_PERIODS)
    assert len(futuro) == config.FORECAST_PERIODS
    assert (futuro["inferior"] <= futuro["pronostico"] + 1e-6).all()
