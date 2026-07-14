"""Pruebas del preprocesamiento (usan un dataset sintetico pequeno, no el real)."""

import numpy as np
import pandas as pd
import pytest

import config
from src import preprocessing as pp


@pytest.fixture
def raw_sintetico():
    """Casos individuales sinteticos: 2 distritos, varios anios."""
    filas = []
    for ano in range(2018, 2024):
        for semana in range(1, 53):
            # distrito A: casos crecientes; distrito B: pocos casos
            for _ in range((semana % 5) + 1):
                filas.append(["LORETO", "MAYNAS", "IQUITOS", "DENGUE SIN SIGNOS DE ALARMA",
                              ano, semana, "160101", "25", "A", "M"])
            for _ in range(semana % 2):
                filas.append(["PIURA", "PIURA", "PIURA", "DENGUE CON SIGNOS DE ALARMA",
                              ano, semana, "200101", "30", "A", "F"])
    df = pd.DataFrame(filas, columns=[
        "departamento", "provincia", "distrito", "enfermedad", "ano", "semana",
        "ubigeo", "edad", "tipo_edad", "sexo"])
    return df


def test_no_modifica_dataset_original(raw_sintetico):
    original = raw_sintetico.copy(deep=True)
    _ = pp.clean_raw(raw_sintetico)
    pd.testing.assert_frame_equal(raw_sintetico, original)


def test_no_elimina_casos_duplicados(raw_sintetico):
    # Cada fila es un caso; la limpieza no debe reducir el numero de casos validos
    limpio = pp.clean_raw(raw_sintetico)
    assert len(limpio) == len(raw_sintetico)


def test_edad_conversion_por_tipo():
    df = pd.DataFrame({
        "edad": ["24", "12", "365"],
        "tipo_edad": ["A", "M", "D"],
    })
    anios = pp._edad_a_anios(df["edad"], df["tipo_edad"])
    assert anios.iloc[0] == pytest.approx(24)
    assert anios.iloc[1] == pytest.approx(1.0)      # 12 meses = 1 anio
    assert anios.iloc[2] == pytest.approx(1.0)      # 365 dias = 1 anio


def test_edad_imposible_a_nan():
    anios = pp._edad_a_anios(pd.Series(["200", "-5", "40"]), pd.Series(["A", "A", "A"]))
    assert pd.isna(anios.iloc[0]) and pd.isna(anios.iloc[1])
    assert anios.iloc[2] == 40


def _maestro(raw):
    limpio = pp.clean_raw(raw)
    cal = pp.build_epi_calendar(limpio)
    agg = pp.aggregate_district_week(limpio, cal)
    completo = pp.fill_zero_weeks(agg, cal)
    completo = pp.filtrar_distritos_con_historial(completo, min_semanas=10)
    conf = pp.add_features(completo)
    return pp.add_target_and_split(conf)


def test_fecha_valida_y_ordenada(raw_sintetico):
    m = _maestro(raw_sintetico)
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(m["fecha"]))
    for _, sub in m.groupby("ubigeo"):
        fechas = pd.to_datetime(sub["fecha"]).values
        assert (np.diff(fechas.astype("int64")) > 0).all()  # estrictamente creciente


def test_clave_distrito_fecha_unica(raw_sintetico):
    m = _maestro(raw_sintetico)
    assert m.duplicated(["ubigeo", "fecha"]).sum() == 0


def test_lags_no_usan_futuro(raw_sintetico):
    m = _maestro(raw_sintetico).sort_values(["ubigeo", "week_id"])
    for _, sub in m.groupby("ubigeo"):
        sub = sub.reset_index(drop=True)
        # casos_lag_1 en la fila t debe igualar casos en la fila t-1
        esperado = sub["casos"].shift(1)
        assert (sub["casos_lag_1"].fillna(-999) == esperado.fillna(-999)).all()


def test_objetivo_no_esta_en_predictores():
    assert pp.TARGET not in pp.FEATURE_NUMERIC
    assert pp.TARGET not in pp.FEATURE_CATEGORICAL
    assert "casos_siguiente" not in pp.FEATURE_NUMERIC


def test_objetivo_binario(raw_sintetico):
    m = _maestro(raw_sintetico)
    valores = m[pp.TARGET].dropna().unique()
    assert set(valores).issubset({0, 1})


def test_media_movil_excluye_semana_actual(raw_sintetico):
    m = _maestro(raw_sintetico).sort_values(["ubigeo", "week_id"])
    for _, sub in m.groupby("ubigeo"):
        sub = sub.reset_index(drop=True)
        # promedio_movil_4 en t = media de casos en [t-4, t-1] (shift(1) aplicado)
        esperado = sub["casos"].shift(1).rolling(config.MOVING_AVERAGE_WINDOW).mean()
        comp = np.isclose(sub["promedio_movil_4"], esperado, equal_nan=True)
        assert comp.all()
        break
