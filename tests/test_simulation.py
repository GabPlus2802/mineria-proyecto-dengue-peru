"""Pruebas de la extension simulada y del calendario del pronostico.

Lo critico que se verifica aqui:
  - las filas generadas quedan marcadas y FUERA de train/val/test;
  - la extension no toca ni una sola fila real;
  - la simulacion es reproducible;
  - la serie del pronostico respeta el calendario epidemiologico (el bug del
    asfreq("W-MON") convertia en ceros todo anio que no empezara en lunes).
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

import config
from src import forecasting, preprocessing, simulation


# ---------------------------------------------------------------------------
# Maestro sintetico minimo (no depende del CSV de 70 MB)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def maestro():
    filas = []
    for ubigeo, dep in [("010101", "AMAZONAS"), ("020202", "PIURA")]:
        for ano in range(2019, 2025):
            for semana in range(1, 53):
                # Estacionalidad marcada: pico alrededor de la semana 13
                base = 20 * np.exp(-((semana - 13) ** 2) / 120)
                filas.append({
                    "ubigeo": ubigeo, "departamento": dep, "provincia": "P",
                    "distrito": "D" + ubigeo, "ano": ano, "semana": semana,
                    "casos": int(round(base)), "edad_promedio": 30.0,
                    "prop_con_signos": 0.1, "prop_grave": 0.0, "prop_femenino": 0.5,
                })
    df = pd.DataFrame(filas)
    cal = (df[["ano", "semana"]].drop_duplicates()
           .sort_values(["ano", "semana"]).reset_index(drop=True))
    cal["week_id"] = np.arange(len(cal))
    cal["fecha"] = pd.to_datetime(
        [preprocessing._epi_week_to_date(a, s) for a, s in zip(cal["ano"], cal["semana"])])
    df = df.merge(cal, on=["ano", "semana"])
    df["origen"] = "real"
    df = preprocessing.add_features(df)
    return preprocessing.add_target_and_split(df)


@pytest.fixture(scope="module")
def extendido(maestro):
    return simulation.extender_master(maestro, verbose=False)


# ---------------------------------------------------------------------------
# Integridad del dato real
# ---------------------------------------------------------------------------
def test_la_extension_no_altera_ninguna_fila_real(maestro, extendido):
    """Los casos observados deben sobrevivir intactos a la extension."""
    orig = maestro.set_index(["ubigeo", "week_id"])["casos"].sort_index()
    real = extendido[extendido["origen"] == "real"]
    nuevo = real.set_index(["ubigeo", "week_id"])["casos"].sort_index()
    pd.testing.assert_series_equal(orig, nuevo, check_names=False)


def test_las_filas_simuladas_quedan_fuera_del_modelado(extendido):
    """Ninguna fila generada puede caer en train, val ni test."""
    sim = extendido[extendido["origen"] == "simulado"]
    assert len(sim) > 0
    assert set(sim["split"].unique()) == {"simulado"}
    assert not extendido[extendido["split"].isin(["train", "val", "test"])]["origen"].eq(
        "simulado").any()


def test_el_conjunto_de_prueba_sigue_siendo_el_ultimo_anio_real(maestro, extendido):
    ultimo_real = int(maestro["ano"].max())
    test = extendido[extendido["split"] == "test"]
    assert set(test["ano"].unique()) == {ultimo_real}


# ---------------------------------------------------------------------------
# Propiedades de la simulacion
# ---------------------------------------------------------------------------
def test_la_extension_llega_al_horizonte_configurado(extendido):
    sim = extendido[extendido["origen"] == "simulado"]
    assert int(sim["ano"].max()) == config.SIM_END_ANO
    assert int(sim.loc[sim["ano"] == config.SIM_END_ANO, "semana"].max()) == config.SIM_END_SEMANA


def test_los_casos_simulados_son_enteros_no_negativos(extendido):
    sim = extendido[extendido["origen"] == "simulado"]
    assert (sim["casos"] >= 0).all()
    assert (sim["casos"] == sim["casos"].round()).all()


def test_la_simulacion_es_reproducible(maestro):
    a = simulation.extender_master(maestro, verbose=False)
    b = simulation.extender_master(maestro, verbose=False)
    pd.testing.assert_series_equal(a["casos"], b["casos"])


def test_la_simulacion_conserva_la_estacionalidad(extendido):
    """El pico simulado debe caer cerca del pico historico, no en cualquier semana."""
    sim = extendido[extendido["origen"] == "simulado"]
    pico = int(sim.groupby("semana")["casos"].mean().idxmax())
    assert abs(pico - 13) <= 6, f"pico simulado en la semana {pico}, se esperaba cerca de 13"


def test_no_se_reactivan_distritos_inactivos(maestro):
    """Un distrito que dejo de reportar hace anios no debe recibir extension."""
    viejo = maestro[maestro["ubigeo"] == "010101"]
    viejo = viejo[viejo["ano"] <= 2020]
    parcial = pd.concat([viejo, maestro[maestro["ubigeo"] == "020202"]], ignore_index=True)
    ext = simulation.extender_master(parcial, verbose=False)
    sim = ext[ext["origen"] == "simulado"]
    assert "010101" not in set(sim["ubigeo"].unique())


# ---------------------------------------------------------------------------
# Calendario epidemiologico del pronostico
# ---------------------------------------------------------------------------
def test_fecha_y_ano_semana_son_inversas():
    for ano in range(2020, 2028):
        for semana in [1, 9, 22, 40, 52]:
            fecha = preprocessing._epi_week_to_date(ano, semana)
            assert forecasting._ano_semana(pd.Timestamp(fecha)) == (ano, semana)


def test_la_serie_no_pierde_semanas_en_anios_que_no_empiezan_en_lunes(extendido):
    """Regresion: asfreq('W-MON') anulaba todo anio cuyo 1 de enero no fuera lunes.

    2026 empieza en jueves, asi que sus semanas quedaban en NaN -> 0.
    """
    serie = forecasting.build_series(extendido)
    ultimas = serie[serie.index >= pd.Timestamp(f"{config.SIM_END_ANO}-01-01")]
    assert len(ultimas) > 0
    assert ultimas.sum() > 0, "el ultimo anio quedo en ceros: el calendario se rompio"
    assert not serie.isna().any()


def test_las_fechas_futuras_continuan_el_calendario():
    ultima = pd.Timestamp(preprocessing._epi_week_to_date(2026, 51))
    fechas = forecasting.fechas_futuras(ultima, 4)
    assert len(fechas) == 4
    assert list(fechas) == [
        pd.Timestamp(datetime.date(2026, 12, 24)),   # semana 52
        pd.Timestamp(datetime.date(2027, 1, 1)),     # semana 1 del anio siguiente
        pd.Timestamp(datetime.date(2027, 1, 8)),
        pd.Timestamp(datetime.date(2027, 1, 15)),
    ]
    assert fechas.is_monotonic_increasing


def test_el_pronostico_arranca_despues_del_ultimo_dato(extendido):
    serie = forecasting.build_series(extendido)
    futuro = forecasting.forecast_future(serie, periods=6)
    assert futuro["fecha"].min() > serie.index.max()
    assert len(futuro) == 6
    assert (futuro["inferior"] <= futuro["pronostico"]).all()
    assert (futuro["pronostico"] <= futuro["superior"]).all()
    assert (futuro["inferior"] >= 0).all()
