"""Pronostico de casos de dengue en series temporales agregadas.

Compara una media movil (baseline) contra suavizado exponencial de Holt-Winters
(modelo principal). Evalua con MAPE seguro y RMSE sobre un periodo de prueba
cronologico y proyecta al menos cuatro periodos futuros.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Construccion de la serie
# ---------------------------------------------------------------------------
def build_series(df: pd.DataFrame, nivel: str = "nacional", clave: str | None = None) -> pd.Series:
    """Serie semanal de casos agregada a un nivel.

    nivel: 'nacional' | 'departamento' | 'distrito'
    clave: nombre del departamento o ubigeo del distrito (segun nivel).

    La serie se reindexa sobre el calendario epidemiologico real del dataset
    (no sobre una rejilla de lunes): la fecha de una semana epidemiologica cae
    en el dia de la semana del 1 de enero de su anio, que cambia cada anio.
    Las semanas sin registro dentro del rango activo se completan con 0.
    """
    d = df.copy()
    d["fecha"] = pd.to_datetime(d["fecha"])
    calendario = pd.DatetimeIndex(sorted(d["fecha"].unique()))

    if nivel == "departamento" and clave is not None:
        d = d[d["departamento"] == clave]
    elif nivel == "distrito" and clave is not None:
        d = d[d["ubigeo"].astype(str) == str(clave)]

    serie = d.groupby("fecha")["casos"].sum().sort_index()
    if serie.empty:
        return serie.astype(float)

    # Rejilla completa dentro del periodo activo del nivel seleccionado
    rango = calendario[(calendario >= serie.index.min()) & (calendario <= serie.index.max())]
    return serie.reindex(rango).fillna(0.0).astype(float)


def _ano_semana(fecha: pd.Timestamp) -> tuple[int, int]:
    """Invierte el calendario epidemiologico: fecha -> (ano, semana).

    Es la inversa exacta de preprocessing._epi_week_to_date, que define
    fecha = 1 de enero + (semana - 1) * 7 dias.
    """
    fecha = pd.Timestamp(fecha)
    return int(fecha.year), int((fecha.dayofyear - 1) // 7 + 1)


def fechas_futuras(ultima: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    """Continua el calendario epidemiologico 'periods' semanas hacia adelante."""
    import datetime

    ano, semana = _ano_semana(ultima)
    fechas = []
    for _ in range(periods):
        semana += 1
        if semana > 52:
            ano, semana = ano + 1, 1
        fechas.append(datetime.date(ano, 1, 1) + datetime.timedelta(days=(semana - 1) * 7))
    return pd.DatetimeIndex(pd.to_datetime(fechas))


# ---------------------------------------------------------------------------
# Metricas robustas
# ---------------------------------------------------------------------------
def safe_mape(y_true, y_pred) -> float:
    """MAPE que excluye los denominadores iguales a cero.

    En series de dengue hay semanas con 0 casos reales; incluirlas dividiria
    entre cero. Se excluyen esos puntos y se informa el MAPE sobre el resto.
    Devuelve NaN si no queda ningun punto con valor real > 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def moving_average_forecast(train: pd.Series, steps: int, window: int | None = None) -> np.ndarray:
    """Baseline: pronostica todos los pasos como la media de las ultimas
    'window' observaciones del entrenamiento."""
    window = window or config.MOVING_AVERAGE_WINDOW
    valor = float(train.tail(window).mean())
    return np.repeat(valor, steps)


def _fit_holt_winters(train: pd.Series):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # El calendario epidemiologico no es una frecuencia regular de pandas
    # (cada anio arranca en un dia distinto). Se ajusta sobre los valores, que
    # sí estan igualmente espaciados en semanas.
    train = pd.Series(np.asarray(train, dtype=float)).reset_index(drop=True)

    usar_estacional = len(train) >= 2 * 52
    if usar_estacional:
        modelo = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=52,
            initialization_method="estimated",
        )
    else:
        modelo = ExponentialSmoothing(
            train, trend="add", seasonal=None, initialization_method="estimated"
        )
    return modelo.fit(), usar_estacional


def holt_winters_forecast(train: pd.Series, steps: int):
    """Suavizado exponencial. Devuelve (pronostico, intervalo aproximado).

    El intervalo se aproxima con +-1.96 * desviacion de los residuos (Holt-Winters
    de statsmodels no expone intervalos analiticos directos)."""
    fit, _ = _fit_holt_winters(train)
    pred = np.asarray(fit.forecast(steps), dtype=float)
    resid_std = float(np.nanstd(fit.resid)) if hasattr(fit, "resid") else 0.0
    margen = 1.96 * resid_std
    lower = np.clip(pred - margen, 0, None)
    upper = pred + margen
    return pred, (lower, upper)


# ---------------------------------------------------------------------------
# Evaluacion y pronostico futuro
# ---------------------------------------------------------------------------
def evaluate_models(serie: pd.Series, test_periods: int | None = None) -> dict:
    """Separa cronologicamente 'test_periods' al final y evalua ambos modelos."""
    test_periods = test_periods or config.FORECAST_EVAL_PERIODS
    serie = serie.astype(float)
    train, test = serie.iloc[:-test_periods], serie.iloc[-test_periods:]

    ma_pred = moving_average_forecast(train, len(test))
    hw_pred, _ = holt_winters_forecast(train, len(test))

    resultados = {
        "media_movil": {
            "mape": safe_mape(test.values, ma_pred),
            "rmse": rmse(test.values, ma_pred),
        },
        "holt_winters": {
            "mape": safe_mape(test.values, hw_pred),
            "rmse": rmse(test.values, hw_pred),
        },
    }
    # Modelo elegido: menor RMSE (robusto ante ceros que afectan al MAPE)
    mejor = min(resultados, key=lambda k: resultados[k]["rmse"])
    return {
        "resultados": resultados,
        "mejor_modelo": mejor,
        "train": train,
        "test": test,
        "pred_test": {"media_movil": ma_pred, "holt_winters": hw_pred},
    }


def tabla_robustez(serie: pd.Series, ventanas: list[int] | None = None) -> pd.DataFrame:
    """Compara ambos modelos en varias ventanas de evaluacion.

    Que modelo gana depende de cuantas semanas se reserven para prueba: en una
    ventana corta la media movil (una constante) puede ganar por construccion.
    Mostrar la tabla completa evita elegir la ventana que favorece al resultado
    deseado.
    """
    ventanas = ventanas or config.FORECAST_EVAL_VENTANAS
    filas = []
    for v in ventanas:
        if len(serie) <= v + 2 * config.MOVING_AVERAGE_WINDOW:
            continue
        ev = evaluate_models(serie, v)
        r = ev["resultados"]
        filas.append({
            "ventana de prueba (semanas)": v,
            "MAPE media movil": round(r["media_movil"]["mape"], 1),
            "RMSE media movil": round(r["media_movil"]["rmse"], 1),
            "MAPE Holt-Winters": round(r["holt_winters"]["mape"], 1),
            "RMSE Holt-Winters": round(r["holt_winters"]["rmse"], 1),
            "elegido (menor RMSE)": ev["mejor_modelo"],
        })
    return pd.DataFrame(filas)


def forecast_future(serie: pd.Series, periods: int | None = None) -> pd.DataFrame:
    """Ajusta con toda la serie y proyecta 'periods' semanas futuras con intervalo."""
    periods = periods or config.FORECAST_PERIODS
    serie = serie.astype(float)
    pred, (lower, upper) = holt_winters_forecast(serie, periods)

    fechas = fechas_futuras(serie.index[-1], periods)
    return pd.DataFrame(
        {"fecha": fechas, "pronostico": pred, "inferior": lower, "superior": upper}
    )
