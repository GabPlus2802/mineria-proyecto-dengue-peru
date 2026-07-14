"""Preprocesamiento del dataset de vigilancia de dengue (MINSA Peru 2000-2024).

El dataset original es una lista de CASOS individuales notificados. Este modulo
lo transforma en la unidad de analisis distrito-semana, agrega ingenieria de
caracteristicas sin fuga de informacion futura, define la variable objetivo y
aplica una division temporal.

Funciones reutilizables (importadas por train.py, notebooks y tests).
"""

from __future__ import annotations

import datetime
import unicodedata

import numpy as np
import pandas as pd

import config

# Columnas identificadoras y de features derivadas (para el modelado)
FEATURE_NUMERIC = [
    "casos",
    "casos_lag_1",
    "casos_lag_2",
    "casos_lag_4",
    "promedio_movil_4",
    "promedio_movil_8",
    "desviacion_movil_4",
    "crecimiento_semanal",
    "semana_sen",
    "semana_cos",
]
FEATURE_CATEGORICAL = ["departamento"]
TARGET = "alta_incidencia_siguiente_semana"
CLAVE = ["ubigeo", "fecha"]


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------
def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar_texto(serie: pd.Series) -> pd.Series:
    """Mayusculas, sin acentos, sin espacios sobrantes."""
    return (
        serie.astype(str)
        .str.strip()
        .str.upper()
        .map(_strip_accents)
        .str.replace(r"\s+", " ", regex=True)
    )


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------
def load_raw(path=None) -> pd.DataFrame:
    """Carga el CSV original SIN modificarlo (solo lectura)."""
    path = path or config.RAW_DATASET
    return pd.read_csv(
        path, sep=config.RAW_SEP, encoding=config.RAW_ENCODING, dtype=str
    )


def _edad_a_anios(edad: pd.Series, tipo: pd.Series) -> pd.Series:
    """Convierte edad a anios usando tipo_edad (A=anios, M=meses, D=dias)."""
    edad_num = pd.to_numeric(edad, errors="coerce").astype(float)
    tipo = tipo.astype(str).str.upper().str.strip()
    anios = edad_num.copy()
    anios = anios.mask(tipo == "M", edad_num / 12.0)
    anios = anios.mask(tipo == "D", edad_num / 365.0)
    # Edades imposibles -> NaN (el dataset tiene algunos valores basura)
    anios = anios.mask((anios < 0) | (anios > 120))
    return anios


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza a nivel de caso. NO elimina filas duplicadas: cada fila es un
    caso individual distinto (mismos atributos = pacientes diferentes)."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Tipos numericos basicos
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["semana"] = pd.to_numeric(df["semana"], errors="coerce").astype("Int64")

    # Geografia normalizada
    for col in ["departamento", "provincia", "distrito"]:
        df[col] = normalizar_texto(df[col])
    df["ubigeo"] = df["ubigeo"].astype(str).str.strip()

    # Edad en anios y severidad
    df["edad_anios"] = _edad_a_anios(df["edad"], df["tipo_edad"])
    df["enfermedad"] = normalizar_texto(df["enfermedad"])
    df["con_signos_o_grave"] = df["enfermedad"].isin(
        ["DENGUE CON SIGNOS DE ALARMA", "DENGUE GRAVE"]
    ).astype(int)
    df["es_grave"] = (df["enfermedad"] == "DENGUE GRAVE").astype(int)
    df["sexo"] = df["sexo"].astype(str).str.upper().str.strip()
    df["es_femenino"] = (df["sexo"] == "F").astype(int)

    # Validaciones minimas de semana epidemiologica (1..53)
    df = df[df["semana"].between(1, 53) & df["ano"].between(2000, 2100)]
    return df


# ---------------------------------------------------------------------------
# Calendario epidemiologico y agregacion distrito-semana
# ---------------------------------------------------------------------------
def _epi_week_to_date(year: int, week: int) -> datetime.date:
    """Fecha aproximada de la semana epidemiologica: 1 de enero + (semana-1)*7.

    Es una aproximacion (no usa el ano ISO) pero garantiza que cada par
    (ano, semana) produzca una fecha unica y estrictamente creciente, evitando
    colisiones en los limites de anio. Sirve para ordenar y graficar.
    """
    return datetime.date(int(year), 1, 1) + datetime.timedelta(days=(int(week) - 1) * 7)


def build_epi_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Grid global de semanas epidemiologicas (ano, semana) con orden continuo.

    Incluye todas las semanas 1..W de cada anio, donde W = maxima semana
    observada ese anio. Asi las semanas internas ausentes pueden completarse
    con cero casos.
    """
    max_week = df.groupby("ano")["semana"].max()
    filas = []
    for ano, wmax in max_week.items():
        for semana in range(1, int(wmax) + 1):
            filas.append((int(ano), semana))
    cal = pd.DataFrame(filas, columns=["ano", "semana"])
    cal = cal.sort_values(["ano", "semana"]).reset_index(drop=True)
    cal["week_id"] = np.arange(len(cal))
    cal["fecha"] = [
        _epi_week_to_date(a, s) for a, s in zip(cal["ano"], cal["semana"])
    ]
    cal["fecha"] = pd.to_datetime(cal["fecha"])
    return cal


def aggregate_district_week(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Agrega los casos individuales a nivel ubigeo + semana epidemiologica."""
    # Nombre geografico dominante por ubigeo (evita conflictos de escritura)
    geo = (
        df.groupby("ubigeo")[["departamento", "provincia", "distrito"]]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )

    agg = (
        df.groupby(["ubigeo", "ano", "semana"])
        .agg(
            casos=("enfermedad", "size"),
            edad_promedio=("edad_anios", "mean"),
            prop_con_signos=("con_signos_o_grave", "mean"),
            prop_grave=("es_grave", "mean"),
            prop_femenino=("es_femenino", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(calendar, on=["ano", "semana"], how="left")
    agg = agg.merge(geo, on="ubigeo", how="left")
    return agg


def fill_zero_weeks(agg: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Completa con 0 casos las semanas sin registro dentro del periodo activo
    de cada distrito (desde su primera hasta su ultima semana observada).

    Justificacion: el dataset es una lista de notificaciones, por lo que una
    combinacion distrito-semana ausente dentro del periodo activo puede
    interpretarse como 0 casos observados. No se extrapola mas alla del ultimo
    registro de cada distrito.
    """
    geo_cols = ["departamento", "provincia", "distrito"]
    geo = agg.drop_duplicates("ubigeo").set_index("ubigeo")[geo_cols]

    partes = []
    cal_idx = calendar.set_index("week_id")
    for ubigeo, sub in agg.groupby("ubigeo"):
        wmin, wmax = sub["week_id"].min(), sub["week_id"].max()
        rango = calendar[(calendar["week_id"] >= wmin) & (calendar["week_id"] <= wmax)]
        full = rango.merge(
            sub.drop(columns=geo_cols + ["fecha"]).assign(ubigeo=ubigeo),
            on=["ano", "semana", "week_id"],
            how="left",
        )
        full["ubigeo"] = ubigeo
        full["casos"] = full["casos"].fillna(0).astype(int)
        partes.append(full)

    out = pd.concat(partes, ignore_index=True)
    out = out.merge(geo, left_on="ubigeo", right_index=True, how="left")
    out = out.sort_values(["ubigeo", "week_id"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Ingenieria de caracteristicas (sin informacion futura)
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea lags, medias moviles y estacionalidad por distrito.

    Todas las medias/desviaciones moviles usan shift(1): solo datos anteriores
    a la semana actual, evitando fuga de informacion.
    """
    df = df.sort_values(["ubigeo", "week_id"]).copy()
    g = df.groupby("ubigeo")["casos"]

    df["casos_lag_1"] = g.shift(1)
    df["casos_lag_2"] = g.shift(2)
    df["casos_lag_4"] = g.shift(4)

    prev = g.shift(1)  # serie desplazada: excluye la semana actual
    df["promedio_movil_4"] = prev.rolling(config.MOVING_AVERAGE_WINDOW).mean()
    df["promedio_movil_8"] = prev.rolling(2 * config.MOVING_AVERAGE_WINDOW).mean()
    df["desviacion_movil_4"] = prev.rolling(config.MOVING_AVERAGE_WINDOW).std()

    df["crecimiento_semanal"] = (df["casos"] - df["casos_lag_1"]) / (
        df["casos_lag_1"] + 1.0
    )

    df["semana_sen"] = np.sin(2 * np.pi * df["semana"].astype(float) / 52.0)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana"].astype(float) / 52.0)
    return df


# ---------------------------------------------------------------------------
# Variable objetivo y division temporal
# ---------------------------------------------------------------------------
def temporal_split_years(df: pd.DataFrame):
    """Devuelve (train_years, val_years, test_years) segun config."""
    years = sorted(df["ano"].dropna().unique().tolist())
    test_years = years[-config.TEST_YEARS:]
    val_years = years[-(config.TEST_YEARS + config.VALIDATION_YEARS): -config.TEST_YEARS]
    train_years = years[: -(config.TEST_YEARS + config.VALIDATION_YEARS)]
    return train_years, val_years, test_years


def add_target_and_split(df: pd.DataFrame, percentile=None) -> pd.DataFrame:
    """Define la variable objetivo y la columna 'split'.

    alta_incidencia_siguiente_semana = 1 si los casos de la semana siguiente
    superan el percentil historico del distrito. El umbral se calcula SOLO con
    el periodo de entrenamiento (sin datos de validacion/prueba) para evitar
    fuga de informacion.
    """
    percentile = percentile if percentile is not None else config.TARGET_PERCENTILE
    df = df.sort_values(["ubigeo", "week_id"]).copy()

    train_years, val_years, test_years = temporal_split_years(df)

    df["split"] = "train"
    df.loc[df["ano"].isin(val_years), "split"] = "val"
    df.loc[df["ano"].isin(test_years), "split"] = "test"

    # Umbral por distrito usando SOLO entrenamiento
    train_mask = df["split"] == "train"
    umbral = (
        df[train_mask]
        .groupby("ubigeo")["casos"]
        .quantile(percentile / 100.0)
        .rename("umbral_incidencia")
    )
    df = df.merge(umbral, on="ubigeo", how="left")

    # Casos de la semana SIGUIENTE (objetivo). No es un predictor.
    df["casos_siguiente"] = df.groupby("ubigeo")["casos"].shift(-1)
    df[TARGET] = (df["casos_siguiente"] > df["umbral_incidencia"]).astype("Int64")
    # Filas sin semana siguiente o sin umbral (distrito sin historial train) -> NaN
    df.loc[df["casos_siguiente"].isna() | df["umbral_incidencia"].isna(), TARGET] = pd.NA
    return df


def filtrar_distritos_con_historial(df: pd.DataFrame, min_semanas=None) -> pd.DataFrame:
    """Conserva solo distritos con suficiente historial (config.MIN_SEMANAS_DISTRITO)."""
    min_semanas = min_semanas if min_semanas is not None else config.MIN_SEMANAS_DISTRITO
    conteo = df.groupby("ubigeo")["week_id"].transform("size")
    return df[conteo >= min_semanas].copy()


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def build_master_dataset(raw_path=None, verbose=True) -> pd.DataFrame:
    """Ejecuta todo el preprocesamiento y devuelve el dataset maestro."""
    def log(*a):
        if verbose:
            print(*a)

    log("[1/7] Cargando dataset original...")
    raw = load_raw(raw_path)
    log(f"      filas crudas: {len(raw):,}")

    log("[2/7] Limpiando casos...")
    limpio = clean_raw(raw)
    log(f"      filas validas: {len(limpio):,}")

    log("[3/7] Construyendo calendario epidemiologico...")
    calendar = build_epi_calendar(limpio)

    log("[4/7] Agregando a distrito-semana...")
    agg = aggregate_district_week(limpio, calendar)
    log(f"      combinaciones distrito-semana observadas: {len(agg):,}")

    log("[5/7] Completando semanas sin registros (0 casos)...")
    completo = fill_zero_weeks(agg, calendar)
    log(f"      filas tras completar: {len(completo):,}")

    log("[6/7] Filtrando distritos con poco historial e ingenieria de features...")
    completo = filtrar_distritos_con_historial(completo)
    conf = add_features(completo)
    log(f"      distritos conservados: {conf['ubigeo'].nunique()}")

    log("[7/7] Variable objetivo y division temporal...")
    final = add_target_and_split(conf)

    cols = (
        ["ubigeo", "departamento", "provincia", "distrito", "ano", "semana",
         "fecha", "week_id", "casos", "edad_promedio", "prop_con_signos",
         "prop_grave", "prop_femenino"]
        + [c for c in FEATURE_NUMERIC if c not in ("casos",)]
        + ["umbral_incidencia", "split", TARGET]
    )
    final = final[cols]
    return final


if __name__ == "__main__":
    df = build_master_dataset()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DENGUE_SEMANAL, index=False, encoding="utf-8")
    print(f"\nGuardado: {config.DENGUE_SEMANAL}  ({len(df):,} filas)")
