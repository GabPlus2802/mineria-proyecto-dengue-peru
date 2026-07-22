"""Carga cacheada de datos y modelos para las paginas de Streamlit.

Usa @st.cache_data para dataframes y @st.cache_resource para modelos, de modo
que no se recargan ni se reentrenan en cada interaccion.
"""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

import config


@st.cache_data(show_spinner="Cargando datos procesados...")
def load_master() -> pd.DataFrame:
    df = pd.read_csv(config.DENGUE_SEMANAL)
    df["fecha"] = pd.to_datetime(df["fecha"])
    if "origen" not in df.columns:
        df["origen"] = "real"
    return df


def solo_real(df: pd.DataFrame) -> pd.DataFrame:
    """Filas de vigilancia real del MINSA (excluye la extension simulada)."""
    return df[df["origen"] == "real"] if "origen" in df.columns else df


def corte_simulado(df: pd.DataFrame):
    """Ultima fecha con dato real, o None si el dataset no tiene extension."""
    if "origen" not in df.columns or not (df["origen"] == "simulado").any():
        return None
    return pd.to_datetime(solo_real(df)["fecha"]).max()


@st.cache_data(show_spinner=False)
def resumen_simulacion() -> dict:
    """Cifras de la extension simulada, para los avisos del dashboard."""
    from src import simulation

    return simulation.resumen(load_master())


@st.cache_data(show_spinner=False)
def load_clusters() -> pd.DataFrame:
    return pd.read_csv(config.DISTRITOS_CLUSTERS)


@st.cache_data(show_spinner=False)
def load_metricas_clasificacion() -> pd.DataFrame:
    return pd.read_csv(config.METRICAS_CLASIFICACION)


@st.cache_data(show_spinner=False)
def load_metricas_balanceo() -> pd.DataFrame | None:
    return pd.read_csv(config.METRICAS_BALANCEO) if config.METRICAS_BALANCEO.exists() else None


@st.cache_data(show_spinner=False)
def load_metricas_pronostico() -> pd.DataFrame:
    return pd.read_csv(config.METRICAS_PRONOSTICO)


@st.cache_resource(show_spinner="Cargando modelos...")
def load_models() -> dict:
    """Carga modelos y metadatos. Devuelve dict; None si faltan artefactos."""
    modelos = {}
    for nombre, ruta in [
        ("random_forest", config.PATH_RANDOM_FOREST),
        ("xgboost", config.PATH_XGBOOST),
        ("kmeans", config.PATH_KMEANS),
        ("scaler_clustering", config.PATH_SCALER_CLUSTERING),
        ("preprocessor", config.PATH_PREPROCESSOR),
        ("meta", config.PATH_MODEL_META),
    ]:
        modelos[nombre] = joblib.load(ruta) if ruta.exists() else None
    # Modelos adicionales (gradient_boosting, logistic_regression, decision_tree)
    if config.PATH_MODELOS_EXTRA.exists():
        modelos.update(joblib.load(config.PATH_MODELOS_EXTRA))
    return modelos


def load_clasificadores(modelos: dict) -> dict:
    """Devuelve solo los pipelines de clasificacion presentes, en orden."""
    from src.modeling import MODEL_LABELS

    return {k: modelos[k] for k in MODEL_LABELS if modelos.get(k) is not None}


def artefactos_listos() -> bool:
    """True si existen los artefactos minimos para el dashboard."""
    requeridos = [
        config.DENGUE_SEMANAL,
        config.PATH_RANDOM_FOREST,
        config.PATH_XGBOOST,
        config.PATH_KMEANS,
    ]
    return all(p.exists() for p in requeridos)


def distritos_de(df: pd.DataFrame, departamento: str) -> pd.DataFrame:
    """Distritos (ubigeo, nombre) de un departamento, ordenados."""
    sub = df[df["departamento"] == departamento][["ubigeo", "distrito"]].drop_duplicates()
    return sub.sort_values("distrito").reset_index(drop=True)
