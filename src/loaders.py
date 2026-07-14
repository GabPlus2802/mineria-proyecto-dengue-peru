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
    return df


@st.cache_data(show_spinner=False)
def load_clusters() -> pd.DataFrame:
    return pd.read_csv(config.DISTRITOS_CLUSTERS)


@st.cache_data(show_spinner=False)
def load_metricas_clasificacion() -> pd.DataFrame:
    return pd.read_csv(config.METRICAS_CLASIFICACION)


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
    return modelos


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
