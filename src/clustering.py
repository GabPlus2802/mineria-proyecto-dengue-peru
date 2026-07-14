"""Perfilado de distritos y clustering K-means.

Construye un perfil numerico por distrito y agrupa los distritos segun su
comportamiento epidemiologico, seleccionando el numero de clusters con el
metodo del codo y el coeficiente de silueta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config

PERFIL_COLS = [
    "promedio_semanal",
    "mediana_semanal",
    "maximo_semanal",
    "desviacion_semanal",
    "pct_semanas_alta",
    "crecimiento_promedio",
    "frecuencia_semanas_con_casos",
    "semana_pico",
]


def district_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil numerico por distrito (una fila por ubigeo)."""
    df = df.copy()

    def semana_pico(sub: pd.DataFrame) -> float:
        # Semana del anio con mayor promedio historico de casos
        prom = sub.groupby("semana")["casos"].mean()
        return float(prom.idxmax()) if len(prom) else np.nan

    filas = []
    for ubigeo, sub in df.groupby("ubigeo"):
        casos = sub["casos"]
        alta = sub["alta_incidencia_siguiente_semana"].dropna()
        filas.append({
            "ubigeo": ubigeo,
            "departamento": sub["departamento"].iloc[0],
            "distrito": sub["distrito"].iloc[0],
            "promedio_semanal": casos.mean(),
            "mediana_semanal": casos.median(),
            "maximo_semanal": casos.max(),
            "desviacion_semanal": casos.std(ddof=0),
            "pct_semanas_alta": alta.mean() if len(alta) else 0.0,
            "crecimiento_promedio": sub["crecimiento_semanal"].replace(
                [np.inf, -np.inf], np.nan).mean(),
            "frecuencia_semanas_con_casos": (casos > 0).mean(),
            "semana_pico": semana_pico(sub),
        })
    perfil = pd.DataFrame(filas).set_index("ubigeo")
    perfil[PERFIL_COLS] = perfil[PERFIL_COLS].fillna(0.0)
    return perfil


def evaluar_k(X_scaled: np.ndarray, k_min=None, k_max=None) -> pd.DataFrame:
    """Inercia (codo) y silueta para cada k en el rango configurado."""
    k_min = k_min or config.K_MIN
    k_max = k_max or config.K_MAX
    # La silueta requiere 2 <= k <= n_muestras-1
    k_max = min(k_max, len(X_scaled) - 1)
    k_min = min(k_min, k_max)
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        filas.append({
            "k": k,
            "inercia": km.inertia_,
            "silueta": silhouette_score(X_scaled, labels),
        })
    return pd.DataFrame(filas)


def run_kmeans(perfil: pd.DataFrame, k: int | None = None):
    """Ejecuta el flujo completo de clustering.

    Devuelve dict con: scaler, kmeans, k elegido, tabla de evaluacion, silueta,
    perfil con etiqueta de cluster y coordenadas PCA 2D.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(perfil[PERFIL_COLS].values)

    evaluacion = evaluar_k(X)
    # Seleccion: mejor silueta; si config fija K_SELECTED, se respeta.
    k_auto = int(evaluacion.loc[evaluacion["silueta"].idxmax(), "k"])
    k_final = k or config.K_SELECTED or k_auto

    kmeans = KMeans(n_clusters=k_final, random_state=config.RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)
    silueta_final = float(silhouette_score(X, labels))

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    coords = pca.fit_transform(X)

    perfil_out = perfil.copy()
    perfil_out["cluster"] = labels
    perfil_out["pca_1"] = coords[:, 0]
    perfil_out["pca_2"] = coords[:, 1]

    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "k": k_final,
        "k_auto_silueta": k_auto,
        "evaluacion": evaluacion,
        "silueta": silueta_final,
        "perfil": perfil_out,
        "pca_var": pca.explained_variance_ratio_.tolist(),
    }


def resumen_clusters(perfil: pd.DataFrame) -> pd.DataFrame:
    """Perfil promedio de cada cluster (para interpretacion)."""
    return (
        perfil.groupby("cluster")[PERFIL_COLS]
        .mean()
        .assign(n_distritos=perfil.groupby("cluster").size())
        .round(2)
    )
