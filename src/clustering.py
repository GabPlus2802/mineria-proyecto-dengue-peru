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


# ---------------------------------------------------------------------------
# Interpretacion: de "cluster 0/1/2" a un perfil epidemiologico con nombre
# ---------------------------------------------------------------------------
# Nombres por NIVEL de transmision, no por el numero que asigno K-means: la
# etiqueta numerica cambia entre ejecuciones, el orden por intensidad no.
NOMBRES_POR_NIVEL = [
    ("Transmision esporadica",
     "Distritos donde el dengue aparece de forma aislada: la mayoria de semanas "
     "cierran sin ningun caso y los brotes, cuando ocurren, son pequenos y "
     "breves. Son la mayor parte del pais."),
    ("Transmision estacional",
     "Distritos con presencia recurrente del vector: acumulan casos durante la "
     "temporada de lluvias y bajan el resto del ano. El brote es predecible en "
     "el calendario, aunque su tamano varie."),
    ("Transmision alta y sostenida",
     "Focos endemicos: notifican casos en buena parte del ano y concentran los "
     "picos mas grandes del pais. Son pocos distritos, pero explican una "
     "fraccion enorme de los casos totales."),
    ("Transmision muy alta",
     "Nucleo de maxima intensidad, con actividad casi continua y los brotes de "
     "mayor magnitud registrados."),
]


def etiquetar_clusters(perfil: pd.DataFrame) -> pd.DataFrame:
    """Anade nombre, descripcion y orden de intensidad a cada cluster.

    Los clusters se ordenan por su promedio semanal de casos y se les asigna un
    nombre segun ese nivel. Asi la interpretacion no depende de la etiqueta
    numerica que haya elegido K-means en esa ejecucion.
    """
    resumen = (
        perfil.groupby("cluster")
        .agg(promedio_semanal=("promedio_semanal", "mean"),
             maximo_semanal=("maximo_semanal", "mean"),
             frecuencia_semanas_con_casos=("frecuencia_semanas_con_casos", "mean"),
             pct_semanas_alta=("pct_semanas_alta", "mean"),
             semana_pico=("semana_pico", "mean"),
             n_distritos=("promedio_semanal", "size"))
        .sort_values("promedio_semanal")
        .reset_index()
    )
    n = len(resumen)
    if n == 2:
        # Con dos grupos conviene saltar el nivel intermedio
        elegidos = [NOMBRES_POR_NIVEL[0], NOMBRES_POR_NIVEL[2]]
    elif n <= len(NOMBRES_POR_NIVEL):
        elegidos = NOMBRES_POR_NIVEL[:n]
    else:
        elegidos = NOMBRES_POR_NIVEL + [
            (f"Nivel {i}", "Grupo adicional de intensidad intermedia.")
            for i in range(len(NOMBRES_POR_NIVEL), n)
        ]

    resumen["nivel"] = range(n)
    resumen["nombre"] = [e[0] for e in elegidos[:n]]
    resumen["descripcion"] = [e[1] for e in elegidos[:n]]
    return resumen


def aporte_de_casos(df: pd.DataFrame, perfil: pd.DataFrame) -> pd.Series:
    """Porcentaje del total nacional de casos que aporta cada cluster.

    Es la cifra que vuelve tangible el resultado del clustering: unos pocos
    distritos concentran la mayor parte de la carga de enfermedad.
    """
    mapa = perfil["cluster"]
    casos = df.groupby("ubigeo")["casos"].sum()
    por_cluster = casos.groupby(mapa.reindex(casos.index)).sum()
    return (por_cluster / por_cluster.sum() * 100).round(1)
