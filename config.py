"""Parametros centralizados del proyecto.

Todos los valores que el profesor puede pedir modificar en vivo estan aqui.
Ver la seccion "Modificaciones rapidas para la exposicion" en el README.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas (siempre relativas a la raiz del proyecto)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Nombre real del dataset original (ver "Estado inicial del dataset" en README)
RAW_DATASET = DATA_RAW / "datos_abiertos_vigilancia_dengue_2000_2024.csv"
RAW_SEP = ";"
RAW_ENCODING = "utf-8-sig"

# Artefactos procesados
DENGUE_SEMANAL = DATA_PROCESSED / "dengue_semanal.csv"
DISTRITOS_CLUSTERS = DATA_PROCESSED / "distritos_clusters.csv"
METRICAS_CLASIFICACION = DATA_PROCESSED / "metricas_clasificacion.csv"
METRICAS_BALANCEO = DATA_PROCESSED / "metricas_balanceo.csv"
METRICAS_PRONOSTICO = DATA_PROCESSED / "metricas_pronostico.csv"

# Modelos
PATH_PREPROCESSOR = MODELS_DIR / "preprocessor.joblib"
PATH_RANDOM_FOREST = MODELS_DIR / "random_forest.joblib"
PATH_XGBOOST = MODELS_DIR / "xgboost.joblib"
PATH_MODELOS_EXTRA = MODELS_DIR / "modelos_extra.joblib"  # gradient_boosting, logistica, arbol
PATH_KMEANS = MODELS_DIR / "kmeans.joblib"
PATH_SCALER_CLUSTERING = MODELS_DIR / "scaler_clustering.joblib"
PATH_MODEL_META = MODELS_DIR / "model_meta.joblib"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Clustering (K-means)
# ---------------------------------------------------------------------------
K_MIN = 2
K_MAX = 8
K_SELECTED = 3  # se puede sobrescribir con la seleccion automatica por silueta

# ---------------------------------------------------------------------------
# Clasificacion
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 16          # acotado: evita un modelo de cientos de MB y limita overfitting
RF_MIN_SAMPLES_LEAF = 20   # hojas con >=20 muestras -> arbol mas compacto y estable

XGB_N_ESTIMATORS = 200
XGB_LEARNING_RATE = 0.10

CLASSIFICATION_THRESHOLD = 0.50

# Percentil historico por distrito que define "alta incidencia" (variable objetivo)
TARGET_PERCENTILE = 75

# Historial minimo de semanas por distrito para incluirlo en el modelado
MIN_SEMANAS_DISTRITO = 30

# ---------------------------------------------------------------------------
# Pronostico
# ---------------------------------------------------------------------------
FORECAST_PERIODS = 4
MOVING_AVERAGE_WINDOW = 4

# ---------------------------------------------------------------------------
# Division temporal (por anio). El ultimo anio es prueba, el previo validacion.
# ---------------------------------------------------------------------------
TEST_YEARS = 1        # ultimos N anios -> prueba
VALIDATION_YEARS = 1  # N anios anteriores a prueba -> validacion
