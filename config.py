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

# Semanas reservadas al final de la serie para evaluar los modelos.
# 13 semanas = un trimestre epidemiologico. Con 8 semanas o menos el componente
# estacional de 52 semanas no alcanza a expresarse y la media movil (que
# pronostica una constante) gana por construccion. Que modelo gana DEPENDE de
# esta ventana: el Panel 3 muestra la tabla de robustez con varias ventanas.
FORECAST_EVAL_PERIODS = 13
FORECAST_EVAL_VENTANAS = [8, 13, 26]  # ventanas de la tabla de robustez

# ---------------------------------------------------------------------------
# Division temporal (por anio). El ultimo anio REAL es prueba, el previo validacion.
# Las filas simuladas quedan fuera (split = "simulado").
# ---------------------------------------------------------------------------
TEST_YEARS = 1        # ultimos N anios -> prueba
VALIDATION_YEARS = 1  # N anios anteriores a prueba -> validacion

# ---------------------------------------------------------------------------
# Extension SIMULADA del dataset (src/simulation.py)
#
# La vigilancia publicada del MINSA llega hasta 2024. Para que el pronostico se
# muestre en fechas vigentes se generan registros posteriores por bootstrap
# estacional. NO SON DATOS REALES: van marcados con origen = "simulado" y no
# intervienen en el entrenamiento, las metricas ni el clustering.
# ---------------------------------------------------------------------------
SIMULAR_EXTENSION = True   # False -> el dataset se queda solo con datos reales
SIM_END_ANO = 2026         # horizonte de la extension
SIM_END_SEMANA = 22        # semana 22 de 2026 ~ finales de mayo
SIM_LOOKBACK_ANIOS = 6     # anios historicos que alimentan el muestreo
SIM_ANIOS_ACTIVIDAD = 2    # solo se extienden distritos activos en los ultimos N anios reales
SIM_VENTANA_SEMANAS = 2    # ventana estacional +-N semanas alrededor de la semana objetivo
SIM_DECAIMIENTO = 0.65     # peso por recencia: peso = DECAIMIENTO ** (antiguedad en anios)
SIM_RUIDO_SIGMA = 0.25     # amplitud del ruido multiplicativo (escala log)
SIM_PERSISTENCIA = 0.75    # AR(1): que tanto persiste la intensidad de una semana a la siguiente
SIM_SUAVIZADO = 3          # ventana de suavizado del nivel estacional muestreado
# Factor de intensidad por anio: refleja el descenso posterior al pico epidemico
# de 2023-2024. Editable para mostrar escenarios en la exposicion.
SIM_INTENSIDAD = {2025: 0.80, 2026: 0.90}
