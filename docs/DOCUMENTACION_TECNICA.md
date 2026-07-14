# Documentación técnica — Dashboard de dengue Perú

Este documento explica **cómo funciona el proyecto por dentro**: su arquitectura,
el flujo de datos, cada archivo, los modelos y las decisiones técnicas. Está pensado
para entender y defender el proyecto durante la exposición.

Para instrucciones de uso paso a paso, ver [MANUAL_USUARIO.md](MANUAL_USUARIO.md).

---

## 1. Visión general

El sistema toma el registro oficial de **casos individuales** de dengue del MINSA
(2000–2024) y lo convierte en un tablero interactivo que:

1. **Explora** los datos históricos (EDA).
2. **Agrupa** distritos por comportamiento (clustering).
3. **Clasifica** si un distrito tendrá *alta incidencia* la semana siguiente.
4. **Explica** cada predicción (SHAP).
5. **Pronostica** los próximos periodos (series temporales).
6. **Registra** consultas con un CRUD persistente.

El principio de diseño es **simple y mantenible**: la lógica pesada vive en `src/`
(funciones reutilizables), las páginas de Streamlit solo llaman a esas funciones y
muestran resultados, y todos los parámetros ajustables están en `config.py`.

---

## 2. Arquitectura y flujo de datos

```
 CSV crudo (data/raw/, 104 MB, NO versionado)
        │   src/preprocessing.py
        ▼
 dengue_semanal.csv  (dataset maestro, distrito×semana)
        │
        │   train.py  ──────────────────────────────────────────┐
        ▼                                                        │
 ┌───────────────┬───────────────────┬────────────────────┐     │
 │ clustering    │ clasificación     │ pronóstico         │     │
 │ K-means       │ RF / XGBoost      │ Holt-Winters       │     │
 ▼               ▼                   ▼                    │     │
 kmeans.joblib   random_forest.joblib  metricas_*.csv     │     │
 scaler.joblib   xgboost.joblib                           │     │
 distritos_      preprocessor.joblib                      │     │
 clusters.csv    model_meta.joblib                        │     │
        │                                                        │
        ▼   src/loaders.py (carga cacheada)                      │
 ┌──────────────────────────────────────────────────────────────┘
 │  app.py + pages/  (Streamlit)  ──►  usuario en el navegador
 │  pages/4_CRUD.py  ──►  src/database.py  ──►  Supabase / SQLite
 └──────────────────────────────────────────────────────────────
```

**Regla de oro:** los modelos se entrenan **una sola vez** con `train.py` y se
guardan en `models/`. El dashboard **solo carga** artefactos (nunca reentrena al
abrirse), gracias a `@st.cache_resource` y `@st.cache_data`.

---

## 3. La unidad de análisis: distrito × semana

El dataset original tiene **una fila por caso**. Para modelar, se agrega a
**un distrito en una semana epidemiológica**:

- **Clave:** `ubigeo` (código de 6 dígitos, más confiable que el nombre) + `semana`.
- **`casos`:** número de casos notificados esa semana en ese distrito.
- **Semanas sin registro:** como el dataset es una *lista de notificaciones*, una
  semana ausente dentro del periodo activo del distrito se interpreta como **0 casos**.
- **Fecha:** se construye de forma **monótona y única** por `(año, semana)` con
  `1-ene + (semana-1)·7 días` (aproximación suficiente para ordenar y graficar).

---

## 4. Ingeniería de características (sin fuga de información)

Por cada fila distrito-semana se calculan predictores usando **solo el pasado**:

| Variable | Definición |
|---|---|
| `casos` | Casos de la semana actual |
| `casos_lag_1/2/4` | Casos de 1, 2 y 4 semanas atrás |
| `promedio_movil_4/8` | Media de las 4 / 8 semanas **anteriores** (`shift(1)` + `rolling`) |
| `desviacion_movil_4` | Desviación de las 4 semanas anteriores |
| `crecimiento_semanal` | Variación relativa respecto a la semana anterior |
| `semana_sen`, `semana_cos` | Estacionalidad (codificación circular de la semana) |
| `departamento` | Variable categórica (One-Hot en el pipeline) |

> **Clave anti-fuga:** las medias móviles aplican `shift(1)` **antes** del `rolling`,
> por lo que nunca incluyen la semana que se está prediciendo.

### Variable objetivo

`alta_incidencia_siguiente_semana`:

```
1 = los casos de la SEMANA SIGUIENTE superan el percentil 75 histórico del distrito
0 = no lo superan
```

- El **umbral (percentil 75) se calcula solo con el periodo de entrenamiento**
  (≤ 2022), nunca con validación/prueba → sin fuga de información.
- Es una etiqueta **estadística**, no una definición epidemiológica oficial de brote.

### División temporal (sin mezclar futuro y pasado)

| Partición | Años | Uso |
|---|---|---|
| Entrenamiento | 2000–2022 | Ajustar modelos y umbral |
| Validación | 2023 | Referencia intermedia |
| Prueba | 2024 | Evaluación final |

---

## 5. Descripción de cada archivo

### Raíz
| Archivo | Función |
|---|---|
| `app.py` | Portada del dashboard (problema, fuente, estado de modelos, avisos). |
| `config.py` | **Todos** los parámetros ajustables (semilla, k, RF/XGB, umbral, pronóstico, rutas). |
| `train.py` | Script único que genera todos los artefactos (`python train.py`). |
| `requirements.txt` | Dependencias. |
| `.gitignore` | Ignora secretos, entorno virtual, dataset crudo y BD local. |

### `src/` (lógica reutilizable)
| Módulo | Responsabilidad |
|---|---|
| `preprocessing.py` | Limpieza, agregación distrito-semana, features, objetivo y split. |
| `clustering.py` | Perfil por distrito + K-means (codo, silueta, PCA). |
| `modeling.py` | Pipeline RF/XGBoost, métricas, umbral, SHAP y fila de predicción. |
| `forecasting.py` | Serie temporal, media móvil, Holt-Winters, MAPE seguro y RMSE. |
| `visualizations.py` | Gráficos Plotly reutilizables (histogramas, correlación, clusters, etc.). |
| `database.py` | Capa CRUD desacoplada: Supabase o SQLite local. |
| `loaders.py` | Carga **cacheada** de datos y modelos para Streamlit. |

### `pages/` (una página por panel)
| Página | Contenido |
|---|---|
| `1_EDA_Clustering.py` | Filtros, resumen, EDA (evolución, distribución, correlación), outliers 1.5·IQR, clustering. |
| `2_Modelo_Predictivo.py` | RF vs XGBoost, umbral, matriz de confusión, SHAP global/local, predicción en vivo. |
| `3_Pronostico.py` | Serie por nivel, evaluación (MAPE/RMSE), pronóstico 4+ semanas con intervalo. |
| `4_CRUD.py` | Crear, listar, editar y eliminar consultas. |

### `data/`, `models/`, `notebooks/`, `tests/`
- `data/raw/`: dataset original (no versionado).
- `data/processed/`: `dengue_semanal.csv` y CSV de clusters/métricas.
- `models/`: modelos y preprocesador (`.joblib`).
- `notebooks/01–04`: exploración/validación por fase.
- `tests/`: pruebas mínimas con pytest.

---

## 6. Los cuatro paneles en detalle

### Panel 1 — EDA y Clustering
- **EDA:** total de registros y casos, número de deptos/distritos, periodo,
  estadísticas descriptivas, histograma, boxplot, matriz de correlación, evolución
  temporal y casos por departamento/distrito, con **filtros interactivos**.
- **Outliers:** regla `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`. Los valores altos (brotes
  reales) **se conservan**: son justamente lo que el modelo debe detectar.
- **Clustering:** cada distrito se resume en 8 variables (promedio, mediana, máximo,
  desviación, % de semanas de alta incidencia, crecimiento, frecuencia de semanas con
  casos, semana típica del pico). Se escala con `StandardScaler`, se evalúa `k=2..8`
  con **método del codo** y **coeficiente de silueta**, se elige `k`, se entrena
  K-means y se visualiza en **PCA 2D**. Resultado: 3 grupos (baja/media/alta transmisión).

### Panel 2 — Modelo Predictivo y Explicabilidad
- **Modelos:** Random Forest y XGBoost dentro de un `Pipeline` con
  `ColumnTransformer` (imputación + escala numérica, One-Hot con
  `handle_unknown="ignore"`). El preprocesador se ajusta **solo con entrenamiento**.
- **Desbalance:** la clase mayoritaria en train es ~90%, por lo que se aplica
  `class_weight="balanced"` (RF) y `scale_pos_weight` (XGBoost).
- **Métricas:** accuracy, precision, recall, F1, ROC-AUC y matriz de confusión
  (TP/TN/FP/FN). El ganador se elige por **F1** (equilibrio), no por accuracy.
- **Umbral:** deslizador que muestra el efecto sobre precision/recall/F1 y FP/FN.
- **SHAP:** `TreeExplainer` sobre el mejor modelo; `summary_plot` global y
  `waterfall` local. Muestra **asociación, no causalidad**.
- **Predicción en vivo:** el usuario elige distrito; las variables derivadas se
  calculan automáticamente desde el historial (no se piden variables técnicas).

### Panel 3 — Pronóstico
- **Serie:** casos agregados a nivel nacional, de departamento o de distrito.
- **Modelos:** media móvil (baseline) vs **Holt-Winters** (suavizado exponencial,
  estacionalidad de 52 semanas si hay historia suficiente).
- **Evaluación:** se separa un tramo final cronológico; se reportan **MAPE seguro**
  (excluye semanas con 0 casos reales para no dividir entre cero) y **RMSE**.
- **Pronóstico:** ≥ 4 semanas futuras con intervalo aproximado (±1.96·σ de residuos).

### Panel 4 — CRUD
- Registra las consultas/predicciones realizadas.
- **Persistencia:** Supabase en producción; si no hay credenciales, cae
  automáticamente a **SQLite local** (`data/consultas_local.db`). Misma interfaz en
  ambos modos. Campos: departamento, distrito, semana, datos de entrada (JSON),
  modelo, predicción, probabilidad y timestamps automáticos.

---

## 7. Resultados reales (test = 2024)

**Clasificación** (umbral 0.50):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.727 | 0.669 | 0.921 | 0.775 | 0.879 |
| XGBoost (mejor por F1) | 0.736 | 0.680 | 0.912 | 0.779 | 0.879 |

**Pronóstico nacional:** Holt-Winters MAPE **31.7 %** / RMSE 417 vs media móvil
44.9 % / 459. **Clustering:** k = 3, silueta ≈ 0.35.

> Los números exactos se regeneran con `python train.py` y quedan en
> `data/processed/metricas_*.csv`; deben coincidir con lo mostrado en el dashboard.

### Nota importante: cambio de distribución
La proporción de "alta incidencia" pasa de ~10 % en entrenamiento a ~50 % en prueba.
**No es un error:** refleja la **epidemia real de dengue de 2023–2024**. Como el
umbral se fija solo con datos de entrenamiento (por rigor metodológico), casi todas
las semanas de 2024 lo superan. El alto **recall (≈0.92)** indica que el modelo
detecta bien las semanas de alta actividad.

---

## 8. Decisiones técnicas destacadas

1. **No se eliminan los duplicados de fila** del CSV original: cada fila es un
   paciente distinto; borrarlos perdería casos reales.
2. **Edad normalizada** a años según `tipo_edad` (A/M/D); edades imposibles → nulas.
3. **Random Forest acotado** (`max_depth=16`, `min_samples_leaf=20`) para que el
   `.joblib` sea desplegable (786 MB → 21 MB) sin perder desempeño.
4. **Modelos comprimidos** con `joblib.dump(..., compress=3)`.
5. **Rutas relativas** vía `config.py` (funciona igual en local y en la nube).
6. **Sin secretos en el repo:** credenciales solo en `.streamlit/secrets.toml`
   (ignorado); se versiona únicamente el `.example`.

---

## 9. Reproducibilidad

```bash
pip install -r requirements.txt
python train.py --rebuild   # reconstruye todo desde el CSV crudo
pytest -q                   # 18 pruebas mínimas
streamlit run app.py        # levanta el dashboard
```

Todo resultado mostrado en el dashboard proviene de estas ejecuciones reales y es
reproducible; las métricas coinciden entre `train.py`, los CSV generados y la app.
