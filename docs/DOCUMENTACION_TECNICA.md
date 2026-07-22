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
 │ K-means       │ 5 clasificadores  │ Holt-Winters       │     │
 ▼               ▼                   ▼                    │     │
 kmeans.joblib   random_forest.joblib  metricas_*.csv     │     │
 scaler.joblib   xgboost.joblib                           │     │
 distritos_      preprocessor.joblib                      │     │
 clusters.csv    model_meta.joblib                        │     │
        │                                                        │
        ▼   src/loaders.py (carga cacheada)                      │
 ┌──────────────────────────────────────────────────────────────┘
 │  app.py + views/  (Streamlit)  ──►  usuario en el navegador
 │  views/crud.py    ──►  src/database.py  ──►  Supabase / SQLite
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
| **Simulado** | **2025–2026** | **Ninguno**: excluido del modelado |

La división se calcula **solo sobre los años reales**, de modo que el conjunto de
prueba nunca puede caer sobre registros generados.

### Extensión simulada del dataset (`src/simulation.py`)

La vigilancia publicada del MINSA termina en 2024. Para que el pronóstico se
muestre en fechas vigentes, el maestro se extiende hasta la semana 22 de 2026
(28/05/2026) con **40 922 filas generadas** en 553 distritos.

**Método — bootstrap estacional por distrito:**

1. Solo se extienden distritos activos en los últimos 2 años reales: uno que dejó
   de reportar hace años no se "reactiva".
2. Para cada semana objetivo `w` se muestrea de los casos históricos del propio
   distrito en las semanas `[w−2, w+2]` (ventana circular), de los últimos 6 años,
   ponderando por recencia (`peso = 0.65^antigüedad`).
3. Se aplica un factor de intensidad anual (2025: 0.80, 2026: 0.90).
4. El nivel se **suaviza** con una ventana de 3 y se multiplica por una
   **intensidad persistente AR(1)** (ρ = 0.75). Sin este paso la serie generada
   tendría un salto artificial semana a semana que las curvas epidémicas reales
   no presentan.
5. Todas las features derivadas (lags, medias móviles, objetivo) se recalculan
   sobre la serie completa para que no queden discontinuidades en el empalme.

**Garantías (con pruebas automáticas en `tests/test_simulation.py`):** no se
altera ninguna fila real, las generadas quedan fuera de train/val/test, el
conjunto de prueba sigue siendo 2024, la salida es reproducible con
`RANDOM_STATE` y el pico estacional se conserva.

Todo es desactivable con `python train.py --sin-simulacion` o
`config.SIMULAR_EXTENSION = False`.

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
| `simulation.py` | Extensión **simulada** 2025–mayo 2026 por bootstrap estacional (`origen`/`split` = `simulado`). |
| `clustering.py` | Perfil por distrito + K-means (codo, silueta, PCA). |
| `modeling.py` | Pipelines de los 5 modelos, métricas, umbral, explicadores SHAP y fila de predicción. |
| `forecasting.py` | Serie temporal, media móvil, Holt-Winters, MAPE seguro y RMSE. |
| `visualizations.py` | Gráficos Plotly y **paleta validada para daltonismo** sobre superficie oscura. |
| `ui.py` | CSS del tema, cabeceras, tarjetas KPI y aviso de datos simulados. |
| `database.py` | Capa CRUD desacoplada: Supabase o SQLite local. |
| `loaders.py` | Carga **cacheada** de datos y modelos para Streamlit. |

### `views/` (una página por sección)
| Página | Contenido |
|---|---|
| `acerca.py` | Portada: problema, fuente, definición del objetivo, equipo y advertencias. |
| `eda_clustering.py` | Filtros, resumen, EDA (evolución, distribución, correlación), outliers 1.5·IQR, clustering. |
| `modelo_predictivo.py` | Simulador con sliders, comparación de 5 modelos, umbral, matriz de confusión, SHAP global/local. |
| `pronostico.py` | Serie por nivel, evaluación (MAPE/RMSE), pronóstico con intervalo y tabla de robustez. |
| `crud.py` | Crear, listar, editar y eliminar consultas. |

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
- **Modelos (5):** Random Forest, XGBoost, Gradient Boosting (HistGradientBoosting),
  Regresión Logística y Árbol de Decisión, cada uno dentro de un `Pipeline` con
  `ColumnTransformer` (imputación + escala numérica, One-Hot con
  `handle_unknown="ignore"`). El preprocesador se ajusta **solo con entrenamiento**.
  Se cubren tres familias: lineal (logística), árbol simple y ensembles.
- **Desbalance:** la clase mayoritaria en train es ~90%, por lo que se aplica
  `class_weight="balanced"` (RF) y `scale_pos_weight` (XGBoost).
- **Métricas:** accuracy, precision, recall, F1, ROC-AUC y matriz de confusión
  (TP/TN/FP/FN). El ganador se elige por **F1** (equilibrio), no por accuracy.
- **Umbral:** deslizador que muestra el efecto sobre precision/recall/F1 y FP/FN.
- **SHAP:** `TreeExplainer` en los modelos de árbol y `LinearExplainer` en la
  Regresión Logística, de modo que el mejor modelo por F1 también queda explicado.
  `summary_plot` global y barras de contribución locales. Muestra **asociación,
  no causalidad**.
- **Simulador de predicción:** un slider por cada variable del modelo, precargados
  con los valores reales del distrito elegido. La probabilidad y la explicación
  SHAP se recalculan en cada movimiento. `semana_sen`/`semana_cos` se derivan de
  un slider de semana epidemiológica en vez de editarse a mano. El botón
  *Sincronizar derivadas* recalcula promedios, variabilidad y crecimiento desde
  los lags para volver a un escenario coherente.

### Panel 3 — Pronóstico
- **Serie:** casos agregados a nivel nacional, de departamento o de distrito.
- **Modelos:** media móvil (baseline) vs **Holt-Winters** (suavizado exponencial,
  estacionalidad de 52 semanas si hay historia suficiente).
- **Evaluación:** se separa un tramo final cronológico (13 semanas por defecto);
  se reportan **MAPE seguro** (excluye semanas con 0 casos reales para no dividir
  entre cero) y **RMSE**.
- **Robustez:** qué modelo gana depende de la ventana de evaluación, así que el
  panel publica la tabla con varias ventanas en lugar de una sola cifra.
- **Calendario:** la serie se reindexa sobre el calendario epidemiológico real. Un
  `asfreq("W-MON")` previo anulaba todo año cuyo 1 de enero no cayera en lunes
  (2025 y 2026), dejando la serie en ceros; hay pruebas de regresión para esto.
- **Pronóstico:** 4–26 semanas futuras con intervalo aproximado (±1.96·σ de residuos).

### Panel 4 — CRUD
- Registra las consultas/predicciones realizadas.
- **Persistencia:** Supabase en producción; si no hay credenciales, cae
  automáticamente a **SQLite local** (`data/consultas_local.db`). Misma interfaz en
  ambos modos. Campos: departamento, distrito, semana, datos de entrada (JSON),
  modelo, predicción, probabilidad y timestamps automáticos.

---

## 7. Resultados reales (test = 2024, solo datos reales)

**Clasificación** (umbral 0.50, 5 modelos, ordenado por F1):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Regresión Logística (mejor por F1)** | 0.770 | 0.756 | 0.806 | **0.780** | 0.855 |
| XGBoost | 0.729 | 0.671 | 0.914 | 0.774 | 0.876 |
| Gradient Boosting | 0.724 | 0.665 | 0.920 | 0.772 | 0.876 |
| Random Forest | 0.720 | 0.660 | 0.923 | 0.769 | 0.875 |
| Árbol de Decisión | 0.716 | 0.660 | 0.908 | 0.764 | 0.867 |

**Pronóstico nacional** (ventana de 13 semanas, medido **solo sobre datos
observados**): Holt-Winters MAPE **27.2 %** / RMSE 313 vs media móvil 42.4 % / 464.
Holt-Winters gana también con ventanas de 8 y 26 semanas: ver la tabla de
robustez del §12 del README.

El modelo se ajusta en **escala logarítmica con tendencia amortiguada**. Los
brotes crecen de forma multiplicativa, así que `log(1+casos)` estabiliza la
varianza e impide que el pico de 2023–2024 domine el ajuste; la amortiguación
evita extrapolar una caída indefinida tras un pico. El intervalo resultante es
asimétrico, que es lo correcto para un conteo.

**Clustering:** k = 3, silueta ≈ **0.569** (437 distritos de transmisión
esporádica, 111 estacional y 23 alta y sostenida). K-means usa 5 de las 8
variables del perfil: se excluyen `semana_pico` (circular), `pct_semanas_alta`
(derivada del objetivo de la clasificación) y `crecimiento_promedio` (dominada
por el ruido en distritos con pocos casos).

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
