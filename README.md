# Análisis, clasificación y pronóstico del dengue en el Perú

Dashboard académico de minería de datos sobre la vigilancia epidemiológica del
dengue en el Perú (MINSA, 2000–2024). Explora los datos históricos, agrupa
distritos, clasifica el riesgo de alta incidencia para la semana siguiente,
explica las predicciones con SHAP y pronostica los próximos periodos.

> ⚠️ **Uso académico.** Las etiquetas de "alta incidencia" son estadísticas
> (percentil histórico por distrito), no una definición epidemiológica oficial de
> brote. Las explicaciones SHAP muestran asociación, no causalidad.

---

## 1. Descripción del problema

El dengue, transmitido por el mosquito *Aedes aegypti*, es una de las principales
emergencias de salud pública en el Perú, con una expansión notable en 2023–2024.
Este proyecto construye un sistema analítico en línea para apoyar la comprensión y
anticipación de la actividad del dengue a nivel distrital.

## 2. Fuente del dataset

- **Origen:** MINSA — Vigilancia epidemiológica del dengue (Plataforma Nacional de
  Datos Abiertos del Perú).
- **Archivo original:** `datos_abiertos_vigilancia_dengue_2000_2024.csv`
- **Descarga:** el archivo original (~104 MB) **no** se versiona en GitHub. Colócalo
  en `data/raw/` antes de reconstruir el dataset (ver *Instalación*).

## 3. Periodo y unidad de análisis

- **Periodo:** 2000–2024 (semanas epidemiológicas 1–53).
- **Granularidad original:** caso individual notificado (1 fila = 1 caso).
- **Unidad de análisis:** **distrito × semana epidemiológica**.

---

## Estado inicial del dataset

Auditoría del CSV original (`datos_abiertos_vigilancia_dengue_2000_2024.csv`):

| Propiedad | Valor |
|---|---|
| Formato / separador | CSV / `;` |
| Codificación | UTF-8 con BOM (`utf-8-sig`) |
| Filas | 1 029 421 (nivel caso individual) |
| Columnas | 14 |
| Periodo | 2000–2024 · semanas 1–53 |
| Departamentos / provincias / distritos | 23 / 124 / 625 |
| Ubigeos | 662 (6 dígitos) — llave geográfica real |
| Severidad (`enfermedad`) | SIN SIGNOS 915 243 · CON SIGNOS 110 166 · GRAVE 4 012 |
| `diagnostic` | A97.0 / A97.1 / A97.2 (coincide con severidad) |
| Edad | `tipo_edad` A/M/D (años/meses/días); valores basura aislados (máx. 71 M) |
| Duplicados de fila | 152 737 → **casos distintos legítimos, no se eliminan** |
| Nulos relevantes | `localidad` (122 598), `localcod` (148 641) — no críticos |

**Columnas:** `departamento, provincia, distrito, localidad, enfermedad, ano,
semana, diagnostic, diresa, ubigeo, localcod, edad, tipo_edad, sexo`.

**Decisiones técnicas clave:**
- El dataset es una **lista de notificaciones**: una combinación distrito-semana
  ausente dentro del periodo activo del distrito se interpreta como **0 casos**.
- Los **duplicados de fila no se eliminan** (cada fila es un paciente distinto).
- La edad se normaliza a años según `tipo_edad`; edades > 120 o < 0 → nulas.
- La fecha semanal se genera de forma **monótona y única** por `(año, semana)`.
- La variable objetivo es **desbalanceada y con cambio de distribución** entre
  entrenamiento (~10 % positivos) y prueba (~50 %), reflejando la epidemia real de
  2023–2024. El umbral se calcula **solo con datos de entrenamiento** (sin fuga).

---

## 4. Funcionalidades

- **Panel 1 — EDA y Clustering:** estadísticas, histogramas, boxplots, correlación,
  evolución temporal, casos por departamento/distrito, outliers (1.5·IQR) y
  agrupamiento K-means (codo + silueta + PCA 2D).
- **Panel 2 — Modelo Predictivo:** Random Forest vs XGBoost, métricas completas,
  matriz de confusión, barrido de umbral, SHAP global y local, y formulario de
  predicción en vivo.
- **Panel 3 — Pronóstico:** media móvil (baseline) vs Holt-Winters, MAPE seguro y
  RMSE, proyección de 4+ semanas con intervalo.
- **Panel 4 — CRUD:** registrar, listar, editar y eliminar consultas con
  persistencia en Supabase (o SQLite local en desarrollo).

## 5. Estructura del proyecto

```
mineria-proyecto-dengue-peru/
├── app.py                  # Portada del dashboard
├── config.py               # Parámetros centralizados (editables en vivo)
├── train.py                # Genera TODOS los artefactos
├── requirements.txt
├── data/
│   ├── raw/                # dataset original (no versionado)
│   └── processed/          # dengue_semanal.csv, clusters, métricas
├── models/                 # *.joblib (modelos y preprocesador)
├── pages/                  # 4 páginas de Streamlit
├── src/                    # preprocessing, modeling, forecasting,
│                           # clustering, visualizations, database, loaders
├── notebooks/              # 01–04 exploración y validación
└── tests/                  # pruebas mínimas (pytest)
```

## 6. Instalación

```bash
# 1. Clonar
git clone https://github.com/GabPlus2802/mineria-proyecto-dengue-peru.git
cd mineria-proyecto-dengue-peru

# 2. Entorno virtual (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Dependencias
pip install -r requirements.txt

# 4. Dataset original (solo si vas a reconstruir el maestro)
#    Copiar datos_abiertos_vigilancia_dengue_2000_2024.csv en data/raw/
```

## 7. Ejecución del entrenamiento

```bash
python train.py            # usa data/processed/dengue_semanal.csv si existe
python train.py --rebuild  # reconstruye el maestro desde data/raw/ (necesita el CSV original)
```

Genera: `dengue_semanal.csv`, `distritos_clusters.csv`,
`metricas_clasificacion.csv`, `metricas_pronostico.csv` y los `.joblib` en `models/`.

## 8. Ejecución de Streamlit

```bash
streamlit run app.py
```

## 9. Configuración de Supabase (Panel 4)

1. Crear un proyecto en [supabase.com](https://supabase.com) y la tabla:

   ```sql
   create table consultas (
       id bigint generated by default as identity primary key,
       departamento text,
       distrito text,
       semana integer,
       datos_entrada jsonb,
       modelo text,
       prediccion text,
       probabilidad double precision,
       created_at timestamptz default now(),
       updated_at timestamptz default now()
   );
   ```

2. Copiar `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` y completar
   `SUPABASE_URL` y `SUPABASE_KEY`. **Nunca subir `secrets.toml`** (está en `.gitignore`).

3. Sin credenciales, el Panel 4 usa automáticamente **SQLite local**
   (`data/consultas_local.db`) — solo para desarrollo, no es el CRUD definitivo.

## 10. Despliegue (Streamlit Cloud)

1. Confirmar que `main` es estable, sin secretos, con rutas relativas.
2. En [share.streamlit.io](https://share.streamlit.io) apuntar a `app.py`.
3. Configurar `SUPABASE_URL` y `SUPABASE_KEY` en *Secrets* de Streamlit Cloud.
4. Verificar: acceso público, carga de las 4 páginas, gráficos, modelos y CRUD.

## 11. Modelos utilizados

| Tarea | Modelos |
|---|---|
| Clustering | K-means (`k` por codo + silueta, PCA 2D) |
| Clasificación | Random Forest y XGBoost (baseline: balanceo de clases) |
| Explicabilidad | SHAP (`TreeExplainer`) |
| Pronóstico | Media móvil (baseline) y Holt-Winters (suavizado exponencial) |

## 12. Métricas (ejecución real, test = 2024)

**Clasificación** (umbral 0.50):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.727 | 0.669 | 0.921 | 0.775 | 0.879 |
| XGBoost (mejor por F1) | 0.736 | 0.680 | 0.912 | 0.779 | 0.879 |

**Pronóstico** (serie nacional):

| Modelo | MAPE | RMSE |
|---|---|---|
| Media móvil | 44.9 % | 459.0 |
| Holt-Winters | **31.7 %** | **417.5** |

**Clustering:** k = 3, silueta ≈ 0.35 (baja / media / alta transmisión).

> Los valores exactos se regeneran con `python train.py` y quedan en
> `data/processed/metricas_*.csv`. Deben coincidir con lo mostrado en el dashboard.

## 13. Advertencias y limitaciones

- El objetivo es una etiqueta **estadística**, no una definición oficial de brote.
- Existe **cambio de distribución** train→test por la epidemia de 2023–2024.
- SHAP indica asociación, **no causalidad**.
- El intervalo del pronóstico es **aproximado** (±1.96·σ de los residuos).
- El pronóstico a nivel distrito requiere suficiente historia continua.

---

## 14. Modificaciones rápidas para la exposición

Todos los parámetros están centralizados en [`config.py`](config.py). Tras
cambiarlos, reentrenar con `python train.py` y revisar el resultado indicado.

| Parámetro | Variable (`config.py`) | Efecto | Revisar |
|---|---|---|---|
| N.º de árboles RF | `RF_N_ESTIMATORS` | Capacidad/tamaño del RF | Métricas Panel 2 |
| Profundidad RF | `RF_MAX_DEPTH` | Complejidad / overfitting / tamaño del `.joblib` | Métricas Panel 2 |
| Learning rate XGB | `XGB_LEARNING_RATE` | Velocidad de aprendizaje XGB | Métricas Panel 2 |
| N.º de clusters | `K_SELECTED` | Cantidad de grupos K-means | Panel 1 (silueta, scatter) |
| Rango de `k` | `K_MIN`, `K_MAX` | Rango del codo/silueta | Panel 1 (curva) |
| Umbral de clasificación | `CLASSIFICATION_THRESHOLD` | Precision vs recall | Panel 2 (matriz, umbral) |
| Percentil del objetivo | `TARGET_PERCENTILE` | Definición de "alta incidencia" | Distribución de clases |
| Periodo de prueba | `TEST_YEARS`, `VALIDATION_YEARS` | Años de val/test | Split temporal |
| Ventana de media móvil | `MOVING_AVERAGE_WINDOW` | Suavizado del baseline | Panel 3 (MAPE/RMSE) |
| Periodos de pronóstico | `FORECAST_PERIODS` | Semanas proyectadas | Panel 3 |

**Comando de reentrenamiento:** `python train.py` (añadir `--rebuild` solo si se
cambió el preprocesamiento o el dataset original).

## 15. Pruebas

```bash
pytest -q
```

Verifican: el dataset original no se modifica, la fecha semanal es válida y única,
los lags/medias móviles no usan futuro, el objetivo no está entre los predictores,
la clave distrito-fecha es única, y que los modelos entrenan, guardan/cargan y
producen probabilidades en [0, 1].

## 16. Integrantes

- _(completar)_

## 17. Enlace al dashboard

- _(completar tras el despliegue en Streamlit Cloud)_
