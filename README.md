# Análisis, clasificación y pronóstico del dengue en el Perú

Dashboard académico de minería de datos sobre la vigilancia epidemiológica del
dengue en el Perú (MINSA, 2000–2024). Explora los datos históricos, agrupa
distritos, clasifica el riesgo de alta incidencia para la semana siguiente,
explica las predicciones con SHAP y pronostica los próximos periodos.

> ⚠️ **Uso académico.** Las etiquetas de "alta incidencia" son estadísticas
> (percentil histórico por distrito), no una definición epidemiológica oficial de
> brote. Las explicaciones SHAP muestran asociación, no causalidad.
>
> ⚠️ **Datos 2025–2026 simulados.** La vigilancia real del MINSA llega hasta 2024;
> el periodo posterior se genera por bootstrap estacional para que el pronóstico
> muestre fechas vigentes. Está marcado como tal y **no interviene en el
> entrenamiento ni en ninguna métrica de modelo**. Ver [§3.1](#31-extensión-simulada-del-dataset-2025--mayo-2026).

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

- **Periodo real (MINSA):** 2000–2024 (semanas epidemiológicas 1–53).
- **Extensión simulada:** enero 2025 – mayo 2026 (ver §3.1). **No son datos reales.**
- **Granularidad original:** caso individual notificado (1 fila = 1 caso).
- **Unidad de análisis:** **distrito × semana epidemiológica**.

### 3.1 Extensión simulada del dataset (2025 – mayo 2026)

La vigilancia publicada por el MINSA termina en 2024. Para que el pronóstico se
muestre en fechas vigentes, el dataset maestro se extiende con **40 922 registros
generados** (553 distritos, hasta el 28/05/2026) mediante **bootstrap estacional
por distrito** ([`src/simulation.py`](src/simulation.py)):

1. Para cada distrito y cada semana objetivo `w`, se toman sus casos históricos en
   las semanas `[w−2, w+2]` (ventana circular) de los últimos 6 años.
2. Se muestrea un valor ponderando por recencia (`peso = 0.65^antigüedad`).
3. Se aplica un factor de intensidad anual (2025: 0.80, 2026: 0.90), que refleja
   el descenso posterior al pico epidémico de 2023–2024.
4. El nivel muestreado se **suaviza** (ventana 3) y se multiplica por una
   **intensidad persistente AR(1)** (ρ = 0.75): la intensidad de un brote real
   persiste varias semanas, un ruido independiente por semana no lo representa.

> ⚠️ **Estas filas NO son notificaciones reales**, son valores estimados. Llevan
> `origen = "simulado"` y `split = "simulado"`, por lo que quedan **excluidas del
> entrenamiento, de todas las métricas de clasificación y del clustering**. Solo
> alimentan la exploración temporal y el punto de partida del pronóstico.
>
> En el dashboard la distinción va donde corresponde en un gráfico de pronóstico:
> **en la leyenda** — *"Observado (MINSA)"* frente a *"Proyección estacional"*,
> con el tramo estimado punteado y en otro color, y una nota de método al pie. El
> método completo se explica en la sección *Acerca de*.
>
> Para reconstruir el proyecto **solo con datos reales**: `python train.py --sin-simulacion`.

Todos los parámetros de la simulación son editables en [`config.py`](config.py)
(`SIM_*`).

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

- **Acerca de:** presentación del proyecto, fuente de datos, definición del
  objetivo, estado de los modelos, equipo y advertencias metodológicas.
- **EDA y Clustering:** estadísticas, histogramas, boxplots, correlación,
  evolución temporal (con el tramo proyectado diferenciado en la leyenda), casos
  por departamento/distrito y outliers (1.5·IQR). El agrupamiento K-means se
  presenta **interpretado**: por qué se agrupa, cómo se construyen los grupos en
  3 pasos, una tarjeta por perfil con nombre epidemiológico y sus cifras clave, y
  cuánto aporta cada grupo al total de casos del país.
- **Modelo Predictivo:** **simulador con sliders** — se mueve cada variable del
  modelo y la probabilidad de alta incidencia y su explicación SHAP se recalculan
  al instante. Además: comparación de los 5 modelos, matriz de confusión, barrido
  de umbral, métricas por clase, efecto del balanceo y SHAP global.
- **Pronóstico:** media móvil (baseline) vs Holt-Winters, MAPE seguro y RMSE,
  proyección de 4–26 semanas con intervalo y **tabla de robustez** ante la ventana
  de evaluación.
- **Datos (CRUD):** registrar, listar, editar y eliminar consultas con
  persistencia en Supabase (o SQLite local).

### Diseño

Tema claro: superficies blancas sobre gris muy suave, acento turquesa de marca y
tarjetas con aire. La paleta categórica de los gráficos está **validada para
daltonismo** (banda de luminosidad, piso de croma, separación CVD ≥ 8 entre pares
adyacentes y distinción a visión normal ≥ 15). Tres tonos quedan por debajo de
3:1 de contraste sobre blanco, así que todo gráfico que los use lleva leyenda y
etiquetas visibles o una tabla al lado: **el color nunca carga el significado
solo**. El acento turquesa es cromo de interfaz y no codifica ningún dato.
Definiciones en [`src/visualizations.py`](src/visualizations.py) (datos) y
[`src/ui.py`](src/ui.py) (interfaz).

## 5. Estructura del proyecto

```
mineria-proyecto-dengue-peru/
├── app.py                  # Punto de entrada + navegación por pestañas
├── config.py               # Parámetros centralizados (editables en vivo)
├── train.py                # Genera TODOS los artefactos
├── requirements.txt
├── data/
│   ├── raw/                # dataset original (no versionado)
│   └── processed/          # dengue_semanal.csv, clusters, métricas
├── models/                 # *.joblib (modelos y preprocesador)
├── views/                  # paneles del dashboard (Acerca de + 4 secciones)
├── src/                    # preprocessing, simulation, modeling, forecasting,
│                           # clustering, visualizations, ui, database, loaders
├── notebooks/              # 01–04 exploración y validación
└── tests/                  # pruebas (pytest)
```

**Columnas añadidas al dataset maestro:** `origen` (`real` | `simulado`) y el valor
`simulado` en `split`.

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
python train.py                   # usa data/processed/dengue_semanal.csv si existe
python train.py --rebuild         # reconstruye el maestro desde data/raw/ (necesita el CSV original)
python train.py --sin-simulacion  # solo datos reales del MINSA, sin la extensión 2025–2026
```

Genera: `dengue_semanal.csv`, `distritos_clusters.csv`,
`metricas_clasificacion.csv`, `metricas_balanceo.csv`, `metricas_pronostico.csv`
y los `.joblib` en `models/`.

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
   (`data/consultas_local.db`).

## 10. Despliegue (Streamlit Cloud)

1. Confirmar que `main` es estable, sin secretos, con rutas relativas.
2. En [share.streamlit.io](https://share.streamlit.io) apuntar a `app.py`.
3. Configurar `SUPABASE_URL` y `SUPABASE_KEY` en *Secrets* de Streamlit Cloud.
4. Verificar: acceso público, carga de las 4 páginas, gráficos, modelos y CRUD.

## 11. Modelos utilizados

| Tarea | Modelos |
|---|---|
| Clustering | K-means (`k` por codo + silueta, PCA 2D) |
| Clasificación | **5 modelos**: Random Forest, XGBoost, Gradient Boosting, Regresión Logística y Árbol de Decisión (con balanceo de clases) |
| Explicabilidad | SHAP: `TreeExplainer` en los modelos de árbol y `LinearExplainer` en la Regresión Logística, de modo que **el mejor modelo por F1 también queda explicado** |
| Pronóstico | Media móvil (baseline) y Holt-Winters (suavizado exponencial) |
| Simulación de datos | Bootstrap estacional por distrito con recencia, factor de intensidad anual y persistencia AR(1) |

## 12. Métricas (ejecución real)

**Clasificación** — umbral 0.50, test = 2024, **solo datos reales** (la extensión
simulada está excluida). Ordenado por F1:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Regresión Logística (mejor por F1)** | 0.770 | 0.756 | 0.806 | **0.780** | 0.855 |
| XGBoost | 0.729 | 0.671 | 0.914 | 0.774 | 0.876 |
| Gradient Boosting | 0.724 | 0.665 | 0.920 | 0.772 | 0.876 |
| Random Forest | 0.720 | 0.660 | 0.923 | 0.769 | 0.875 |
| Árbol de Decisión | 0.716 | 0.660 | 0.908 | 0.764 | 0.867 |

> Los modelos de árbol logran mayor **recall** de la clase minoritaria (~0.92),
> útil si el costo de un falso negativo es alto; la Regresión Logística ofrece el
> mejor **equilibrio** (F1). El mejor modelo se elige automáticamente por F1.

**Efecto del balanceo** (recall de la clase 1 en test): Random Forest pasa de
0.618 a 0.923 y XGBoost de 0.636 a 0.914 al aplicar `class_weight` /
`scale_pos_weight`.

**Pronóstico** (serie nacional). Qué modelo gana **depende de la ventana de
evaluación**, así que se publica la tabla completa en lugar de una sola cifra:

| Ventana de prueba | MAPE media móvil | RMSE media móvil | MAPE Holt-Winters | RMSE Holt-Winters | Elegido |
|---|---|---|---|---|---|
| 8 semanas | 19.6 % | 1 730.9 | 96.3 % | 8 400.1 | media móvil |
| **13 semanas (por defecto)** | 43.0 % | 4 205.4 | **26.6 %** | **2 597.2** | **Holt-Winters** |
| 26 semanas | 65.0 % | 5 706.7 | 58.4 % | 5 296.6 | Holt-Winters |

> La ventana por defecto es de **13 semanas (un trimestre epidemiológico)**. Con 8
> semanas o menos la media móvil gana por construcción: pronostica una constante y
> en tan pocas semanas eso se parece al promedio real, mientras que un componente
> estacional de 52 semanas no alcanza a expresarse. El Panel 3 muestra esta tabla
> y permite mover la ventana en vivo.

**Clustering:** k = 3, silueta ≈ 0.353, PCA 2D explica 68.8 % de la varianza.

| Perfil | Distritos | Casos/semana | Pico histórico | Semanas activas | % casos del país |
|---|---|---|---|---|---|
| Transmisión esporádica | 342 | 0.5 | 32 | 9 % | 14.1 % |
| Transmisión estacional | 195 | 3.4 | 96 | 34 % | 39.8 % |
| Transmisión alta y sostenida | 34 | 16.0 | 663 | 50 % | 46.1 % |

> El hallazgo accionable: **34 distritos (6 % del país) concentran el 46 % de
> todos los casos notificados**. Ahí rinde más cada sol invertido en control del
> vector; los grupos de menor intensidad necesitan vigilancia para detectar un
> brote inusual, no inversión permanente.

> Los valores exactos se regeneran con `python train.py` y quedan en
> `data/processed/metricas_*.csv`. Deben coincidir con lo mostrado en el dashboard.

## 13. Advertencias y limitaciones

- **Los registros de 2025 y 2026 son simulados**, no vigilancia real del MINSA
  (ver §3.1). No entrenan ni evalúan ningún modelo, pero sí son el punto de
  partida del pronóstico mostrado.
- El objetivo es una etiqueta **estadística**, no una definición oficial de brote.
- Existe **cambio de distribución** train→test por la epidemia de 2023–2024: la
  clase positiva pasa de ~10 % en entrenamiento a ~50 % en prueba.
- SHAP indica asociación, **no causalidad**.
- El intervalo del pronóstico es **aproximado** (±1.96·σ de los residuos), no un
  intervalo de predicción exacto.
- Qué modelo de pronóstico gana **depende de la ventana de evaluación** (ver §12).
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
| Ventana de evaluación | `FORECAST_EVAL_PERIODS` | Qué modelo de pronóstico gana | Panel 3 (tabla de robustez) |
| Horizonte simulado | `SIM_END_ANO`, `SIM_END_SEMANA` | Hasta cuándo llega el dataset | Todos los paneles |
| Intensidad simulada | `SIM_INTENSIDAD` | Magnitud del brote 2025/2026 | Panel 3 (nivel de la serie) |
| Persistencia simulada | `SIM_PERSISTENCIA` | Suavidad de la curva generada | Panel 1 (evolución temporal) |
| Desactivar simulación | `SIMULAR_EXTENSION = False` | Vuelve al dataset solo real | Periodo mostrado |

**Comando de reentrenamiento:** `python train.py` (añadir `--rebuild` solo si se
cambió el preprocesamiento o el dataset original).

## 15. Pruebas

```bash
pytest -q
```

**30 pruebas** que verifican:

- *Preprocesamiento:* el dataset original no se modifica, la fecha semanal es
  válida y única, los lags/medias móviles no usan futuro, el objetivo no está
  entre los predictores y la clave distrito-fecha es única.
- *Modelos:* entrenan, se guardan/cargan y producen probabilidades en [0, 1].
- *Simulación:* la extensión no altera ninguna fila real, las filas generadas
  quedan fuera de train/val/test, el conjunto de prueba sigue siendo el último
  año real, la simulación es reproducible, conserva la estacionalidad y no
  reactiva distritos que dejaron de reportar.
- *Calendario del pronóstico:* fecha ↔ (año, semana) son inversas exactas, la
  serie no pierde semanas en años que no empiezan en lunes (regresión: el
  `asfreq("W-MON")` anterior anulaba 2025 y 2026) y el pronóstico arranca después
  del último dato con intervalos coherentes.

## 16. Integrantes

- **Herrera Gómez, Gerardo Jesús**
- **Mejía Carrasco, Marlo Gabriel**
- **Ortiz Herrera, Fabrizio Peter**
- **Sosa Lupuche, Carlos Manuel**

## 17. Enlace al dashboard

- _(completar tras el despliegue en Streamlit Cloud)_
