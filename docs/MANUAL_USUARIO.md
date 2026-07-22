# Manual de usuario — Dashboard de dengue Perú

Guía paso a paso para **instalar, ejecutar y usar** el sistema. No requiere
conocimientos de programación para operar el dashboard una vez instalado.

Para entender cómo funciona por dentro, ver
[DOCUMENTACION_TECNICA.md](DOCUMENTACION_TECNICA.md).

---

## Parte A — Instalación y arranque

### Requisitos
- **Python 3.11 o superior** (probado en 3.13).
- Windows, macOS o Linux.

### Paso 1: Abrir el proyecto
Abrir una terminal (PowerShell en Windows) dentro de la carpeta del proyecto
`mineria-proyecto-dengue-peru`.

### Paso 2: Crear y activar el entorno virtual
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux
```
Debe aparecer `(.venv)` al inicio de la línea de la terminal.

### Paso 3: Instalar dependencias
```powershell
pip install -r requirements.txt
```

### Paso 4 (opcional): Regenerar modelos
El repositorio ya incluye los modelos entrenados. Solo si quieres recrearlos:
```powershell
python train.py                   # usa el dataset procesado existente
python train.py --rebuild         # reconstruye TODO desde el CSV crudo (data/raw/)
python train.py --sin-simulacion  # solo datos reales, sin la extensión 2025-2026
```

> ⚠️ **Importante — datos simulados.** Los datos reales del MINSA llegan hasta 2024.
> Para que el pronóstico muestre fechas vigentes, el sistema añade registros de
> enero 2025 a mayo 2026 **generados por simulación**. El dashboard lo advierte con
> un aviso ámbar en cada panel afectado y el tramo simulado aparece punteado en los
> gráficos. **Esos registros no entrenan ni evalúan ningún modelo:** todas las
> métricas de clasificación y el clustering usan únicamente datos reales.

### Paso 5: Ejecutar el dashboard
```powershell
streamlit run app.py
```
Se abrirá automáticamente en el navegador (normalmente `http://localhost:8501`).
Para cerrarlo: `Ctrl + C` en la terminal.

---

## Parte B — Uso del dashboard

Al abrir verás la **portada** con la descripción del problema, la fuente de datos y
el estado de los modelos (✅/❌). En la **barra lateral izquierda** se elige el panel.

---

### Panel 1 — 📊 EDA y Clustering

**Para qué sirve:** explorar los datos y ver los grupos de distritos.

1. En la barra lateral, elige un **Departamento** (o "Todos") y el **rango de años**.
2. **Resumen:** tarjetas con registros, casos totales, número de deptos/distritos y periodo.
3. **Pestañas de análisis:**
   - *Evolución temporal:* casos por semana.
   - *Distribución:* histograma y boxplot + estadísticas.
   - *Por ubicación:* casos por departamento y por distrito.
   - *Correlación:* relación entre variables.
4. **Outliers:** muestra el límite superior (1.5·IQR) y las semanas atípicas
   (brotes reales, que se conservan).
5. **Clustering:** la curva del **codo y silueta**, el **scatter PCA** de los grupos
   y una tabla con el **perfil de cada cluster** (baja / media / alta transmisión).

> 💡 Todos los gráficos son interactivos: pasa el mouse para ver detalles, haz zoom
> o descarga la imagen con el ícono de la cámara.

---

### Panel 2 — 🤖 Modelo Predictivo

**Para qué sirve:** comparar modelos, entender qué influye y hacer una predicción.

El panel tiene tres pestañas.

#### 🎛️ Simulador de predicción (la principal)

1. Elige **Departamento**, **Distrito**, **Modelo** y el **umbral de decisión**.
2. Los sliders se precargan con los **valores reales de la última semana** de ese
   distrito. Cada uno tiene un **−** a la izquierda y un **+** a la derecha:
   - Casos de esta semana y de hace 1, 2 y 4 semanas
   - Promedio de las 4 y de las 8 semanas previas
   - Variabilidad de las 4 semanas previas
   - Crecimiento respecto a la semana previa
   - Semana epidemiológica (de ella se derivan `semana_sen` y `semana_cos`)
3. **Todo se recalcula al instante** al mover cualquier slider: el medidor de
   probabilidad, el estado (RIESGO ALTO / BAJO) y la explicación SHAP.
4. Dos botones te ayudan:
   - **↺ Volver a los valores reales** — deshace tus cambios.
   - **⚙️ Sincronizar derivadas** — recalcula promedios, variabilidad y crecimiento
     a partir de los lags que pusiste, para que el escenario sea coherente.
5. En **Por qué esta predicción** verás qué variable empujó el resultado: rojo hacia
   ALTA incidencia, azul hacia baja, con el valor numérico en cada barra.
6. El escenario queda guardado para registrarlo en **Datos (CRUD)**.

#### 🏁 Comparación de modelos

Tabla de los 5 modelos (accuracy, precision, recall, F1, ROC-AUC, TP/TN/FP/FN),
matriz de confusión, curva de umbral, métricas por clase y efecto del balanceo.

#### 🔍 Explicabilidad global

Qué variables pesan más en el conjunto de prueba completo, con *summary plot* e
importancia media.

> 💡 Puedes mover libremente cada variable para responder preguntas del tipo *"¿y si
> los casos se duplicaran esta semana?"*. Si quieres volver a un escenario realista,
> usa **Sincronizar derivadas** o **Volver a los valores reales**.

---

### Panel 3 — 📈 Pronóstico

**Para qué sirve:** proyectar los casos de las próximas semanas.

1. En la barra lateral elige el **nivel de agregación**:
   - *Nacional* (recomendado, serie más estable),
   - *Departamento* (elige cuál),
   - *Distrito* (elige departamento y distrito con suficiente historia).
2. Ajusta las **semanas futuras a pronosticar** (4 a 26) y las **semanas
   reservadas para evaluar** (8 a 26).
3. Verás:
   - Tarjetas con **MAPE** y **RMSE** de la media móvil y de Holt-Winters.
   - El **modelo elegido** (menor RMSE).
   - Un **gráfico** con histórico, valores reales, estimados, pronóstico futuro y su
     intervalo. Una línea vertical marca **dónde terminan los datos reales**.
   - Una **tabla** con el pronóstico semana a semana.
   - Una **tabla de robustez**: qué modelo gana con distintas ventanas de
     evaluación. Con ventanas cortas la media móvil puede ganar por construcción,
     porque pronostica una constante.

> ⚠️ Si eliges un distrito con poca historia, el sistema te pedirá usar un nivel más
> agregado.

---

### 🗂️ Datos (CRUD)

**Para qué sirve:** guardar, revisar, editar y borrar consultas/predicciones.

En la parte superior se indica el modo:
- ✅ **Supabase** (persistencia en producción), o
- ⚠️ **Local (SQLite)** — modo de desarrollo, guarda en `data/consultas_local.db`.

**Crear:**
1. Completa Departamento, Distrito, Semana, Modelo, Predicción y Probabilidad.
   (Si vienes del Panel 2, los campos aparecen **prellenados**.)
2. Pulsa **Crear registro**.

**Listar:** la tabla muestra todas las consultas con fecha/hora automáticas.

**Editar / Eliminar:**
1. Elige el **ID** en el desplegable.
2. Modifica los campos y pulsa **Guardar cambios**, o pulsa **Eliminar registro**.

---

## Parte C — Preguntas frecuentes

**El dashboard dice "Faltan artefactos".**
Ejecuta `python train.py --rebuild` (necesitas el CSV original en `data/raw/`) o
`python train.py` si el dataset procesado ya existe.

**¿Qué significa "alta incidencia"?**
Que los casos de la semana siguiente superarían el percentil 75 histórico del
distrito. Es una etiqueta **estadística**, no un diagnóstico oficial de brote.

**¿Por qué el modelo acierta "distinto" en 2024?**
Porque 2023–2024 fue una epidemia real y casi todas las semanas superan el umbral
histórico. El modelo mantiene un **recall alto** (detecta bien la actividad).

**¿Se pierde lo que guardé en el CRUD si cierro la app?**
No. En modo local se guarda en el archivo `data/consultas_local.db`; en Supabase, en
la nube.

**¿Cómo cambio un parámetro (n.º de árboles, clusters, umbral…)?**
Edita `config.py`, vuelve a ejecutar `python train.py` y recarga el dashboard.
Ver la tabla *Modificaciones rápidas* en el `README.md`.

---

## Parte D — Detener y reiniciar

- **Detener:** `Ctrl + C` en la terminal donde corre Streamlit.
- **Reiniciar:** vuelve a ejecutar `streamlit run app.py`.
- **Salir del entorno virtual:** escribe `deactivate`.
