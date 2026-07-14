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
python train.py            # usa el dataset procesado existente
python train.py --rebuild  # reconstruye TODO desde el CSV crudo (data/raw/)
```

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

1. **Umbral de decisión:** mueve el deslizador para ver cómo cambian las métricas.
2. **Tabla de comparación:** Random Forest vs XGBoost (accuracy, precision, recall,
   F1, ROC-AUC, TP/TN/FP/FN). Debajo se indica el **mejor modelo** por F1.
3. **Matriz de confusión** y **curva de umbral** (precision/recall/F1).
4. **SHAP:**
   - *Explicación global:* qué variables pesan más en general.
   - *Explicación local:* elige un índice para ver por qué se predijo un caso.
5. **Predicción en vivo:**
   1. Elige **Departamento** y **Distrito**.
   2. Ajusta (o deja los valores por defecto) los **casos de la semana actual** y la
      **semana epidemiológica**.
   3. Elige el **modelo** y pulsa **Predecir**.
   4. Verás la **clase predicha** (ALTA / Baja incidencia), la **probabilidad**, el
      **modelo usado** y una **explicación SHAP** de esa predicción.
   5. La predicción queda guardada en memoria para registrarla en el Panel 4.

> 💡 No necesitas calcular variables técnicas (lags, medias móviles): el sistema las
> calcula solo a partir del historial del distrito.

---

### Panel 3 — 📈 Pronóstico

**Para qué sirve:** proyectar los casos de las próximas semanas.

1. En la barra lateral elige el **nivel de agregación**:
   - *Nacional* (recomendado, serie más estable),
   - *Departamento* (elige cuál),
   - *Distrito* (elige departamento y distrito con suficiente historia).
2. Ajusta los **periodos futuros a pronosticar** (4 a 12).
3. Verás:
   - Tarjetas con **MAPE** y **RMSE** de la media móvil y de Holt-Winters.
   - El **modelo elegido** (menor RMSE).
   - Un **gráfico** con histórico, valores reales, estimados, pronóstico futuro y su
     intervalo.
   - Una **tabla** con el pronóstico y su intervalo.

> ⚠️ Si eliges un distrito con poca historia, el sistema te pedirá usar un nivel más
> agregado.

---

### Panel 4 — 🗂️ CRUD de consultas

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
