"""Dashboard de analisis, clasificacion y pronostico del dengue en el Peru.

Pagina principal. Las cuatro secciones estan en la carpeta pages/.
Ejecutar con:  streamlit run app.py
"""

import streamlit as st

import config
from src import loaders, ui

ui.setup_page("Dengue Peru | Mineria de Datos", "🦟")

ui.hero(
    "🦟 Dengue en el Peru — Analisis, clasificacion y pronostico",
    "Tablero analitico sobre la vigilancia epidemiologica del MINSA (2000-2024).",
    badges=["25 anios de datos", "Distrito x semana", "4 paneles interactivos",
            "Mineria de Datos"],
)

listo = loaders.artefactos_listos()
if not listo:
    st.error(
        "Faltan artefactos generados. Ejecuta primero el entrenamiento:\n\n"
        "```bash\npython train.py --rebuild\n```"
    )
    st.stop()

# KPIs de contexto
df = loaders.load_master()
ui.kpi_row([
    {"label": "Registros distrito-semana", "value": f"{len(df):,}", "icon": "📅",
     "accent": "#2a78d6"},
    {"label": "Casos totales", "value": f"{int(df['casos'].sum()):,}", "icon": "🦟",
     "accent": "#e34948"},
    {"label": "Distritos", "value": df["ubigeo"].nunique(), "icon": "📍",
     "accent": "#1baf7a"},
    {"label": "Departamentos", "value": df["departamento"].nunique(), "icon": "🗺️",
     "accent": "#eda100"},
    {"label": "Periodo", "value": f"{int(df['ano'].min())}–{int(df['ano'].max())}",
     "icon": "⏱️", "accent": "#4a3aa7"},
])

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    ui.section("El problema")
    st.markdown(
        """
El dengue, transmitido por el mosquito *Aedes aegypti*, es una de las principales
emergencias de salud publica en el Peru, con una expansion notable en 2023-2024.
Este tablero permite **explorar** los datos historicos, **agrupar** distritos segun
su comportamiento, **clasificar** el riesgo de alta incidencia para la semana
siguiente, **explicar** las predicciones y **pronosticar** los proximos periodos.
        """
    )

    ui.section("Los cuatro paneles")
    p = st.columns(2, gap="medium")
    with p[0]:
        with st.container(border=True):
            st.markdown("#### 📊 1 · EDA y Clustering")
            st.caption("Estadisticas, outliers (1.5·IQR) y agrupamiento K-means de distritos.")
        with st.container(border=True):
            st.markdown("#### 📈 3 · Pronostico")
            st.caption("Media movil vs Holt-Winters, MAPE y RMSE, proyeccion de 4+ semanas.")
    with p[1]:
        with st.container(border=True):
            st.markdown("#### 🤖 2 · Modelo Predictivo")
            st.caption("Random Forest vs XGBoost, umbral, SHAP y formulario de prediccion.")
        with st.container(border=True):
            st.markdown("#### 🗂️ 4 · CRUD de consultas")
            st.caption("Registrar, listar, editar y eliminar consultas (Supabase o local).")
    st.info("Usa el menu lateral para navegar entre los paneles.", icon="👈")

with col2:
    ui.section("Fuente de datos")
    with st.container(border=True):
        st.markdown(
            """
**MINSA — Vigilancia epidemiologica del dengue**
Plataforma Nacional de Datos Abiertos del Peru.

- Periodo: **2000-2024** (semanas epidemiologicas)
- Granularidad original: **caso individual notificado**
- Unidad de analisis: **distrito × semana**
            """
        )

    ui.section("Estado de los modelos")
    modelos = loaders.load_models()
    meta = modelos.get("meta") or {}
    with st.container(border=True):
        for nombre in ["random_forest", "xgboost", "kmeans", "preprocessor"]:
            ok = modelos.get(nombre) is not None
            st.write(("✅ " if ok else "❌ ") + nombre)
        if meta:
            st.caption(f"Mejor clasificador: **{meta.get('mejor_modelo', '—')}** "
                       f"(umbral {meta.get('threshold', config.CLASSIFICATION_THRESHOLD)})")

st.divider()
st.markdown(
    "🔗 Repositorio: "
    "[github.com/GabPlus2802/mineria-proyecto-dengue-peru]"
    "(https://github.com/GabPlus2802/mineria-proyecto-dengue-peru)"
)
st.warning(
    "**Uso academico.** Las etiquetas de 'alta incidencia' son estadisticas "
    "(percentil historico por distrito), no una definicion epidemiologica oficial "
    "de brote. Las explicaciones SHAP muestran asociacion, no causalidad.",
    icon="⚠️",
)
