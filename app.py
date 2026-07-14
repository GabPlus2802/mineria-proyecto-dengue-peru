"""Dashboard de analisis, clasificacion y pronostico del dengue en el Peru.

Pagina principal. Las cuatro secciones estan en la carpeta pages/.
Ejecutar con:  streamlit run app.py
"""

import streamlit as st

import config
from src import loaders

st.set_page_config(
    page_title="Dengue Peru | Mineria de Datos",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🦟 Analisis, clasificacion y pronostico del dengue en el Peru")
st.caption("Proyecto academico de Mineria de Datos — datos de vigilancia MINSA 2000-2024")

listo = loaders.artefactos_listos()
if not listo:
    st.error(
        "Faltan artefactos generados. Ejecuta primero el entrenamiento:\n\n"
        "```bash\npython train.py --rebuild\n```"
    )
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Problema")
    st.markdown(
        """
El dengue es una enfermedad transmitida por el mosquito *Aedes aegypti* y una de
las principales emergencias de salud publica en el Peru. Este tablero permite
**explorar** los datos historicos, **agrupar** distritos segun su comportamiento,
**clasificar** el riesgo de alta incidencia para la semana siguiente,
**explicar** las predicciones y **pronosticar** los proximos periodos.
        """
    )

    st.subheader("Los cuatro paneles")
    st.markdown(
        """
1. **EDA y Clustering** — estadisticas, outliers (1.5·IQR) y agrupamiento K-means de distritos.
2. **Modelo Predictivo** — Random Forest vs XGBoost, metricas, umbral, SHAP y formulario de prediccion.
3. **Pronostico** — media movil vs Holt-Winters, MAPE y RMSE, proyeccion de 4+ semanas.
4. **CRUD** — registrar, listar, editar y eliminar consultas (Supabase o modo local).
        """
    )
    st.info("Usa el menu lateral para navegar entre los paneles.", icon="👈")

with col2:
    st.subheader("Fuente de datos")
    st.markdown(
        """
**MINSA — Vigilancia epidemiologica del dengue**
Plataforma Nacional de Datos Abiertos del Peru.

- Periodo: **2000–2024** (semanas epidemiologicas)
- Granularidad original: **caso individual notificado**
- Unidad de analisis: **distrito × semana**
        """
    )

    st.subheader("Estado de los modelos")
    modelos = loaders.load_models()
    meta = modelos.get("meta") or {}
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
