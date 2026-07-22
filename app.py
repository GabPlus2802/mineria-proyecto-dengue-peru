"""Dashboard de analisis, clasificacion y pronostico del dengue en el Peru.

Punto de entrada. Define la navegacion por pestanas superiores; cada panel
vive en la carpeta views/.
Ejecutar con:  streamlit run app.py
"""

import streamlit as st

from src import ui

st.set_page_config(
    page_title="Dengue Peru | Mineria de Datos",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.apply_base_style()

# --- Pestanas superiores (una por seccion) -------------------------------
paginas = [
    st.Page("views/acerca.py", title="Acerca de", icon="🦟", default=True),
    st.Page("views/eda_clustering.py", title="EDA & Clustering", icon="📊"),
    st.Page("views/modelo_predictivo.py", title="Modelo Predictivo", icon="🤖"),
    st.Page("views/pronostico.py", title="Pronostico", icon="📈"),
    st.Page("views/crud.py", title="Datos (CRUD)", icon="🗂️"),
]

st.navigation(paginas, position="top").run()
