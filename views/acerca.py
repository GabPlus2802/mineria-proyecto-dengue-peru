"""Acerca de: presentacion del proyecto, fuente de datos, modelos y equipo."""

import streamlit as st

import config
from src import loaders, modeling, ui, visualizations as viz

REPO = "https://github.com/GabPlus2802/mineria-proyecto-dengue-peru"

ui.hero(
    "🦟 Dengue en el Peru — analisis, clasificacion y pronostico",
    "Tablero analitico sobre la vigilancia epidemiologica del dengue del MINSA. "
    "Explora 25 anios de datos, agrupa distritos por su comportamiento, clasifica "
    "el riesgo de la semana siguiente y proyecta los proximos periodos.",
    badges=["Mineria de Datos", "MINSA 2000-2024", "Distrito x semana",
            "5 modelos + SHAP", "Streamlit"],
)

listo = loaders.artefactos_listos()
if not listo:
    st.error(
        "Faltan artefactos generados. Ejecuta primero el entrenamiento:\n\n"
        "```bash\npython train.py --rebuild\n```"
    )
    st.stop()

df = loaders.load_master()
df_real = loaders.solo_real(df)
res_sim = loaders.resumen_simulacion()

ui.banner_simulacion(res_sim)

# ---------------------------------------------------------------------------
# KPIs de contexto (sobre el dato real)
# ---------------------------------------------------------------------------
ui.kpi_row([
    {"label": "Registros distrito-semana", "value": f"{len(df_real):,}", "icon": "📅",
     "accent": viz.SERIE[0]},
    {"label": "Casos notificados", "value": f"{int(df_real['casos'].sum()):,}",
     "icon": "🦟", "accent": viz.SERIE[7]},
    {"label": "Distritos", "value": f"{df_real['ubigeo'].nunique():,}", "icon": "📍",
     "accent": viz.SERIE[2]},
    {"label": "Departamentos", "value": df_real["departamento"].nunique(), "icon": "🗺️",
     "accent": viz.SERIE[3]},
    {"label": "Periodo real", "value": f"{int(df_real['ano'].min())}–{int(df_real['ano'].max())}",
     "icon": "⏱️", "accent": viz.SERIE[6]},
])
st.caption("Cifras calculadas sobre la vigilancia real del MINSA, sin la extension simulada.")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    ui.section("El problema")
    st.markdown(
        """
El dengue, transmitido por el mosquito *Aedes aegypti*, es una de las principales
emergencias de salud publica en el Peru, con una expansion notable en 2023-2024.
Anticipar donde subira la transmision permite dirigir la prevencion antes de que
el brote crezca.

Este tablero convierte 1.03 millones de notificaciones individuales en la unidad
de analisis **distrito x semana epidemiologica**, y sobre ella **explora**,
**agrupa**, **clasifica**, **explica** y **pronostica**.
        """
    )

    ui.section("Que hace cada seccion")
    p = st.columns(2, gap="medium")
    with p[0]:
        with st.container(border=True):
            st.markdown("#### 📊 EDA y Clustering")
            st.caption("Estadisticas descriptivas, outliers por la regla 1.5·IQR y "
                       "agrupamiento K-means de distritos con PCA en 2D.")
        with st.container(border=True):
            st.markdown("#### 📈 Pronostico")
            st.caption("Media movil frente a Holt-Winters, MAPE seguro y RMSE, "
                       "proyeccion futura con intervalo y prueba de robustez.")
    with p[1]:
        with st.container(border=True):
            st.markdown("#### 🤖 Modelo Predictivo")
            st.caption("Simulador con sliders: mueve cada variable y observa el riesgo "
                       "y su explicacion SHAP al instante. Compara 5 modelos.")
        with st.container(border=True):
            st.markdown("#### 🗂️ Datos (CRUD)")
            st.caption("Registrar, listar, editar y eliminar consultas con "
                       "persistencia en Supabase o SQLite local.")
    st.caption("Navega con las pestanas de la parte superior.")

    ui.section("Como se define el objetivo")
    st.markdown(
        f"""
La variable objetivo es **alta incidencia en la semana siguiente**: vale 1 si los
casos de la proxima semana superan el **percentil {config.TARGET_PERCENTILE}
historico de ese mismo distrito**.

Dos decisiones importantes para que la evaluacion sea honesta:

- El umbral se calcula **solo con el periodo de entrenamiento** (hasta 2022), nunca
  con validacion ni prueba: sin fuga de informacion.
- Todas las medias moviles y desviaciones usan `shift(1)`, es decir **solo semanas
  anteriores** a la que se predice.
        """
    )

with col2:
    ui.section("Fuente de datos")
    with st.container(border=True):
        st.markdown(
            f"""
**MINSA — Vigilancia epidemiologica del dengue**
Plataforma Nacional de Datos Abiertos del Peru.

- Periodo publicado: **2000-2024**
- Granularidad original: **caso individual**
- Unidad de analisis: **distrito × semana**
- Registros crudos: **1 029 421**
- Ubigeos: **662**
            """
        )

    ui.section("Estado de los modelos")
    modelos = loaders.load_models()
    meta = modelos.get("meta") or {}
    with st.container(border=True):
        clasif = loaders.load_clasificadores(modelos)
        st.markdown(f"**{len(clasif)} clasificadores** entrenados:")
        for nombre in clasif:
            st.write("✅ " + modeling.MODEL_LABELS.get(nombre, nombre))
        for extra in ["kmeans", "preprocessor"]:
            ok = modelos.get(extra) is not None
            st.write(("✅ " if ok else "❌ ") + extra)
        if meta:
            mejor = meta.get("mejor_modelo", "—")
            st.caption(f"Mejor clasificador por F1: **{modeling.MODEL_LABELS.get(mejor, mejor)}** "
                       f"(umbral {meta.get('threshold', config.CLASSIFICATION_THRESHOLD)})")

    ui.section("Equipo de trabajo")
    with st.container(border=True):
        st.markdown(
            """
- **Herrera Gómez, Gerardo Jesús**
- **Mejía Carrasco, Marlo Gabriel**
- **Ortiz Herrera, Fabrizio Peter**
- **Sosa Lupuche, Carlos Manuel**
            """
        )

    ui.section("Repositorio")
    with st.container(border=True):
        st.markdown(f"🔗 [github.com/GabPlus2802/mineria-proyecto-dengue-peru]({REPO})")
        st.caption("Codigo, notebooks, documentacion tecnica y manual de usuario.")

# ---------------------------------------------------------------------------
# Advertencias metodologicas
# ---------------------------------------------------------------------------
ui.section("Advertencias metodologicas", tag="leelas antes de interpretar")
a, b, c = st.columns(3, gap="medium")
with a:
    with st.container(border=True):
        st.markdown("**No es una definicion oficial de brote**")
        st.caption("La etiqueta de *alta incidencia* es estadistica —un percentil "
                   "historico por distrito— y no corresponde a la definicion "
                   "epidemiologica oficial del MINSA.")
with b:
    with st.container(border=True):
        st.markdown("**SHAP no prueba causalidad**")
        st.caption("Las explicaciones muestran que variables se asocian a la "
                   "probabilidad predicha por el modelo, no que la causen.")
with c:
    with st.container(border=True):
        st.markdown("**Cambio de distribucion 2023-2024**")
        st.caption("La clase positiva pasa de ~10 % en entrenamiento a ~50 % en "
                   "prueba por la epidemia real de esos anios. Las metricas deben "
                   "leerse con ese contexto.")

if res_sim.get("tiene_simulacion"):
    ui.section("Sobre la extension simulada", tag="transparencia")
    with st.container(border=True):
        st.markdown(
            f"""
La vigilancia publicada del MINSA termina en **{res_sim['ultimo_ano_real']}**. Para que
el pronostico se muestre en fechas vigentes, el dataset se extiende hasta
**{res_sim['hasta']:%d/%m/%Y}** con **{res_sim['filas_simuladas']:,} registros generados**
mediante *bootstrap estacional* por distrito: para cada semana objetivo se muestrea
el historico del propio distrito en esa misma epoca del anio, ponderando los anios
recientes, y se aplica un factor de intensidad anual mas una intensidad persistente
AR(1) que imita la continuidad de un brote real.

**Estas filas no son datos reales.** Van marcadas con `origen = "simulado"` y quedan
excluidas del entrenamiento, de las metricas de clasificacion y del clustering: solo
alimentan la exploracion temporal y el pronostico. Puedes regenerar el proyecto sin
ellas con `python train.py --sin-simulacion`.
            """
        )
