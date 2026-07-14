"""Panel 1: Analisis exploratorio (EDA), outliers y clustering de distritos."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src import clustering, loaders, ui, visualizations as viz

ui.setup_page("EDA y Clustering", "📊")

if not loaders.artefactos_listos():
    st.error("Faltan artefactos. Ejecuta `python train.py --rebuild`.")
    st.stop()

ui.hero("📊 Panel 1 — EDA y Clustering",
        "Exploracion de los datos, deteccion de outliers y agrupamiento de distritos.")

df = loaders.load_master()
clusters = loaders.load_clusters()

# ---------------------------------------------------------------------------
# Filtros interactivos
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")
deps = ["(Todos)"] + sorted(df["departamento"].unique().tolist())
dep_sel = st.sidebar.selectbox("Departamento", deps)
anos = sorted(df["ano"].unique().tolist())
rango = st.sidebar.select_slider("Rango de anios", options=anos,
                                 value=(anos[0], anos[-1]))

f = df[(df["ano"] >= rango[0]) & (df["ano"] <= rango[1])].copy()
if dep_sel != "(Todos)":
    f = f[f["departamento"] == dep_sel]

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
ui.section("Resumen", "Indicadores segun los filtros seleccionados.")
ui.kpi_row([
    {"label": "Registros", "value": f"{len(f):,}", "icon": "📅", "accent": "#2a78d6"},
    {"label": "Total de casos", "value": f"{int(f['casos'].sum()):,}", "icon": "🦟",
     "accent": "#e34948"},
    {"label": "Departamentos", "value": f["departamento"].nunique(), "icon": "🗺️",
     "accent": "#eda100"},
    {"label": "Distritos", "value": f["ubigeo"].nunique(), "icon": "📍", "accent": "#1baf7a"},
    {"label": "Periodo", "value": f"{rango[0]}–{rango[1]}", "icon": "⏱️", "accent": "#4a3aa7"},
])

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
ui.section("Analisis exploratorio")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Evolucion temporal", "Distribucion", "Por ubicacion", "Correlacion"]
)

with tab1:
    serie = f.groupby("fecha")["casos"].sum()
    st.plotly_chart(viz.evolucion_temporal(serie), width='stretch')
    st.caption("Casos semanales agregados segun los filtros seleccionados.")

with tab2:
    col1, col2 = st.columns(2)
    positivos = f[f["casos"] > 0]
    with col1:
        st.plotly_chart(viz.histograma(positivos, "casos",
                        "Histograma de casos (>0)"), width='stretch')
    with col2:
        st.plotly_chart(viz.boxplot(positivos, "casos",
                        "Boxplot de casos (>0)"), width='stretch')
    st.markdown("**Estadisticas descriptivas de `casos`**")
    st.dataframe(f["casos"].describe().to_frame().T, width='stretch')

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.barras_por_categoria(f, "departamento", "casos", top=15),
                        width='stretch')
    with col2:
        top_dist = (f.groupby("distrito")["casos"].sum().sort_values(ascending=False)
                    .head(15).reset_index())
        st.plotly_chart(px.bar(top_dist, x="distrito", y="casos",
                        template=viz.PLANTILLA, title="Casos por distrito (top 15)")
                        .update_layout(xaxis_tickangle=-45), width='stretch')

with tab4:
    cols_corr = ["casos", "casos_lag_1", "casos_lag_4", "promedio_movil_4",
                 "promedio_movil_8", "desviacion_movil_4", "edad_promedio",
                 "prop_con_signos", "prop_femenino"]
    cols_corr = [c for c in cols_corr if c in f.columns]
    st.plotly_chart(viz.matriz_correlacion(f, cols_corr), width='stretch')

# ---------------------------------------------------------------------------
# Outliers (1.5 x IQR)
# ---------------------------------------------------------------------------
ui.section("Outliers (regla 1.5 × IQR)")
casos_pos = f.loc[f["casos"] > 0, "casos"]
if len(casos_pos) > 0:
    q1, q3 = casos_pos.quantile(0.25), casos_pos.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = f[(f["casos"] > lim_sup)]
    ui.kpi_row([
        {"label": "Limite superior", "value": f"{lim_sup:.1f}", "icon": "📏", "accent": "#2a78d6"},
        {"label": "Semanas outlier", "value": f"{len(outliers):,}", "icon": "⚠️", "accent": "#eb6834"},
        {"label": "% del total", "value": f"{100*len(outliers)/max(len(f),1):.2f}%",
         "icon": "％", "accent": "#eda100"},
        {"label": "Caso maximo", "value": f"{int(f['casos'].max()):,}", "icon": "🔺", "accent": "#e34948"},
    ])
    st.info(
        "Los valores altos corresponden a semanas epidemicas reales; **no se eliminan** "
        "automaticamente. Se conservan porque representan brotes validos y son justamente "
        "lo que el modelo debe detectar.", icon="ℹ️",
    )
    st.markdown("**Distritos-semana con mas casos (outliers)**")
    st.dataframe(
        outliers.sort_values("casos", ascending=False)
        [["departamento", "distrito", "ano", "semana", "casos"]].head(15),
        width='stretch', hide_index=True,
    )

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
ui.section("Clustering de distritos (K-means)",
           "Cada distrito se resume en un perfil numerico (promedio, maximo, variabilidad, "
           "frecuencia de semanas con casos, semana tipica del pico) y se agrupa con K-means.")

# Recalcular evaluacion de k para mostrar codo y silueta (rapido sobre 571 distritos)
perfil = clustering.district_profile(df)
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(perfil[clustering.PERFIL_COLS].values)
evaluacion = clustering.evaluar_k(X)

col1, col2 = st.columns([3, 2])
with col1:
    st.plotly_chart(viz.curva_evaluacion_k(evaluacion), width='stretch')
with col2:
    k_final = int(clusters["cluster"].nunique())
    sil_final = float(evaluacion.loc[evaluacion["k"] == k_final, "silueta"].iloc[0]) \
        if k_final in evaluacion["k"].values else np.nan
    st.metric("Numero de clusters (k)", k_final)
    st.metric("Coeficiente de silueta", f"{sil_final:.3f}")
    st.caption(f"k configurable en config.py (K_MIN={config.K_MIN}, "
               f"K_MAX={config.K_MAX}, K_SELECTED={config.K_SELECTED}).")

st.plotly_chart(viz.scatter_clusters(clusters.set_index("ubigeo")), width='stretch')

st.markdown("**Perfil promedio de cada cluster**")
resumen = clustering.resumen_clusters(clusters.set_index("ubigeo"))
st.dataframe(resumen, width='stretch')
st.caption(
    "Interpretacion: los clusters ordenan a los distritos desde baja transmision "
    "(promedios y maximos bajos, pocas semanas con casos) hasta alta transmision "
    "(promedios y picos elevados, muchas semanas activas)."
)
