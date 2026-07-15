"""Panel 3: pronostico de series temporales (media movil vs Holt-Winters)."""

import pandas as pd
import streamlit as st

import config
from src import forecasting, loaders, ui, visualizations as viz

ui.setup_page("Pronostico", "📈")

if not loaders.artefactos_listos():
    st.error("Faltan artefactos. Ejecuta `python train.py --rebuild`.")
    st.stop()

ui.hero("📈 Panel 3 — Pronostico de casos",
        "Media movil (baseline) vs suavizado exponencial de Holt-Winters.")

df = loaders.load_master()

st.sidebar.header("Configuracion del pronostico")
nivel = st.sidebar.selectbox("Nivel de agregacion", ["nacional", "departamento", "distrito"])

clave = None
if nivel == "departamento":
    clave = st.sidebar.selectbox("Departamento", sorted(df["departamento"].unique()))
elif nivel == "distrito":
    dep = st.sidebar.selectbox("Departamento", sorted(df["departamento"].unique()))
    dist_df = loaders.distritos_de(df, dep)
    dist_nombre = st.sidebar.selectbox("Distrito", dist_df["distrito"].tolist())
    clave = dist_df.loc[dist_df["distrito"] == dist_nombre, "ubigeo"].iloc[0]

periodos = st.sidebar.slider("Periodos futuros a pronosticar", 4, 12,
                             config.FORECAST_PERIODS)

serie = forecasting.build_series(df, nivel=nivel, clave=clave)

if serie.sum() == 0 or len(serie) < 3 * config.MOVING_AVERAGE_WINDOW:
    st.warning("La serie seleccionada no tiene suficiente historia para un pronostico confiable. "
               "Prueba con un nivel mas agregado (nacional o departamento).")
    st.stop()

# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
ev = forecasting.evaluate_models(serie)
mejor = ev["mejor_modelo"]

ui.section("Evaluacion en periodo de prueba (cronologico)")
ui.kpi_row([
    {"label": "MAPE media movil", "value": f"{ev['resultados']['media_movil']['mape']:.1f}%",
     "icon": "📉", "accent": "#6b7688"},
    {"label": "RMSE media movil", "value": f"{ev['resultados']['media_movil']['rmse']:.1f}",
     "icon": "📉", "accent": "#6b7688"},
    {"label": "MAPE Holt-Winters", "value": f"{ev['resultados']['holt_winters']['mape']:.1f}%",
     "icon": "📈", "accent": "#4c8dff"},
    {"label": "RMSE Holt-Winters", "value": f"{ev['resultados']['holt_winters']['rmse']:.1f}",
     "icon": "📈", "accent": "#4c8dff"},
])
st.success(f"Modelo elegido (menor RMSE): **{mejor}**")
st.caption("MAPE seguro: se excluyen semanas con 0 casos reales para evitar division por cero.")

# ---------------------------------------------------------------------------
# Pronostico futuro y grafico
# ---------------------------------------------------------------------------
futuro = forecasting.forecast_future(serie, periods=periodos)
fig = viz.grafico_pronostico(
    ev["train"], ev["test"], ev["pred_test"][mejor], futuro,
    titulo=f"Pronostico — nivel {nivel}" + (f" ({clave})" if clave else ""),
)
st.plotly_chart(fig, width='stretch')

ui.section(f"Pronostico de las proximas {periodos} semanas")
st.dataframe(futuro.assign(
    pronostico=futuro["pronostico"].round(1),
    inferior=futuro["inferior"].round(1),
    superior=futuro["superior"].round(1),
), width='stretch', hide_index=True)

st.caption("El intervalo es aproximado (±1.96·desviacion de residuos del suavizado exponencial).")
