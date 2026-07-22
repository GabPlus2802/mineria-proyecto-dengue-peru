"""Panel 3: pronostico de series temporales (media movil vs Holt-Winters)."""

import pandas as pd
import streamlit as st

import config
from src import forecasting, loaders, ui, visualizations as viz

if not loaders.artefactos_listos():
    st.error("Faltan artefactos. Ejecuta `python train.py --rebuild`.")
    st.stop()

df = loaders.load_master()
corte = loaders.corte_simulado(df)
res_sim = loaders.resumen_simulacion()

ultima = pd.to_datetime(df["fecha"]).max()
ui.hero(
    "📈 Pronostico de casos",
    "Media movil (baseline) frente a suavizado exponencial de Holt-Winters, "
    "con proyeccion de las proximas semanas.",
    badges=[f"Serie hasta {ultima:%b %Y}", "MAPE seguro + RMSE",
            "Intervalo aproximado"],
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
st.sidebar.header("Configuracion del pronostico")
nivel = st.sidebar.selectbox("Nivel de agregacion", ["nacional", "departamento", "distrito"])

clave = None
etiqueta = "Peru"
if nivel == "departamento":
    clave = st.sidebar.selectbox("Departamento", sorted(df["departamento"].unique()))
    etiqueta = clave.title()
elif nivel == "distrito":
    dep = st.sidebar.selectbox("Departamento", sorted(df["departamento"].unique()))
    dist_df = loaders.distritos_de(df, dep)
    dist_nombre = st.sidebar.selectbox("Distrito", dist_df["distrito"].tolist())
    clave = dist_df.loc[dist_df["distrito"] == dist_nombre, "ubigeo"].iloc[0]
    etiqueta = f"{dist_nombre.title()} ({dep.title()})"

periodos = st.sidebar.slider("Semanas futuras a pronosticar", 4, 26,
                             config.FORECAST_PERIODS)
ventana = st.sidebar.slider("Semanas reservadas para evaluar", 8, 26,
                            config.FORECAST_EVAL_PERIODS,
                            help="Cuantas semanas finales se apartan como conjunto de "
                                 "prueba. Que modelo gana depende de este valor: mira "
                                 "la tabla de robustez mas abajo.")

# La serie completa (con el tramo proyectado) es de donde ARRANCA el pronostico.
serie = forecasting.build_series(df, nivel=nivel, clave=clave)
# La evaluacion se hace SOLO sobre notificaciones observadas: si la ventana de
# prueba cayera en el tramo proyectado, mediriamos que tan bien el modelo
# reproduce nuestra propia proyeccion.
serie_eval = forecasting.serie_para_evaluar(df, nivel=nivel, clave=clave)

if serie.empty or serie.sum() == 0 or len(serie_eval) < ventana + 3 * config.MOVING_AVERAGE_WINDOW:
    st.warning("La serie seleccionada no tiene suficiente historia para un pronostico "
               "confiable. Prueba con un nivel mas agregado (nacional o departamento).")
    st.stop()

# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
ev = forecasting.evaluate_models(serie_eval, ventana)
mejor = ev["mejor_modelo"]
NOMBRES = {"media_movil": "Media movil", "holt_winters": "Holt-Winters"}

ui.section("Evaluacion en el periodo de prueba", f"{etiqueta} — se reservan las "
           f"ultimas {ventana} semanas <b>de notificaciones observadas</b> "
           f"(hasta {serie_eval.index.max():%m/%Y}) y no se usan para ajustar.")
ui.kpi_row([
    {"label": "MAPE media movil",
     "value": f"{ev['resultados']['media_movil']['mape']:.1f}%",
     "icon": "📉", "accent": viz.SERIE[1]},
    {"label": "RMSE media movil",
     "value": f"{ev['resultados']['media_movil']['rmse']:.0f}",
     "icon": "📉", "accent": viz.SERIE[1]},
    {"label": "MAPE Holt-Winters",
     "value": f"{ev['resultados']['holt_winters']['mape']:.1f}%",
     "icon": "📈", "accent": viz.SERIE[0]},
    {"label": "RMSE Holt-Winters",
     "value": f"{ev['resultados']['holt_winters']['rmse']:.0f}",
     "icon": "📈", "accent": viz.SERIE[0]},
    {"label": "Modelo elegido", "value": NOMBRES.get(mejor, mejor),
     "icon": "🏆", "accent": viz.ACENTO, "sub": "menor RMSE"},
])
st.caption("MAPE seguro: se excluyen las semanas con 0 casos reales para no dividir "
           "entre cero. El modelo se elige por RMSE, que no sufre ese problema.")
ui.nota("Estas metricas se calculan <b>solo con datos observados</b>. Evaluarlas "
        "sobre el tramo proyectado mediria que tan bien el modelo reproduce la "
        "propia proyeccion, no la realidad.")

# ---------------------------------------------------------------------------
# Pronostico futuro
# ---------------------------------------------------------------------------
futuro = forecasting.forecast_future(serie, periods=periodos)
st.plotly_chart(
    viz.grafico_pronostico(ev["train"], ev["test"], ev["pred_test"][mejor], futuro,
                           titulo=f"Casos semanales y pronostico — {etiqueta}",
                           corte_simulado=corte),
    width='stretch')

col_t, col_r = st.columns([1, 1.15], gap="large")

with col_t:
    ui.section(f"Proximas {periodos} semanas", tag=f"desde {futuro['fecha'].iloc[0]:%d/%m/%Y}")
    st.dataframe(
        futuro.assign(
            fecha=futuro["fecha"].dt.strftime("%d/%m/%Y"),
            pronostico=futuro["pronostico"].round(0).astype(int),
            inferior=futuro["inferior"].round(0).astype(int),
            superior=futuro["superior"].round(0).astype(int),
        ),
        width='stretch', hide_index=True)
    st.caption("El modelo se ajusta en escala logaritmica, asi que el intervalo es "
               "asimetrico: un conteo no puede bajar de cero y tiene mas recorrido "
               "hacia arriba. Es aproximado (±1.96 · desviacion de los residuos), "
               "no un intervalo de prediccion exacto.")

with col_r:
    ui.section("Robustez ante la ventana de evaluacion",
               "Que modelo gana cambia segun cuantas semanas se reserven para prueba. "
               "Se muestran todas para no elegir la que favorezca un resultado.")
    st.dataframe(forecasting.tabla_robustez(serie_eval), width='stretch', hide_index=True)
    ui.callout(
        "Con ventanas muy cortas la <b>media movil</b> puede ganar por construccion: "
        "pronostica una constante y en pocas semanas eso se parece al promedio real. "
        "Un componente estacional de 52 semanas necesita una ventana mas larga "
        "para mostrar su valor.")

ui.nota_proyeccion(res_sim)
