"""Panel 2: clasificacion (RF vs XGBoost), umbral, SHAP y prediccion en vivo."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import loaders, modeling, ui, visualizations as viz

ui.setup_page("Modelo Predictivo", "🤖")

if not loaders.artefactos_listos():
    st.error("Faltan artefactos. Ejecuta `python train.py --rebuild`.")
    st.stop()

ui.hero("🤖 Panel 2 — Modelo Predictivo y Explicabilidad",
        "Random Forest vs XGBoost, umbral de decision, SHAP y prediccion en vivo.")

df = loaders.load_master()
modelos = loaders.load_models()
meta = modelos.get("meta") or {}
pipelines = {"random_forest": modelos["random_forest"], "xgboost": modelos["xgboost"]}
data = modeling.get_modeling_frame(df)

st.markdown(
    "**Objetivo:** predecir si un distrito superara su *alta incidencia* historica "
    "(percentil 75 del periodo de entrenamiento) en la **semana siguiente**. "
    "Modelos entrenados con division temporal (test = ultimo anio, 2024)."
)

# ---------------------------------------------------------------------------
# Comparacion de modelos
# ---------------------------------------------------------------------------
ui.section("Comparacion de modelos (test 2024)")
umbral = st.slider("Umbral de decision", 0.1, 0.9,
                   float(meta.get("threshold", config.CLASSIFICATION_THRESHOLD)), 0.05)

tabla = modeling.metrics_table(pipelines, data["X_test"], data["y_test"], threshold=umbral)
st.dataframe(tabla, width='stretch')

mejor = modeling.elegir_mejor_modelo(tabla)
st.success(f"Mejor modelo por F1 con umbral {umbral}: **{mejor}** "
           f"(recall {tabla.loc[mejor,'recall']:.3f}, precision {tabla.loc[mejor,'precision']:.3f}, "
           f"ROC-AUC {tabla.loc[mejor,'roc_auc']:.3f})")
st.caption("El ganador no se elige solo por accuracy: se prioriza el equilibrio recall/precision (F1).")

col1, col2 = st.columns(2)
with col1:
    m = tabla.loc[mejor]
    st.plotly_chart(
        viz.matriz_confusion(int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"]),
                             f"Matriz de confusion — {mejor}"),
        width='stretch',
    )
with col2:
    sweep = modeling.threshold_sweep(pipelines[mejor], data["X_test"], data["y_test"])
    st.plotly_chart(viz.curva_umbral(sweep), width='stretch')
    st.caption("Efecto del umbral en precision, recall y F1.")

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
ui.section("Explicabilidad con SHAP",
           "Que variables influyen en la probabilidad predicha (asociacion, no causalidad).")
modelo_shap = mejor if mejor in ("random_forest", "xgboost") else "random_forest"


@st.cache_resource(show_spinner="Calculando valores SHAP...")
def _shap_data(nombre, n=400):
    import shap

    pipe = pipelines[nombre]
    Xs = data["X_test"].sample(min(n, len(data["X_test"])), random_state=config.RANDOM_STATE)
    Xt = modeling.transform_X(pipe, Xs)
    nombres = modeling.transformed_feature_names(pipe)
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    sv = explainer.shap_values(Xt)
    base = explainer.expected_value
    # Normalizar a la clase positiva
    if isinstance(sv, list):
        sv, base = sv[1], base[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv, base = sv[:, :, 1], (base[1] if np.ndim(base) else base)
    return np.asarray(sv), np.asarray(Xt), list(nombres), float(np.ravel(base)[0])


sv, Xt, nombres, base = _shap_data(modelo_shap)

tabg, tabl = st.tabs(["Explicacion global", "Explicacion local"])

with tabg:
    st.markdown("**Importancia global de variables** (summary plot)")
    fig = plt.figure()
    import shap
    shap.summary_plot(sv, Xt, feature_names=nombres, show=False, max_display=12)
    st.pyplot(fig, clear_figure=True)
    st.caption("Cada punto es una observacion; el color indica el valor de la variable. "
               "SHAP muestra asociacion con la probabilidad, no causalidad.")

with tabl:
    idx = st.number_input("Indice de la observacion a explicar", 0, len(Xt) - 1, 0)
    import shap
    expl = shap.Explanation(values=sv[int(idx)], base_values=base,
                            data=Xt[int(idx)], feature_names=nombres)
    fig2 = plt.figure()
    shap.plots.waterfall(expl, max_display=12, show=False)
    st.pyplot(fig2, clear_figure=True)
    st.caption("Variables que empujan la probabilidad hacia arriba (rojo) o hacia abajo (azul).")

# ---------------------------------------------------------------------------
# Formulario de prediccion en vivo
# ---------------------------------------------------------------------------
ui.section("Prediccion en vivo",
           "Selecciona un distrito; las variables derivadas (lags, medias moviles, "
           "estacionalidad) se calculan automaticamente desde su historial.")

with st.form("form_prediccion"):
    c = st.columns(4)
    dep = c[0].selectbox("Departamento", sorted(df["departamento"].unique()))
    dist_df = loaders.distritos_de(df, dep)
    dist_nombre = c[1].selectbox("Distrito", dist_df["distrito"].tolist())
    ubigeo = dist_df.loc[dist_df["distrito"] == dist_nombre, "ubigeo"].iloc[0]

    hist = df[df["ubigeo"].astype(str) == str(ubigeo)].sort_values("week_id")
    ultimo_casos = int(hist["casos"].iloc[-1]) if len(hist) else 0
    ultima_semana = int(hist["semana"].iloc[-1]) if len(hist) else 1

    casos_actual = c[2].number_input("Casos semana actual", 0, 5000, ultimo_casos)
    semana = c[3].number_input("Semana epidemiologica", 1, 53, ultima_semana)
    modelo_pred = st.radio("Modelo", ["random_forest", "xgboost"],
                           index=0 if mejor == "random_forest" else 1, horizontal=True)
    enviar = st.form_submit_button("Predecir", type="primary")

if enviar:
    if len(hist) < 4:
        st.warning("El distrito tiene muy poco historial para una prediccion confiable.")
    else:
        fila = modeling.construir_fila_prediccion(hist, casos_actual, semana, dep)
        pipe = pipelines[modelo_pred]
        proba = float(pipe.predict_proba(fila)[0, 1])
        pred = int(proba >= umbral)
        ui.kpi_row([
            {"label": "Prediccion", "value": "ALTA incidencia" if pred else "Baja incidencia",
             "icon": "🚨" if pred else "✅", "accent": "#e34948" if pred else "#008300"},
            {"label": "Probabilidad", "value": f"{proba:.1%}", "icon": "🎯", "accent": "#2a78d6"},
            {"label": "Modelo", "value": modelo_pred, "icon": "🤖", "accent": "#4a3aa7"},
        ])

        # Explicacion local de esta prediccion
        import shap
        Xt1 = modeling.transform_X(pipe, fila)
        nombres1 = modeling.transformed_feature_names(pipe)
        expl1 = shap.TreeExplainer(pipe.named_steps["clf"])
        sv1 = expl1.shap_values(Xt1)
        b1 = expl1.expected_value
        if isinstance(sv1, list):
            sv1, b1 = sv1[1], b1[1]
        elif getattr(sv1, "ndim", 2) == 3:
            sv1, b1 = sv1[:, :, 1], (b1[1] if np.ndim(b1) else b1)
        e = shap.Explanation(values=np.asarray(sv1)[0], base_values=float(np.ravel(b1)[0]),
                             data=Xt1[0], feature_names=nombres1)
        fig3 = plt.figure()
        shap.plots.waterfall(e, max_display=10, show=False)
        st.pyplot(fig3, clear_figure=True)

        st.session_state["ultima_prediccion"] = {
            "departamento": dep, "distrito": dist_nombre, "semana": int(semana),
            "datos_entrada": {"casos_actual": int(casos_actual), "semana": int(semana)},
            "modelo": modelo_pred, "prediccion": "alta" if pred else "baja",
            "probabilidad": round(proba, 4),
        }
        st.info("Prediccion guardada en memoria: puedes registrarla en el **Panel 4 (CRUD)**.",
                icon="💾")
