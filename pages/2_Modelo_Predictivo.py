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
        "5 modelos comparados, umbral de decision, SHAP y prediccion en vivo.")

df = loaders.load_master()
modelos = loaders.load_models()
meta = modelos.get("meta") or {}
pipelines = loaders.load_clasificadores(modelos)
data = modeling.get_modeling_frame(df)

st.markdown(
    "**Objetivo:** predecir si un distrito superara su *alta incidencia* historica "
    "(percentil 75 del periodo de entrenamiento) en la **semana siguiente**. "
    f"Se comparan **{len(pipelines)} modelos** entrenados con division temporal "
    "(test = ultimo anio, 2024)."
)

# ---------------------------------------------------------------------------
# Comparacion de modelos
# ---------------------------------------------------------------------------
ui.section("Comparacion de modelos (test 2024)",
           "Un modelo lineal, un arbol simple y tres ensembles compitiendo por el mejor F1.")
umbral = st.slider("Umbral de decision", 0.1, 0.9,
                   float(meta.get("threshold", config.CLASSIFICATION_THRESHOLD)), 0.05)

tabla = modeling.metrics_table(pipelines, data["X_test"], data["y_test"], threshold=umbral)
st.dataframe(tabla.rename(index=modeling.MODEL_LABELS), width='stretch')

mejor = modeling.elegir_mejor_modelo(tabla)
base_rate = float(data["y_test"].mean())
ui.stat_chips([
    {"label": "Modelo", "value": modeling.MODEL_LABELS.get(mejor, mejor)},
    {"label": "ROC-AUC", "value": f"{tabla.loc[mejor, 'roc_auc']:.3f}"},
    {"label": "F1", "value": f"{tabla.loc[mejor, 'f1']:.3f}"},
    {"label": "Base rate", "value": f"{base_rate:.1%}"},
    {"label": "Umbral", "value": f"{umbral:.2f}"},
])
st.write("")
st.success(f"Mejor modelo por F1 con umbral {umbral}: **{modeling.MODEL_LABELS.get(mejor, mejor)}** "
           f"(recall {tabla.loc[mejor,'recall']:.3f}, precision {tabla.loc[mejor,'precision']:.3f}, "
           f"ROC-AUC {tabla.loc[mejor,'roc_auc']:.3f})")
st.caption("El ganador no se elige solo por accuracy: se prioriza el equilibrio recall/precision (F1).")

col1, col2 = st.columns(2)
with col1:
    m = tabla.loc[mejor]
    st.plotly_chart(
        viz.matriz_confusion(int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"]),
                             f"Matriz de confusion — {modeling.MODEL_LABELS.get(mejor, mejor)}"),
        width='stretch',
    )
with col2:
    sweep = modeling.threshold_sweep(pipelines[mejor], data["X_test"], data["y_test"])
    st.plotly_chart(viz.curva_umbral(sweep), width='stretch')
    st.caption("Efecto del umbral en precision, recall y F1.")

# Metricas POR CLASE del mejor modelo
st.markdown(f"**Metricas por clase — {modeling.MODEL_LABELS.get(mejor, mejor)}** "
            f"(accuracy global: {tabla.loc[mejor, 'accuracy']:.3f})")
st.dataframe(modeling.per_class_metrics(pipelines[mejor], data["X_test"],
             data["y_test"], threshold=umbral), width='stretch')
st.caption("Clase 1 = alta incidencia la semana siguiente (clase minoritaria en entrenamiento). "
           "¿Que error es mas costoso? Un falso negativo (no anticipar alta incidencia) "
           "suele ser mas grave en salud publica, por lo que se prioriza el recall de la clase 1.")

# Efecto del balanceo de clases (desbalance > 80/20)
bal = loaders.load_metricas_balanceo()
if bal is not None:
    ui.section("Efecto del balanceo de clases",
               "El entrenamiento tiene desbalance ~90/10. Se aplica class_weight (RF) y "
               "scale_pos_weight (XGBoost); aqui el efecto en el recall de la clase minoritaria.")
    piv = bal.pivot(index="modelo", columns="estado", values="recall_clase_1")
    piv["mejora"] = (piv["con balanceo"] - piv["sin balanceo"]).round(4)
    st.dataframe(piv, width='stretch')
    st.caption("recall de la clase 1 (alta incidencia) en el conjunto de prueba, sin y con balanceo.")

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
ui.section("Explicabilidad con SHAP",
           "Que variables influyen en la probabilidad predicha (asociacion, no causalidad).")
# Mejor modelo de arbol disponible (TreeExplainer). Prioriza el ganador si es de arbol.
if mejor in modeling.MODELOS_ARBOL:
    modelo_shap = mejor
else:
    modelo_shap = next((m for m in ["xgboost", "random_forest", "decision_tree"]
                        if m in pipelines), "random_forest")
st.caption(f"Explicando el modelo de arbol: **{modeling.MODEL_LABELS.get(modelo_shap, modelo_shap)}** "
           "(SHAP TreeExplainer).")


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
    opciones = list(pipelines)
    modelo_pred = st.radio("Modelo", opciones,
                           index=opciones.index(mejor) if mejor in opciones else 0,
                           format_func=lambda k: modeling.MODEL_LABELS.get(k, k),
                           horizontal=True)
    enviar = st.form_submit_button("Predecir", type="primary")

if enviar:
    if len(hist) < 4:
        st.warning("El distrito tiene muy poco historial para una prediccion confiable.")
    else:
        fila = modeling.construir_fila_prediccion(hist, casos_actual, semana, dep)
        pipe = pipelines[modelo_pred]
        proba = float(pipe.predict_proba(fila)[0, 1])
        pred = int(proba >= umbral)

        col_g, col_r = st.columns([1, 1.35], gap="large")
        with col_g:
            st.plotly_chart(viz.medidor_probabilidad(proba, umbral,
                            "Probabilidad de alta incidencia"), width='stretch')
        with col_r:
            estado = "Riesgo ALTO" if pred else "Riesgo bajo"
            desc = ("El perfil se acerca al comportamiento de alta incidencia."
                    if pred else "El perfil se aleja del comportamiento de alta incidencia.")
            ui.kpi_row([
                {"label": "Estado", "value": estado, "icon": "🚨" if pred else "✅",
                 "accent": "#ef4444" if pred else "#16a34a"},
                {"label": "Modelo", "value": modeling.MODEL_LABELS.get(modelo_pred, modelo_pred),
                 "icon": "🤖", "accent": "#0ea5a4"},
            ])
            st.write("")
            if pred:
                ui.callout("<b>Vigilar.</b> Reforzar prevencion y monitoreo del distrito "
                           "en las proximas semanas.")
            else:
                ui.callout("<b>Mantener.</b> Sin accion prioritaria; seguir el monitoreo "
                           "en el ciclo normal.")
            st.caption(desc)

        # Contribucion de cada variable (SHAP local, estilo force plot) para arboles
        if modelo_pred in modeling.MODELOS_ARBOL:
            import shap
            Xt1 = modeling.transform_X(pipe, fila)
            nombres1 = [n.split("__")[-1].replace("_", " ")
                        for n in modeling.transformed_feature_names(pipe)]
            expl1 = shap.TreeExplainer(pipe.named_steps["clf"])
            sv1 = expl1.shap_values(Xt1)
            if isinstance(sv1, list):
                sv1 = sv1[1]
            elif getattr(sv1, "ndim", 2) == 3:
                sv1 = sv1[:, :, 1]
            ui.section("Contribucion de cada variable", tag="valores SHAP · este cliente")
            st.plotly_chart(viz.barras_contribucion_shap(nombres1, np.asarray(sv1)[0]),
                            width='stretch')
            st.caption("Coral = empuja hacia ALTA incidencia · turquesa = empuja hacia baja. "
                       "SHAP muestra asociacion, no causalidad.")
        else:
            st.caption("La explicacion local SHAP (TreeExplainer) esta disponible para los "
                       "modelos de arbol (Random Forest, XGBoost, Arbol de Decision).")

        st.session_state["ultima_prediccion"] = {
            "departamento": dep, "distrito": dist_nombre, "semana": int(semana),
            "datos_entrada": {"casos_actual": int(casos_actual), "semana": int(semana)},
            "modelo": modelo_pred, "prediccion": "alta" if pred else "baja",
            "probabilidad": round(proba, 4),
        }
        st.info("Prediccion guardada en memoria: puedes registrarla en el **Panel 4 (CRUD)**.",
                icon="💾")
