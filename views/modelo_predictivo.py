"""Panel 2: simulador de prediccion, comparacion de 5 modelos y explicabilidad SHAP."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import loaders, modeling, ui, visualizations as viz

if not loaders.artefactos_listos():
    st.error("Faltan artefactos. Ejecuta `python train.py --rebuild`.")
    st.stop()

ui.hero(
    "🤖 Modelo Predictivo",
    "Mueve las variables y observa como cambia el riesgo de alta incidencia "
    "de la semana siguiente, con la explicacion de cada decision.",
    badges=["5 modelos", "Prediccion en vivo", "SHAP", "Test = 2024 real"],
)

df = loaders.load_master()
df_real = loaders.solo_real(df)
modelos = loaders.load_models()
meta = modelos.get("meta") or {}
pipelines = loaders.load_clasificadores(modelos)
data = modeling.get_modeling_frame(df_real)

umbral_defecto = float(meta.get("threshold", config.CLASSIFICATION_THRESHOLD))
if "umbral" not in st.session_state:
    st.session_state["umbral"] = umbral_defecto

tab_sim, tab_cmp, tab_shap = st.tabs(
    ["🎛️ Simulador de prediccion", "🏁 Comparacion de modelos", "🔍 Explicabilidad global"]
)


# ===========================================================================
# Utilidades del simulador
# ===========================================================================
CONTROLES = [
    ("casos", "Casos esta semana", "Notificaciones de la semana desde la que se predice."),
    ("casos_lag_1", "Casos hace 1 semana", "Valor de la semana inmediatamente anterior."),
    ("casos_lag_2", "Casos hace 2 semanas", "Dos semanas atras."),
    ("casos_lag_4", "Casos hace 4 semanas", "Un mes atras: referencia de nivel."),
    ("promedio_movil_4", "Promedio de las 4 semanas previas", "Nivel reciente del distrito."),
    ("promedio_movil_8", "Promedio de las 8 semanas previas", "Nivel de fondo del distrito."),
    ("desviacion_movil_4", "Variabilidad de las 4 semanas previas",
     "Desviacion estandar: mide que tan inestable viene la serie."),
    ("crecimiento_semanal", "Crecimiento respecto a la semana previa",
     "(casos - casos hace 1 semana) / (casos hace 1 semana + 1)."),
]


def rangos_de(tope: int) -> dict:
    """Limites y paso de cada slider: (minimo, maximo, paso, tipo).

    El tope se deriva del maximo historico del distrito, de modo que el rango
    del control tenga sentido para ese lugar.
    """
    return {
        "casos": (0, tope, 1, int),
        "casos_lag_1": (0, tope, 1, int),
        "casos_lag_2": (0, tope, 1, int),
        "casos_lag_4": (0, tope, 1, int),
        "promedio_movil_4": (0.0, float(tope), 0.5, float),
        "promedio_movil_8": (0.0, float(tope), 0.5, float),
        "desviacion_movil_4": (0.0, float(max(5, tope // 2)), 0.5, float),
        "crecimiento_semanal": (-1.0, 5.0, 0.05, float),
        "semana": (1, 53, 1, int),
    }


def sembrar_estado(valores: dict, rangos: dict):
    """Escribe los valores en session_state respetando limites y tipo.

    Sin el recorte, un valor fuera de rango (p. ej. un crecimiento mayor al
    maximo del slider) haria fallar al widget.
    """
    for clave, (lo, hi, _paso, tipo) in rangos.items():
        v = valores.get(clave, lo)
        st.session_state[f"sl_{clave}"] = tipo(min(max(v, lo), hi))


def valores_del_distrito(hist: pd.DataFrame) -> dict:
    """Valores reales de la ultima semana observada del distrito.

    Las medias y la desviacion se calculan EXCLUYENDO la semana actual, igual
    que en el entrenamiento (todas las moviles usan shift(1), sin fuga).
    """
    c = hist["casos"].tolist()

    def atras(n):
        return float(c[-n]) if len(c) >= n else 0.0

    previas_4 = c[-5:-1] if len(c) >= 5 else c[:-1] or [0.0]
    previas_8 = c[-9:-1] if len(c) >= 9 else c[:-1] or [0.0]
    lag1 = atras(2)
    actual = atras(1)
    return {
        "casos": actual,
        "casos_lag_1": lag1,
        "casos_lag_2": atras(3),
        "casos_lag_4": atras(5),
        "promedio_movil_4": float(np.mean(previas_4)),
        "promedio_movil_8": float(np.mean(previas_8)),
        "desviacion_movil_4": float(np.std(previas_4, ddof=1)) if len(previas_4) > 1 else 0.0,
        "crecimiento_semanal": (actual - lag1) / (lag1 + 1.0),
        "semana": int(hist["semana"].iloc[-1]) if len(hist) else 1,
    }


def derivadas_desde_lags(v: dict) -> dict:
    """Recalcula las variables derivadas a partir de los lags actuales.

    El promedio de 4 semanas se aproxima con los lags 1, 2 y 4: la semana 3 no
    es un predictor del modelo, asi que no hay un slider para ella.
    """
    lags = [v["casos_lag_1"], v["casos_lag_2"], v["casos_lag_4"]]
    return {
        **v,
        "promedio_movil_4": float(np.mean(lags)),
        "promedio_movil_8": float(np.mean(lags + [v["promedio_movil_8"]])),
        "desviacion_movil_4": float(np.std(lags, ddof=1)),
        "crecimiento_semanal": (v["casos"] - v["casos_lag_1"]) / (v["casos_lag_1"] + 1.0),
    }


def fila_desde_controles(v: dict, departamento: str) -> pd.DataFrame:
    """Arma la fila de features en el orden que espera el pipeline."""
    fila = {k: float(v[k]) for k, _, _ in CONTROLES}
    fila["semana_sen"] = float(np.sin(2 * np.pi * v["semana"] / 52.0))
    fila["semana_cos"] = float(np.cos(2 * np.pi * v["semana"] / 52.0))
    fila["departamento"] = departamento
    from src.preprocessing import FEATURE_CATEGORICAL, FEATURE_NUMERIC

    return pd.DataFrame([fila])[FEATURE_NUMERIC + FEATURE_CATEGORICAL]


@st.cache_resource(show_spinner=False)
def explainer_de(nombre: str):
    """Explicador SHAP cacheado (construirlo en cada rerun seria lentisimo)."""
    fondo = data["X_train"].sample(min(200, len(data["X_train"])),
                                   random_state=config.RANDOM_STATE)
    return modeling.crear_explainer(pipelines[nombre], nombre, fondo)


# ===========================================================================
# TAB 1 — Simulador de prediccion
# ===========================================================================
with tab_sim:
    ui.section("1. Elige el punto de partida",
               "Los sliders se precargan con los valores reales de la ultima semana "
               "registrada del distrito. A partir de ahi puedes mover cada variable.")

    c = st.columns([1.1, 1.4, 1.2, 1.3])
    dep = c[0].selectbox("Departamento", sorted(df["departamento"].unique()))
    dist_df = loaders.distritos_de(df, dep)
    dist_nombre = c[1].selectbox("Distrito", dist_df["distrito"].tolist())
    ubigeo = dist_df.loc[dist_df["distrito"] == dist_nombre, "ubigeo"].iloc[0]

    opciones = list(pipelines)
    mejor_guardado = meta.get("mejor_modelo", opciones[0])
    modelo_pred = c[2].selectbox(
        "Modelo", opciones,
        index=opciones.index(mejor_guardado) if mejor_guardado in opciones else 0,
        format_func=lambda k: modeling.MODEL_LABELS.get(k, k),
    )
    umbral = c[3].slider("Umbral de decision", min_value=0.05, max_value=0.95,
                         step=0.05, key="umbral",
                         help="Probabilidad a partir de la cual se declara riesgo alto. "
                              "Bajarlo detecta mas brotes pero genera mas falsas alarmas.")

    hist = df[df["ubigeo"].astype(str) == str(ubigeo)].sort_values("week_id")

    if len(hist) < 6:
        st.warning("Este distrito tiene muy poco historial para simular una prediccion. "
                   "Elige otro distrito.")
        st.stop()

    reales = valores_del_distrito(hist)
    tope = max(20, int(np.ceil(hist["casos"].max() * 1.5 / 10) * 10))
    rangos = rangos_de(tope)

    # --- Estado de los sliders -------------------------------------------
    # Al cambiar de distrito se recargan los valores reales de ese lugar.
    if st.session_state.get("_ubigeo_sim") != ubigeo:
        st.session_state["_ubigeo_sim"] = ubigeo
        sembrar_estado(reales, rangos)

    b = st.columns([1, 1, 3])
    if b[0].button("↺ Volver a los valores reales", width='stretch'):
        sembrar_estado(reales, rangos)
        st.rerun()
    if b[1].button("⚙️ Sincronizar derivadas", width='stretch',
                   help="Recalcula promedios, variabilidad y crecimiento a partir de los "
                        "lags que tienes puestos, para que el escenario sea coherente."):
        actual = {k: st.session_state[f"sl_{k}"] for k in rangos}
        sembrar_estado(derivadas_desde_lags(actual), rangos)
        st.rerun()

    ui.section("2. Mueve las variables", tag="− izquierda · + derecha")

    izq, der = st.columns(2, gap="large")
    valores = {}
    for i, (clave, etiqueta, ayuda) in enumerate(CONTROLES):
        lo, hi, paso, _tipo = rangos[clave]
        col = izq if i % 2 == 0 else der
        with col:
            # El valor inicial viaja por session_state, no por el argumento
            # 'value': pasar ambos hace que Streamlit ignore uno y avise.
            valores[clave] = float(st.slider(etiqueta, min_value=lo, max_value=hi,
                                             step=paso, help=ayuda, key=f"sl_{clave}"))

    valores["semana"] = st.slider(
        "Semana epidemiologica", min_value=1, max_value=53, step=1, key="sl_semana",
        help="Define la estacionalidad. El modelo la recibe como seno y coseno "
             "para que la semana 53 quede junto a la semana 1.",
    )

    sen = np.sin(2 * np.pi * valores["semana"] / 52.0)
    cos = np.cos(2 * np.pi * valores["semana"] / 52.0)
    st.caption(f"Variables derivadas de la semana — semana_sen = `{sen:+.3f}` · "
               f"semana_cos = `{cos:+.3f}`  (no se editan a mano: dependen de la semana)")

    # --- Prediccion en vivo ----------------------------------------------
    fila = fila_desde_controles(valores, dep)
    pipe = pipelines[modelo_pred]
    proba = float(pipe.predict_proba(fila)[0, 1])
    pred = int(proba >= umbral)

    ui.section("3. Resultado", tag="se actualiza al mover cualquier slider")

    col_g, col_r = st.columns([1, 1.4], gap="large")
    with col_g:
        st.plotly_chart(viz.medidor_probabilidad(proba, umbral,
                        "Probabilidad de alta incidencia"), width='stretch')
    with col_r:
        estado = "RIESGO ALTO" if pred else "RIESGO BAJO"
        base_hist = float(reales["casos"])
        ui.kpi_row([
            {"label": "Estado", "value": estado, "icon": "🚨" if pred else "✅",
             "accent": viz.CRITICO if pred else viz.BUENO},
            {"label": "Probabilidad", "value": f"{proba:.1%}", "icon": "🎯",
             "accent": viz.SERIE[0], "sub": f"umbral {umbral:.2f}"},
            {"label": "Modelo", "value": modeling.MODEL_LABELS.get(modelo_pred, modelo_pred),
             "icon": "🤖", "accent": viz.ACENTO},
        ])
        st.write("")
        if pred:
            ui.callout(
                "<b>Vigilar.</b> El perfil se parece al de las semanas que "
                "precedieron a una alta incidencia. Reforzar prevencion y "
                "monitoreo del distrito.")
        else:
            ui.callout(
                "<b>Mantener.</b> El perfil se aleja del comportamiento de alta "
                "incidencia. Seguir el monitoreo en el ciclo normal.")
        umbral_dist = hist["umbral_incidencia"].dropna()
        if len(umbral_dist):
            st.caption(
                f"Umbral historico de alta incidencia en {dist_nombre}: "
                f"**{float(umbral_dist.iloc[-1]):.1f} casos/semana** "
                f"(percentil {config.TARGET_PERCENTILE} del periodo de entrenamiento). "
                f"Ultima semana registrada: {base_hist:.0f} casos.")
        else:
            st.caption(f"Ultima semana registrada en {dist_nombre}: {base_hist:.0f} casos.")

    # --- Explicabilidad de ESTA prediccion --------------------------------
    ui.section("4. Por que esta prediccion", tag="valores SHAP · este escenario")
    explainer = explainer_de(modelo_pred)
    if explainer is None:
        st.info(
            f"**{modeling.MODEL_LABELS.get(modelo_pred, modelo_pred)}** no tiene un "
            "explicador SHAP exacto en este proyecto. Elige Random Forest, XGBoost, "
            "Arbol de Decision o Regresion Logistica para ver la contribucion de "
            "cada variable.", icon="ℹ️")
    else:
        sv = modeling.valores_shap(explainer, pipe, fila)
        st.plotly_chart(
            viz.barras_contribucion_shap(modeling.nombres_legibles(pipe), sv[0]),
            width='stretch')
        st.caption(
            "Rojo = empuja hacia ALTA incidencia · azul = empuja hacia baja. "
            "La longitud es cuanto mueve la prediccion (log-odds) y el numero "
            "acompana a cada barra, asi el signo no depende solo del color. "
            "SHAP muestra asociacion, no causalidad.")

    st.session_state["ultima_prediccion"] = {
        "departamento": dep, "distrito": dist_nombre, "semana": int(valores["semana"]),
        "datos_entrada": {k: round(float(valores[k]), 3) for k, _, _ in CONTROLES},
        "modelo": modelo_pred, "prediccion": "alta" if pred else "baja",
        "probabilidad": round(proba, 4),
    }
    ui.nota("Este escenario queda disponible para registrarlo en <b>Datos (CRUD)</b>.",
            icono="💾")


# ===========================================================================
# TAB 2 — Comparacion de modelos
# ===========================================================================
with tab_cmp:
    ui.section("Comparacion de modelos", "Division temporal: entrenamiento hasta 2022, "
               "validacion 2023 y prueba 2024. Solo datos reales del MINSA.")
    st.markdown(
        "**Objetivo:** predecir si un distrito superara su *alta incidencia* historica "
        f"(percentil {config.TARGET_PERCENTILE} del periodo de entrenamiento) en la "
        f"**semana siguiente**. Se comparan **{len(pipelines)} modelos**.")

    umbral_cmp = st.slider("Umbral de decision para la tabla", 0.1, 0.9,
                           umbral_defecto, 0.05, key="umbral_cmp")
    tabla = modeling.metrics_table(pipelines, data["X_test"], data["y_test"],
                                   threshold=umbral_cmp)
    st.dataframe(tabla.rename(index=modeling.MODEL_LABELS), width='stretch')

    mejor = modeling.elegir_mejor_modelo(tabla)
    ui.stat_chips([
        {"label": "Mejor por F1", "value": modeling.MODEL_LABELS.get(mejor, mejor)},
        {"label": "F1", "value": f"{tabla.loc[mejor, 'f1']:.3f}"},
        {"label": "ROC-AUC", "value": f"{tabla.loc[mejor, 'roc_auc']:.3f}"},
        {"label": "Recall", "value": f"{tabla.loc[mejor, 'recall']:.3f}"},
        {"label": "Base rate", "value": f"{float(data['y_test'].mean()):.1%}"},
        {"label": "Umbral", "value": f"{umbral_cmp:.2f}"},
    ])
    st.caption("El ganador no se elige por accuracy: se prioriza el equilibrio "
               "entre recall y precision (F1).")

    col1, col2 = st.columns(2)
    with col1:
        m = tabla.loc[mejor]
        st.plotly_chart(
            viz.matriz_confusion(int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"]),
                                 f"Matriz de confusion — {modeling.MODEL_LABELS.get(mejor, mejor)}"),
            width='stretch')
    with col2:
        sweep = modeling.threshold_sweep(pipelines[mejor], data["X_test"], data["y_test"])
        st.plotly_chart(viz.curva_umbral(sweep), width='stretch')

    ui.section("Metricas por clase", f"{modeling.MODEL_LABELS.get(mejor, mejor)} — "
               f"accuracy global {tabla.loc[mejor, 'accuracy']:.3f}")
    st.dataframe(modeling.per_class_metrics(pipelines[mejor], data["X_test"],
                 data["y_test"], threshold=umbral_cmp), width='stretch')
    st.caption("Clase 1 = alta incidencia la semana siguiente (minoritaria en "
               "entrenamiento). Un falso negativo —no anticipar alta incidencia— "
               "suele ser mas costoso en salud publica, por eso se prioriza su recall.")

    bal = loaders.load_metricas_balanceo()
    if bal is not None:
        ui.section("Efecto del balanceo de clases",
                   "El entrenamiento tiene desbalance ~90/10. Se aplica class_weight (RF) "
                   "y scale_pos_weight (XGBoost); aqui el efecto sobre la clase minoritaria.")
        piv = bal.pivot(index="modelo", columns="estado", values="recall_clase_1")
        piv["mejora"] = (piv["con balanceo"] - piv["sin balanceo"]).round(4)
        st.dataframe(piv, width='stretch')


# ===========================================================================
# TAB 3 — Explicabilidad global
# ===========================================================================
with tab_shap:
    ui.section("Importancia global de variables",
               "Que variables mueven la probabilidad en el conjunto de prueba "
               "(asociacion, no causalidad).")

    disponibles = [m for m in pipelines
                   if m in modeling.MODELOS_ARBOL or m == "logistic_regression"]
    modelo_shap = st.selectbox(
        "Modelo a explicar", disponibles,
        format_func=lambda k: modeling.MODEL_LABELS.get(k, k), key="shap_global")

    @st.cache_resource(show_spinner="Calculando valores SHAP...")
    def _shap_global(nombre, n=400):
        pipe_g = pipelines[nombre]
        Xs = data["X_test"].sample(min(n, len(data["X_test"])),
                                   random_state=config.RANDOM_STATE)
        expl = explainer_de(nombre)
        sv = modeling.valores_shap(expl, pipe_g, Xs)
        return np.asarray(sv), modeling.transform_X(pipe_g, Xs), modeling.nombres_legibles(pipe_g)

    sv_g, Xt_g, nombres_g = _shap_global(modelo_shap)

    col_a, col_b = st.columns([1.25, 1], gap="large")
    with col_a:
        st.markdown("**Summary plot** — cada punto es una observacion del conjunto de prueba.")
        import shap

        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(sv_g, Xt_g, feature_names=nombres_g, show=False, max_display=12)
        fig.patch.set_alpha(0)
        ax = plt.gca()
        ax.set_facecolor("none")
        ax.tick_params(colors=viz.TINTA_2)
        for lado in ax.spines.values():
            lado.set_color(viz.EJE)
        ax.xaxis.label.set_color(viz.TINTA_2)
        st.pyplot(fig, clear_figure=True, transparent=True)

    with col_b:
        st.markdown("**Importancia media** — magnitud promedio del efecto.")
        importancia = np.abs(sv_g).mean(axis=0)
        st.plotly_chart(
            viz.barras_contribucion_shap(nombres_g, importancia,
                                         titulo="|SHAP| promedio", top=12),
            width='stretch')

    st.caption("El color de cada punto en el summary plot indica el valor de la variable. "
               "SHAP muestra asociacion con la probabilidad predicha, no causalidad.")
