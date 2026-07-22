"""Panel 1: Analisis exploratorio (EDA), outliers y clustering de distritos."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src import clustering, loaders, ui, visualizations as viz

if not loaders.artefactos_listos():
    st.error("Faltan artefactos. Ejecuta `python train.py --rebuild`.")
    st.stop()

ui.hero("📊 EDA y Clustering",
        "Exploracion de los datos, deteccion de outliers y agrupamiento de distritos "
        "segun su comportamiento epidemiologico.",
        badges=["1.5·IQR", "K-means + PCA", "Filtros en vivo"])

df = loaders.load_master()
clusters = loaders.load_clusters()
corte = loaders.corte_simulado(df)
res_sim = loaders.resumen_simulacion()

# ---------------------------------------------------------------------------
# Filtros interactivos
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")
deps = ["(Todos)"] + sorted(df["departamento"].unique().tolist())
dep_sel = st.sidebar.selectbox("Departamento", deps)
anos = sorted(df["ano"].unique().tolist())
rango = st.sidebar.select_slider("Rango de anios", options=anos,
                                 value=(anos[0], anos[-1]))
solo_reales = st.sidebar.checkbox(
    "Solo notificaciones observadas", value=False,
    help="Deja fuera el tramo proyectado y muestra unicamente lo publicado por el MINSA.")

f = df[(df["ano"] >= rango[0]) & (df["ano"] <= rango[1])].copy()
if dep_sel != "(Todos)":
    f = f[f["departamento"] == dep_sel]
if solo_reales:
    f = loaders.solo_real(f)

if f.empty:
    st.warning("Los filtros seleccionados no dejan ningun registro.")
    st.stop()

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
ui.section("Resumen", "Indicadores segun los filtros seleccionados.")
n_sim = int((f["origen"] == "simulado").sum()) if "origen" in f.columns else 0
ui.kpi_row([
    {"label": "Registros", "value": f"{len(f):,}", "icon": "📅", "accent": viz.SERIE[0],
     "sub": f"{n_sim:,} proyectados" if n_sim else "todo observado"},
    {"label": "Total de casos", "value": f"{int(f['casos'].sum()):,}", "icon": "🦟",
     "accent": viz.SERIE[7]},
    {"label": "Departamentos", "value": f["departamento"].nunique(), "icon": "🗺️",
     "accent": viz.SERIE[3]},
    {"label": "Distritos", "value": f"{f['ubigeo'].nunique():,}", "icon": "📍",
     "accent": viz.SERIE[2]},
    {"label": "Periodo", "value": f"{rango[0]}–{rango[1]}", "icon": "⏱️",
     "accent": viz.SERIE[6]},
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
    st.plotly_chart(
        viz.evolucion_temporal(serie, corte_simulado=None if solo_reales else corte),
        width='stretch')
    st.caption("Casos semanales agregados segun los filtros seleccionados.")
    if not solo_reales:
        ui.nota_proyeccion(res_sim)

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
        {"label": "Limite superior", "value": f"{lim_sup:.1f}", "icon": "📏",
         "accent": viz.SERIE[0]},
        {"label": "Semanas outlier", "value": f"{len(outliers):,}", "icon": "⚠️",
         "accent": viz.SERIE[1]},
        {"label": "% del total", "value": f"{100*len(outliers)/max(len(f),1):.2f}%",
         "icon": "％", "accent": viz.SERIE[3]},
        {"label": "Caso maximo", "value": f"{int(f['casos'].max()):,}", "icon": "🔺",
         "accent": viz.SERIE[7]},
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
ui.section("Agrupamiento de distritos (K-means)", tag="aprendizaje no supervisado")

st.markdown(
    """
**¿Por que agrupar?** Peru tiene **625 distritos** con historial de dengue y no se
pueden vigilar todos con la misma estrategia ni mirar 625 series una por una. El
agrupamiento responde a una pregunta concreta: *¿que distritos se comportan igual
frente al dengue?* Si un grupo comparte patron, comparte tambien el tipo de
intervencion que necesita, y basta con disenar una estrategia por grupo en lugar
de 625.

A diferencia de la clasificacion, aqui **no hay una respuesta correcta que copiar**:
nadie etiqueto los distritos previamente. El algoritmo encuentra la estructura por
si solo a partir de como se comportan.
    """
)

with st.expander("¿Como se construyen los grupos? (metodo en 3 pasos)"):
    st.markdown(
        """
1. **Se resume cada distrito en 8 numeros** que describen su comportamiento a lo
   largo de 25 anios, no un ano suelto: promedio y mediana semanal de casos,
   maximo historico, variabilidad, porcentaje de semanas de alta incidencia,
   crecimiento promedio, frecuencia de semanas con al menos un caso y la semana
   del ano en la que suele estar el pico.
2. **Se estandarizan** esas 8 medidas (media 0, desviacion 1). Sin este paso, el
   maximo historico —que llega a cientos— aplastaria a la frecuencia de semanas
   activas, que va de 0 a 1.
3. **K-means agrupa** los distritos buscando que cada uno quede cerca del centro
   de su grupo. El numero de grupos `k` se elige con el **metodo del codo**
   (donde deja de compensar anadir grupos) y el **coeficiente de silueta**
   (que tan bien separado queda cada distrito de los grupos vecinos).
        """
    )

ui.nota("Los grupos se construyen <b>solo con notificaciones observadas</b>, igual "
        "que el entrenamiento de los modelos: el tramo proyectado no los altera.")


@st.cache_data(show_spinner="Evaluando el numero de grupos...")
def _evaluacion_k():
    """Codo y silueta por k. Se cachea: recorre los distritos uno a uno."""
    from sklearn.preprocessing import StandardScaler

    perfil = clustering.district_profile(loaders.solo_real(df))
    X = StandardScaler().fit_transform(perfil[clustering.PERFIL_COLS].values)
    return clustering.evaluar_k(X)


evaluacion = _evaluacion_k()
perfil_clusters = clusters.set_index("ubigeo")
k_final = int(clusters["cluster"].nunique())
sil_final = float(evaluacion.loc[evaluacion["k"] == k_final, "silueta"].iloc[0]) \
    if k_final in evaluacion["k"].values else np.nan

# --- Los grupos encontrados, en lenguaje llano ---------------------------
ui.section(f"Los {k_final} perfiles encontrados",
           "Cada grupo reune distritos que se comportan de forma parecida frente al "
           "dengue. Los nombres los pusimos nosotros al interpretar los resultados; "
           "el algoritmo solo devuelve grupos numerados.")

etiquetas = clustering.etiquetar_clusters(perfil_clusters)
aporte = clustering.aporte_de_casos(loaders.solo_real(df), perfil_clusters)
# Del mas leve al mas intenso: azul, ambar, rojo
ACENTOS = [viz.SERIE[0], viz.SERIE[3], viz.SERIE[7], viz.SERIE[6]]

cols = st.columns(len(etiquetas), gap="medium")
for col, (_, fila) in zip(cols, etiquetas.iterrows()):
    with col:
        ui.tarjeta_cluster(
            nombre=fila["nombre"],
            n_distritos=int(fila["n_distritos"]),
            descripcion=fila["descripcion"],
            cifras=[
                {"label": "Casos/semana", "value": f"{fila['promedio_semanal']:.1f}"},
                {"label": "Pico historico", "value": f"{fila['maximo_semanal']:.0f}"},
                {"label": "Semanas activas",
                 "value": f"{fila['frecuencia_semanas_con_casos']:.0%}"},
                {"label": "% casos del pais",
                 "value": f"{aporte.get(fila['cluster'], 0):.1f}%"},
            ],
            accent=ACENTOS[int(fila["nivel"]) % len(ACENTOS)],
        )

mas_intenso = etiquetas.iloc[-1]
ui.callout(
    f"<b>Para que sirve esto.</b> Los <b>{int(mas_intenso['n_distritos'])} distritos</b> "
    f"de «{mas_intenso['nombre']}» son el "
    f"<b>{100*int(mas_intenso['n_distritos'])/len(perfil_clusters):.0f}%</b> del pais "
    f"pero concentran el <b>{aporte.get(mas_intenso['cluster'], 0):.0f}%</b> de todos "
    f"los casos notificados. Ahi es donde rinde mas cada sol invertido en control "
    f"del vector. Los grupos de menor intensidad no necesitan la misma inversion "
    f"permanente, sino vigilancia para detectar a tiempo un brote inusual."
)

# --- Como se eligio k ----------------------------------------------------
ui.section("Como se eligio el numero de grupos", tag="codo + silueta")
col1, col2 = st.columns([3, 2], gap="large")
with col1:
    st.plotly_chart(viz.curva_evaluacion_k(evaluacion), width='stretch')
with col2:
    st.metric("Numero de grupos (k)", k_final)
    st.metric("Coeficiente de silueta", f"{sil_final:.3f}")
    st.markdown(
        f"""
La **inercia** siempre baja al anadir grupos, asi que no sirve sola: se busca el
*codo*, el punto donde deja de compensar.

La **silueta** va de −1 a 1 y mide que tan bien separado esta cada distrito de
los grupos vecinos. Aqui vale **{sil_final:.3f}**: una estructura real pero con
fronteras difusas, algo esperable porque la transmision es un **continuo**, no
categorias naturales con lineas nitidas.
        """
    )
    st.caption(f"Configurable en config.py: K_MIN={config.K_MIN}, "
               f"K_MAX={config.K_MAX}, K_SELECTED={config.K_SELECTED}.")

# --- Mapa PCA ------------------------------------------------------------
ui.section("Mapa de los distritos", tag="proyeccion PCA 2D")
st.markdown(
    "Los 8 indicadores se comprimen a 2 dimensiones para poder dibujarlos. Cada "
    "punto es un distrito y la cercania indica comportamiento parecido. Los ejes "
    "no tienen unidades interpretables: solo importa **quien queda cerca de quien**."
)
st.plotly_chart(viz.scatter_clusters(perfil_clusters), width='stretch')

with st.expander("Ver la tabla numerica completa por grupo"):
    st.dataframe(clustering.resumen_clusters(perfil_clusters), width='stretch')
    st.caption(
        "promedio/mediana/maximo semanal = casos por semana · pct_semanas_alta = "
        "proporcion de semanas que superaron el umbral de alta incidencia · "
        "frecuencia_semanas_con_casos = proporcion de semanas con al menos un caso · "
        "semana_pico = semana del ano donde suele estar el maximo."
    )
