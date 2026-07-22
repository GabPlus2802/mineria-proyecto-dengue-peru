"""Graficos interactivos (Plotly) reutilizables por las paginas de Streamlit.

Mantiene la logica visual fuera de las paginas para que el codigo sea facil de
explicar y modificar en vivo.

Tema: superficie clara. La paleta categorica esta validada para daltonismo sobre
blanco (banda de luminosidad, piso de croma, separacion CVD adyacente >= 8 y
distincion a vision normal >= 15). Tres tonos quedan por debajo de 3:1 de
contraste sobre blanco, asi que todo grafico que los use lleva leyenda y
etiquetas visibles o una tabla al lado: el color nunca carga el significado solo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Superficies e ink
# ---------------------------------------------------------------------------
PLANO = "#f5f7fa"         # fondo de pagina
SUPERFICIE = "#ffffff"    # superficie de tarjeta / grafico
TINTA = "#0f172a"         # texto principal
TINTA_2 = "#475569"       # texto secundario
MUTED = "#64748b"         # ejes y etiquetas discretas
GRID = "#e8edf3"          # rejilla (hairline)
EJE = "#cbd5e1"           # linea base

# ---------------------------------------------------------------------------
# Paleta categorica (orden FIJO: el color sigue a la entidad, nunca al ranking)
# ---------------------------------------------------------------------------
SERIE = [
    "#2a78d6",  # 1 azul
    "#eb6834",  # 2 naranja
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 amarillo
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 verde
    "#4a3aa7",  # 7 violeta
    "#e34948",  # 8 rojo
]

# Magnitud de una sola serie: siempre el azul del slot 1
AZUL = SERIE[0]
# Rampa secuencial de un solo tono, claro -> oscuro
AZUL_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

# ---------------------------------------------------------------------------
# Estados reservados (nunca se reutilizan como color de serie).
# Siempre acompanados de icono + etiqueta: el color no lleva el significado solo.
# ---------------------------------------------------------------------------
BUENO = "#0ca30c"
AVISO = "#b45309"
SERIO = "#c2410c"
CRITICO = "#d03b3b"

# Par divergente para polaridad (SHAP): rojo empuja a ALTA, azul empuja a baja
POLO_ALTA = "#e34948"
POLO_BAJA = "#2a78d6"
NEUTRO = "#f0efec"

# Acento de marca: solo cromo de interfaz (nav, bordes, botones), nunca un dato
ACENTO = "#0d9488"
ACENTO_CLARO = "#5eead4"

# Serie proyectada (el tramo modelado de la serie temporal)
PROYECCION = "#7c3aed"

# Aliases historicos usados por src/ui.py y las paginas
PALETTE = SERIE
TEAL = ACENTO
TEAL_DARK = "#0f766e"
CORAL = CRITICO
CORAL_SOFT = "#f87171"

FUENTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_TEMPLATE = go.layout.Template(
    layout=dict(
        colorway=SERIE,
        font=dict(family=FUENTE, color=TINTA, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=15, color=TINTA), x=0.01, xanchor="left"),
        margin=dict(l=48, r=24, t=52, b=44),
        xaxis=dict(gridcolor=GRID, linecolor=EJE, zeroline=False,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=TINTA_2))),
        yaxis=dict(gridcolor=GRID, linecolor=EJE, zeroline=False,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=TINTA_2))),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA_2),
                    title_font=dict(color=TINTA_2)),
        colorscale=dict(sequential=[[i / (len(AZUL_SEQ) - 1), c]
                                    for i, c in enumerate(AZUL_SEQ)]),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=EJE,
                        font=dict(family=FUENTE, size=12, color=TINTA)),
    )
)
pio.templates["dengue"] = _TEMPLATE
PLANTILLA = "dengue"


def use_theme():
    """Fija la plantilla como predeterminada de Plotly (idempotente)."""
    pio.templates.default = "dengue"


# ---------------------------------------------------------------------------
# Distribucion
# ---------------------------------------------------------------------------
def histograma(df: pd.DataFrame, col: str, titulo: str | None = None) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=50, template=PLANTILLA,
                       color_discrete_sequence=[AZUL],
                       title=titulo or f"Distribucion de {col}")
    # 2px de separacion entre barras adyacentes: los rellenos no se tocan
    fig.update_layout(bargap=0.08, height=420,
                      xaxis_title=col, yaxis_title="Frecuencia")
    fig.update_traces(marker_line_width=0)
    return fig


def boxplot(df: pd.DataFrame, col: str, titulo: str | None = None) -> go.Figure:
    fig = px.box(df, y=col, points="outliers", template=PLANTILLA,
                 color_discrete_sequence=[AZUL],
                 title=titulo or f"Boxplot de {col}")
    fig.update_traces(marker=dict(size=5, opacity=0.55), line=dict(width=2))
    fig.update_layout(height=420, yaxis_title=col)
    return fig


def matriz_correlacion(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    corr = df[cols].corr(numeric_only=True)
    # Divergente: dos polos + gris neutro al centro (0 = sin relacion)
    escala = [[0.0, POLO_BAJA], [0.5, NEUTRO], [1.0, POLO_ALTA]]
    fig = px.imshow(corr, aspect="auto", color_continuous_scale=escala,
                    zmin=-1, zmax=1, template=PLANTILLA, title="Matriz de correlacion")
    fig.update_traces(texttemplate="<b>%{z:.2f}</b>", xgap=2, ygap=2,
                      textfont=dict(size=11, color=TINTA),
                      hovertemplate="%{y} vs %{x}<br>correlacion: %{z:.3f}<extra></extra>")
    fig.update_layout(height=560)
    return fig


# ---------------------------------------------------------------------------
# Series temporales
# ---------------------------------------------------------------------------
def evolucion_temporal(serie: pd.Series, titulo="Evolucion temporal de casos",
                       corte_simulado=None) -> go.Figure:
    """Serie semanal de casos.

    Si se pasa 'corte_simulado' (fecha), el tramo posterior se dibuja punteado y
    con su propio color, y la leyenda lo nombra como proyeccion: hasta esa fecha
    son notificaciones observadas y despues son valores estimados por el modelo
    estacional. La distincion va en la leyenda, como en cualquier grafico de
    pronostico, no en un cartel de advertencia.
    """
    fig = go.Figure()
    if corte_simulado is None:
        fig.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines",
                                 name="Casos observados",
                                 line=dict(color=AZUL, width=2)))
    else:
        corte = pd.Timestamp(corte_simulado)
        real = serie[serie.index <= corte]
        sim = serie[serie.index >= corte]
        fig.add_trace(go.Scatter(x=real.index, y=real.values, mode="lines",
                                 name="Observado (MINSA)",
                                 line=dict(color=AZUL, width=2)))
        fig.add_trace(go.Scatter(x=sim.index, y=sim.values, mode="lines",
                                 name="Proyeccion estacional",
                                 line=dict(color=PROYECCION, width=2, dash="dot")))
        fig.add_vline(x=corte, line_width=1, line_dash="dot", line_color=EJE)
    fig.update_layout(title=titulo, xaxis_title="Fecha", yaxis_title="Casos",
                      hovermode="x unified", template=PLANTILLA)
    return fig


def barras_por_categoria(df: pd.DataFrame, cat: str, valor: str, top: int = 15,
                         titulo: str | None = None) -> go.Figure:
    """Ranking horizontal.

    Horizontal y no vertical: con 15 nombres de distrito o departamento, las
    etiquetas verticales se recortan o se solapan. En horizontal cada nombre
    tiene su propia linea y el valor va como etiqueta directa al final de la barra.
    """
    agg = (df.groupby(cat)[valor].sum().sort_values(ascending=False)
           .head(top).reset_index().sort_values(valor))
    fig = px.bar(agg, x=valor, y=cat, orientation="h", template=PLANTILLA,
                 color_discrete_sequence=[AZUL],
                 title=titulo or f"{valor} por {cat} (top {top})",
                 text=agg[valor].map(lambda v: f"{int(v):,}"))
    # Extremo redondeado de 4px anclado a la linea base
    fig.update_traces(marker_line_width=0, marker=dict(cornerradius=4),
                      textposition="outside", cliponaxis=False,
                      textfont=dict(size=11, color=TINTA_2),
                      hovertemplate="%{y}<br>%{x:,.0f} casos<extra></extra>")
    fig.update_layout(height=max(420, 30 * len(agg) + 110), bargap=0.3,
                      yaxis_title="", xaxis_title=valor.capitalize(),
                      margin=dict(l=10, r=80, t=52, b=44))
    return fig


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def scatter_clusters(perfil: pd.DataFrame, nombres: dict | None = None,
                     colores: dict | None = None) -> go.Figure:
    """Mapa PCA de los distritos coloreado por grupo.

    'nombres' mapea el numero de cluster a su nombre epidemiologico, para que la
    leyenda diga que significa cada grupo y no "Cluster 0".
    'colores' mapea el numero de cluster a su color, para que coincida con el de
    las tarjetas de la seccion anterior.
    """
    d = perfil.reset_index()
    d["grupo"] = d["cluster"].map(nombres) if nombres else \
        "Grupo " + d["cluster"].astype(str)
    orden = ([nombres[c] for c in sorted(nombres)] if nombres
             else sorted(d["grupo"].unique()))
    mapa_color = ({nombres[c]: colores[c] for c in nombres} if nombres and colores
                  else None)

    fig = px.scatter(
        d, x="pca_1", y="pca_2", color="grupo",
        category_orders={"grupo": orden},
        color_discrete_map=mapa_color,
        color_discrete_sequence=None if mapa_color else SERIE,
        hover_data={"distrito": True, "departamento": True,
                    "promedio_semanal": ":.2f", "maximo_semanal": ":.0f",
                    "pca_1": False, "pca_2": False},
        template=PLANTILLA,
        title="Mapa de distritos segun su comportamiento (proyeccion PCA 2D)",
        labels={"pca_1": "Componente principal 1", "pca_2": "Componente principal 2"},
    )
    # Anillo de 2px del color de la superficie: los puntos superpuestos se separan
    fig.update_traces(marker=dict(size=9, opacity=0.9,
                                  line=dict(width=2, color=SUPERFICIE)))
    fig.update_layout(height=520, legend=dict(title="", orientation="h",
                                              yanchor="bottom", y=1.01,
                                              xanchor="left", x=0))
    return fig


def curva_evaluacion_k(evaluacion: pd.DataFrame) -> go.Figure:
    """Codo y silueta como dos paneles apilados: nunca dos escalas en un eje."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11,
                        subplot_titles=("Inercia (metodo del codo)",
                                        "Coeficiente de silueta"))
    fig.add_trace(go.Scatter(x=evaluacion["k"], y=evaluacion["inercia"],
                             name="Inercia", mode="lines+markers",
                             line=dict(color=SERIE[0], width=2),
                             marker=dict(size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=evaluacion["k"], y=evaluacion["silueta"],
                             name="Silueta", mode="lines+markers",
                             line=dict(color=SERIE[1], width=2),
                             marker=dict(size=8)), row=2, col=1)
    fig.update_xaxes(title_text="Numero de clusters (k)", row=2, col=1)
    fig.update_layout(template=PLANTILLA, height=430, showlegend=False,
                      title="Seleccion del numero de clusters", hovermode="x unified")
    for anot in fig.layout.annotations:
        anot.font.size = 12
        anot.font.color = TINTA_2
    return fig


# ---------------------------------------------------------------------------
# Clasificacion
# ---------------------------------------------------------------------------
def matriz_confusion(tn: int, fp: int, fn: int, tp: int,
                     titulo="Matriz de confusion") -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    fig = px.imshow(z, x=["Predice 0", "Predice 1"], y=["Real 0", "Real 1"],
                    text_auto=True, color_continuous_scale=AZUL_SEQ,
                    template=PLANTILLA, title=titulo)
    fig.update_traces(textfont=dict(size=17, color=TINTA), xgap=3, ygap=3)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def curva_umbral(sweep: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, metrica in enumerate(["precision", "recall", "f1"]):
        fig.add_trace(go.Scatter(x=sweep["umbral"], y=sweep[metrica], name=metrica,
                                 mode="lines", line=dict(color=SERIE[i], width=2)))
    fig.update_layout(template=PLANTILLA, title="Efecto del umbral de decision",
                      xaxis_title="Umbral", yaxis_title="Metrica",
                      hovermode="x unified")
    return fig


def medidor_probabilidad(proba: float, umbral: float = 0.5,
                         titulo: str = "Probabilidad") -> go.Figure:
    """Medidor circular. El color es un estado reservado y siempre viaja junto a
    la etiqueta de riesgo que muestra el panel."""
    color = CRITICO if proba >= umbral else BUENO
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 42, "color": TINTA}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": EJE,
                     "tickfont": {"color": MUTED, "size": 10}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, umbral * 100], "color": "rgba(12,163,12,0.14)"},
                {"range": [umbral * 100, 100], "color": "rgba(208,59,59,0.16)"},
            ],
            "threshold": {"line": {"color": TINTA_2, "width": 2}, "thickness": 0.85,
                          "value": umbral * 100},
        },
        title={"text": titulo.upper(), "font": {"size": 11, "color": MUTED}},
    ))
    fig.update_layout(template=PLANTILLA, height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def barras_contribucion_shap(nombres, valores, titulo="Contribucion de cada variable",
                             top: int = 10) -> go.Figure:
    """Barras horizontales de valores SHAP.

    Polaridad divergente: rojo empuja hacia ALTA incidencia, azul hacia baja.
    Cada barra lleva su valor como etiqueta directa, asi el signo no depende
    solo del color.
    """
    orden = np.argsort(np.abs(valores))[::-1][:top]
    nom = [nombres[i] for i in orden][::-1]
    val = [float(valores[i]) for i in orden][::-1]
    colores = [POLO_ALTA if v > 0 else POLO_BAJA for v in val]
    fig = go.Figure(go.Bar(
        x=val, y=nom, orientation="h", marker_color=colores,
        marker=dict(cornerradius=4),
        text=[f"{v:+.2f}" for v in val], textposition="outside",
        textfont=dict(color=TINTA_2, size=11),
        hovertemplate="%{y}<br>SHAP: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(template=PLANTILLA, title=titulo,
                      height=max(260, 34 * len(nom) + 90),
                      xaxis_title="Valor SHAP (log-odds)", bargap=0.35,
                      margin=dict(l=10, r=40, t=52, b=40))
    fig.add_vline(x=0, line_width=1, line_color=EJE)
    return fig


# ---------------------------------------------------------------------------
# Pronostico
# ---------------------------------------------------------------------------
def grafico_pronostico(train: pd.Series, test: pd.Series, pred_test: np.ndarray,
                       futuro: pd.DataFrame, titulo="Pronostico de casos",
                       corte_simulado=None) -> go.Figure:
    """Historico, ajuste en prueba y proyeccion futura con intervalo.

    'corte_simulado' marca desde donde el historico deja de ser dato observado
    del MINSA y pasa a ser la extension generada.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Historico",
                             mode="lines", line=dict(color=SERIE[0], width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Real (prueba)",
                             mode="lines", line=dict(color=SERIE[2], width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=pred_test, name="Estimado (prueba)",
                             mode="lines", line=dict(color=SERIE[1], width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=list(futuro["fecha"]) + list(futuro["fecha"][::-1]),
        y=list(futuro["superior"]) + list(futuro["inferior"][::-1]),
        fill="toself", fillcolor="rgba(230,103,103,0.14)", line=dict(width=0),
        name="Intervalo aprox.", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=futuro["fecha"], y=futuro["pronostico"],
                             name="Pronostico futuro", mode="lines+markers",
                             line=dict(color=POLO_ALTA, width=2),
                             marker=dict(size=8, line=dict(width=2, color=SUPERFICIE))))
    if corte_simulado is not None:
        fig.add_vline(x=pd.Timestamp(corte_simulado), line_width=1, line_dash="dot",
                      line_color=EJE,
                      annotation_text="ultima notificacion observada",
                      annotation_font=dict(size=10, color=MUTED),
                      annotation_position="top left")
    fig.update_layout(template=PLANTILLA, title=titulo, xaxis_title="Fecha",
                      yaxis_title="Casos", hovermode="x unified")
    return fig
