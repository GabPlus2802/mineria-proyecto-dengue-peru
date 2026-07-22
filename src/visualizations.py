"""Graficos interactivos (Plotly) reutilizables por las paginas de Streamlit.

Mantiene la logica visual fuera de las paginas para que el codigo sea facil de
explicar y modificar en vivo.

Tema: superficie oscura tipo sala de vigilancia. La paleta categorica esta
validada para daltonismo sobre la superficie #0f1829 (banda de luminosidad,
piso de croma, separacion CVD adyacente >= 8 y contraste >= 3:1).
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
PLANO = "#080e1a"         # fondo de pagina
SUPERFICIE = "#0f1829"    # superficie de tarjeta / grafico
TINTA = "#f1f5f9"         # texto principal
TINTA_2 = "#b9c2d0"       # texto secundario
MUTED = "#8492a6"         # ejes y etiquetas discretas
GRID = "#1c2740"          # rejilla (hairline)
EJE = "#2b3a52"           # linea base

# ---------------------------------------------------------------------------
# Paleta categorica (orden FIJO: el color sigue a la entidad, nunca al ranking)
# ---------------------------------------------------------------------------
SERIE = [
    "#3987e5",  # 1 azul
    "#d95926",  # 2 naranja
    "#199e70",  # 3 aqua
    "#c98500",  # 4 amarillo
    "#d55181",  # 5 magenta
    "#008300",  # 6 verde
    "#9085e9",  # 7 violeta
    "#e66767",  # 8 rojo
]

# Magnitud de una sola serie: siempre el azul del slot 1
AZUL = SERIE[0]
# Rampa secuencial de un solo tono; el extremo bajo se acerca a la superficie
AZUL_SEQ = ["#12325e", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"]

# ---------------------------------------------------------------------------
# Estados reservados (nunca se reutilizan como color de serie).
# Siempre acompanados de icono + etiqueta: el color no lleva el significado solo.
# ---------------------------------------------------------------------------
BUENO = "#0ca30c"
AVISO = "#fab219"
SERIO = "#ec835a"
CRITICO = "#d03b3b"

# Par divergente para polaridad (SHAP): rojo empuja a ALTA, azul empuja a baja
POLO_ALTA = "#e66767"
POLO_BAJA = "#3987e5"
NEUTRO = "#38445c"

# Acento de marca: solo cromo de interfaz (nav, bordes, botones), nunca un dato
ACENTO = "#22d3ee"

# Aliases historicos usados por src/ui.py y las paginas
PALETTE = SERIE
TEAL = ACENTO
TEAL_DARK = "#0e7490"
CORAL = CRITICO
CORAL_SOFT = "#e66767"

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
        hoverlabel=dict(bgcolor="#16203a", bordercolor=EJE,
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
    fig.update_layout(bargap=0.08)
    fig.update_traces(marker_line_width=0)
    return fig


def boxplot(df: pd.DataFrame, col: str, titulo: str | None = None) -> go.Figure:
    fig = px.box(df, y=col, points="outliers", template=PLANTILLA,
                 color_discrete_sequence=[AZUL],
                 title=titulo or f"Boxplot de {col}")
    fig.update_traces(marker=dict(size=5, opacity=0.55), line=dict(width=2))
    return fig


def matriz_correlacion(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    corr = df[cols].corr(numeric_only=True)
    # Divergente: dos polos + gris neutro al centro (0 = sin relacion)
    escala = [[0.0, POLO_BAJA], [0.5, NEUTRO], [1.0, POLO_ALTA]]
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=escala,
                    zmin=-1, zmax=1, template=PLANTILLA, title="Matriz de correlacion")
    fig.update_traces(textfont=dict(size=10, color=TINTA), xgap=2, ygap=2)
    return fig


# ---------------------------------------------------------------------------
# Series temporales
# ---------------------------------------------------------------------------
def evolucion_temporal(serie: pd.Series, titulo="Evolucion temporal de casos",
                       corte_simulado=None) -> go.Figure:
    """Serie semanal. Si se pasa 'corte_simulado' (fecha), el tramo posterior se
    dibuja punteado y en otro color: son registros generados, no observados."""
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
                                 name="Simulado",
                                 line=dict(color=AVISO, width=2, dash="dot")))
        fig.add_vline(x=corte, line_width=1, line_dash="dash", line_color=MUTED)
    fig.update_layout(title=titulo, xaxis_title="Fecha", yaxis_title="Casos",
                      hovermode="x unified", template=PLANTILLA)
    return fig


def barras_por_categoria(df: pd.DataFrame, cat: str, valor: str, top: int = 15,
                         titulo: str | None = None) -> go.Figure:
    agg = df.groupby(cat)[valor].sum().sort_values(ascending=False).head(top).reset_index()
    fig = px.bar(agg, x=cat, y=valor, template=PLANTILLA,
                 color_discrete_sequence=[AZUL],
                 title=titulo or f"{valor} por {cat} (top {top})")
    # Extremo redondeado de 4px anclado a la linea base
    fig.update_traces(marker_line_width=0, marker=dict(cornerradius=4))
    fig.update_layout(xaxis_tickangle=-45, bargap=0.28)
    return fig


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def scatter_clusters(perfil: pd.DataFrame) -> go.Figure:
    d = perfil.reset_index()
    d["cluster"] = "Cluster " + d["cluster"].astype(str)
    fig = px.scatter(
        d, x="pca_1", y="pca_2", color="cluster",
        hover_data=["distrito", "departamento", "promedio_semanal", "maximo_semanal"],
        template=PLANTILLA, color_discrete_sequence=SERIE,
        title="Clusters de distritos (proyeccion PCA 2D)",
    )
    # Anillo de 2px del color de la superficie: los puntos superpuestos se separan
    fig.update_traces(marker=dict(size=9, opacity=0.9,
                                  line=dict(width=2, color=SUPERFICIE)))
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
        fig.add_vline(x=pd.Timestamp(corte_simulado), line_width=1, line_dash="dash",
                      line_color=MUTED,
                      annotation_text="fin del dato real",
                      annotation_font=dict(size=10, color=MUTED),
                      annotation_position="top left")
    fig.update_layout(template=PLANTILLA, title=titulo, xaxis_title="Fecha",
                      yaxis_title="Casos", hovermode="x unified")
    return fig
