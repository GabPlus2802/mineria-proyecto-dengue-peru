"""Graficos interactivos (Plotly) reutilizables por las paginas de Streamlit.

Mantiene la logica visual fuera de las paginas para que el codigo sea facil de
explicar y modificar en vivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Paleta y plantilla profesional para TEMA OSCURO
# (colores categoricos accesibles, escalonados para superficie oscura)
# ---------------------------------------------------------------------------
# Categorica en orden fijo (identidad); no se cicla ni se reordena.
PALETTE = ["#4c8dff", "#2dd4bf", "#f5b301", "#22c55e", "#a78bfa",
           "#f87171", "#f472b6", "#fb923c"]
AZUL = "#4c8dff"          # hue secuencial por defecto (magnitud)
AZUL_SEQ = ["#12233f", "#1e4b86", "#2f6fd0", "#4c8dff", "#8fbaff"]  # oscuro->claro
TINTA = "#e6e9ef"         # texto principal (claro)
TINTA_2 = "#9aa4b2"       # texto secundario
GRID = "#212836"          # rejilla tenue sobre fondo oscuro
EJE = "#2f3847"

_TEMPLATE = go.layout.Template(
    layout=dict(
        colorway=PALETTE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  color=TINTA, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=16, color=TINTA), x=0.01, xanchor="left"),
        margin=dict(l=48, r=24, t=52, b=44),
        xaxis=dict(gridcolor=GRID, linecolor=EJE, zeroline=False,
                   tickfont=dict(color=TINTA_2), title=dict(font=dict(color=TINTA_2))),
        yaxis=dict(gridcolor=GRID, linecolor=EJE, zeroline=False,
                   tickfont=dict(color=TINTA_2), title=dict(font=dict(color=TINTA_2))),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA_2), title_font=dict(color=TINTA_2)),
        colorscale=dict(sequential=[[i / (len(AZUL_SEQ) - 1), c] for i, c in enumerate(AZUL_SEQ)]),
        hoverlabel=dict(bgcolor="#141a24", bordercolor="#2f3847",
                        font=dict(family='system-ui, sans-serif', size=12, color=TINTA)),
    )
)
pio.templates["dengue"] = _TEMPLATE
PLANTILLA = "dengue"


def use_theme():
    """Fija la plantilla como predeterminada de Plotly (idempotente)."""
    pio.templates.default = "dengue"


def histograma(df: pd.DataFrame, col: str, titulo: str | None = None) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=50, template=PLANTILLA,
                       color_discrete_sequence=[AZUL],
                       title=titulo or f"Distribucion de {col}")
    fig.update_layout(bargap=0.05)
    return fig


def boxplot(df: pd.DataFrame, col: str, titulo: str | None = None) -> go.Figure:
    return px.box(df, y=col, points="outliers", template=PLANTILLA,
                  color_discrete_sequence=[AZUL],
                  title=titulo or f"Boxplot de {col}")


def matriz_correlacion(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    corr = df[cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, template=PLANTILLA, title="Matriz de correlacion")
    return fig


def evolucion_temporal(serie: pd.Series, titulo="Evolucion temporal de casos") -> go.Figure:
    fig = px.line(x=serie.index, y=serie.values, template=PLANTILLA, title=titulo,
                  color_discrete_sequence=[AZUL])
    fig.update_traces(line=dict(width=2))
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Casos")
    return fig


def barras_por_categoria(df: pd.DataFrame, cat: str, valor: str, top: int = 15,
                         titulo: str | None = None) -> go.Figure:
    agg = df.groupby(cat)[valor].sum().sort_values(ascending=False).head(top).reset_index()
    fig = px.bar(agg, x=cat, y=valor, template=PLANTILLA,
                 color_discrete_sequence=[AZUL],
                 title=titulo or f"{valor} por {cat} (top {top})")
    fig.update_traces(marker_line_width=0)
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def scatter_clusters(perfil: pd.DataFrame) -> go.Figure:
    d = perfil.reset_index()
    d["cluster"] = "Cluster " + d["cluster"].astype(str)
    fig = px.scatter(
        d, x="pca_1", y="pca_2", color="cluster",
        hover_data=["distrito", "departamento", "promedio_semanal", "maximo_semanal"],
        template=PLANTILLA, color_discrete_sequence=PALETTE,
        title="Clusters de distritos (proyeccion PCA 2D)",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="#0b0e14")))
    return fig


def curva_evaluacion_k(evaluacion: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=evaluacion["k"], y=evaluacion["inercia"],
                             name="Inercia (codo)", yaxis="y1", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=evaluacion["k"], y=evaluacion["silueta"],
                             name="Silueta", yaxis="y2", mode="lines+markers"))
    fig.update_layout(
        template=PLANTILLA, title="Metodo del codo y coeficiente de silueta",
        xaxis_title="Numero de clusters (k)",
        yaxis=dict(title="Inercia", side="left"),
        yaxis2=dict(title="Silueta", overlaying="y", side="right"),
    )
    return fig


def matriz_confusion(tn: int, fp: int, fn: int, tp: int, titulo="Matriz de confusion") -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    etiquetas = ["Predice 0", "Predice 1"]
    reales = ["Real 0", "Real 1"]
    fig = px.imshow(z, x=etiquetas, y=reales, text_auto=True, color_continuous_scale=AZUL_SEQ,
                    template=PLANTILLA, title=titulo)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def curva_umbral(sweep: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for metrica in ["precision", "recall", "f1"]:
        fig.add_trace(go.Scatter(x=sweep["umbral"], y=sweep[metrica],
                                 name=metrica, mode="lines+markers"))
    fig.update_layout(template=PLANTILLA, title="Efecto del umbral de decision",
                      xaxis_title="Umbral", yaxis_title="Metrica")
    return fig


def grafico_pronostico(train: pd.Series, test: pd.Series, pred_test: np.ndarray,
                       futuro: pd.DataFrame, titulo="Pronostico de casos") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Historico (train)",
                             mode="lines", line=dict(color=AZUL, width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Real (test)",
                             mode="lines", line=dict(color="#22c55e", width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=pred_test, name="Estimado (test)",
                             mode="lines", line=dict(color="#f5b301", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=futuro["fecha"], y=futuro["pronostico"],
                             name="Pronostico futuro", mode="lines+markers",
                             line=dict(color="#f87171", width=2)))
    fig.add_trace(go.Scatter(
        x=list(futuro["fecha"]) + list(futuro["fecha"][::-1]),
        y=list(futuro["superior"]) + list(futuro["inferior"][::-1]),
        fill="toself", fillcolor="rgba(248,113,113,0.16)", line=dict(width=0),
        name="Intervalo aprox.", hoverinfo="skip",
    ))
    fig.update_layout(template=PLANTILLA, title=titulo,
                      xaxis_title="Fecha", yaxis_title="Casos")
    return fig
