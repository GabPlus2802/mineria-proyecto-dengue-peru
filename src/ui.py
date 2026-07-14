"""Componentes y estilos de interfaz para un aspecto de dashboard analitico.

Centraliza el CSS, el banner, los encabezados de seccion y las tarjetas KPI para
que las paginas queden limpias y consistentes. Paleta accesible (ver src/visualizations).
"""

from __future__ import annotations

import streamlit as st

from src import visualizations as viz

PALETTE = viz.PALETTE
AZUL = viz.AZUL

# ---------------------------------------------------------------------------
# CSS global (se inyecta una vez por ejecucion de pagina)
# ---------------------------------------------------------------------------
_CSS = """
<style>
:root {
  --azul:#2a78d6; --tinta:#0b0b0b; --tinta2:#52514e; --muted:#898781;
  --surface:#ffffff; --line:rgba(11,11,11,0.10); --radius:14px;
}
/* Ancho y espaciado del contenido */
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }

/* Tipografia de titulos */
h1, h2, h3 { letter-spacing: -0.01em; }

/* Metricas nativas -> tarjetas */
[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 14px 16px 12px;
  box-shadow: 0 1px 2px rgba(11,11,11,0.04);
}
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; color: var(--tinta); }

/* Graficos Plotly -> tarjeta blanca */
[data-testid="stPlotlyChart"] {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 8px 10px 4px;
  box-shadow: 0 1px 2px rgba(11,11,11,0.04);
}

/* Tablas dentro de tarjeta */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }

/* Pestañas mas marcadas */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
[data-baseweb="tab"] { font-weight: 600; }

/* Sidebar */
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* Banner (hero) */
.hero {
  background: linear-gradient(120deg, #1b5ba8 0%, #2a78d6 55%, #3d90ec 100%);
  border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
  color: #fff; box-shadow: 0 4px 18px rgba(42,120,214,0.22);
}
.hero .h-title { font-size: 1.7rem; font-weight: 700; line-height: 1.15; }
.hero .h-sub { font-size: 0.98rem; opacity: 0.92; margin-top: 4px; }
.hero .h-badges { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
.hero .badge { background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
  padding: 4px 11px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }

/* Encabezado de seccion con barra de acento */
.section-h { display: flex; align-items: center; gap: 10px; margin: 22px 0 6px; }
.section-h .bar { width: 4px; height: 22px; border-radius: 3px; background: var(--azul); }
.section-h .t { font-size: 1.18rem; font-weight: 700; color: var(--tinta); }
.section-desc { color: var(--tinta2); font-size: 0.9rem; margin: 0 0 10px 14px; }

/* Tira de KPIs */
.kpi-strip { display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 4px; }
.kpi { flex: 1 1 160px; background: var(--surface); border: 1px solid var(--line);
  border-left: 4px solid var(--accent, var(--azul)); border-radius: var(--radius);
  padding: 13px 15px; display: flex; align-items: center; gap: 12px;
  box-shadow: 0 1px 2px rgba(11,11,11,0.04); }
.kpi .ico { font-size: 1.35rem; line-height: 1; }
.kpi .lbl { color: var(--muted); font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.04em; font-weight: 700; }
.kpi .val { color: var(--tinta); font-size: 1.35rem; font-weight: 700;
  font-variant-numeric: tabular-nums; line-height: 1.15; }
.kpi .sub { color: var(--tinta2); font-size: 0.74rem; }
</style>
"""


def setup_page(title: str, icon: str, layout: str = "wide"):
    """Configura la pagina y aplica el estilo del dashboard."""
    st.set_page_config(page_title=title, page_icon=icon, layout=layout,
                       initial_sidebar_state="expanded")
    apply_base_style()


def apply_base_style():
    """Inyecta el CSS y fija la plantilla de Plotly (una vez por render)."""
    viz.use_theme()
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", badges: list[str] | None = None):
    """Banner superior con degradado."""
    badges_html = ""
    if badges:
        chips = "".join(f'<span class="badge">{b}</span>' for b in badges)
        badges_html = f'<div class="h-badges">{chips}</div>'
    st.markdown(
        f'<div class="hero"><div class="h-title">{title}</div>'
        f'<div class="h-sub">{subtitle}</div>{badges_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, desc: str = ""):
    """Encabezado de seccion con barra de acento."""
    st.markdown(f'<div class="section-h"><span class="bar"></span>'
                f'<span class="t">{title}</span></div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="section-desc">{desc}</div>', unsafe_allow_html=True)


def kpi_row(items: list[dict]):
    """Tira de tarjetas KPI.

    Cada item: {label, value, icon?, sub?, accent?}. accent es un color hex.
    """
    cards = []
    for it in items:
        accent = it.get("accent", AZUL)
        ico = f'<div class="ico">{it["icon"]}</div>' if it.get("icon") else ""
        sub = f'<div class="sub">{it["sub"]}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="kpi" style="--accent:{accent}">{ico}'
            f'<div><div class="lbl">{it["label"]}</div>'
            f'<div class="val">{it["value"]}</div>{sub}</div></div>'
        )
    st.markdown(f'<div class="kpi-strip">{"".join(cards)}</div>', unsafe_allow_html=True)
