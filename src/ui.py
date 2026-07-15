"""Componentes y estilos de interfaz — tema OSCURO premium (glass + glow).

Centraliza el CSS, el banner, los encabezados de seccion y las tarjetas KPI para
que las paginas queden limpias y consistentes. Paleta accesible en src/visualizations.
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
  --bg:#0b0e14; --surface:#141a24; --surface-2:#1a2130;
  --ink:#e6e9ef; --ink-2:#9aa4b2; --muted:#6b7688;
  --line:rgba(255,255,255,0.08); --line-2:rgba(255,255,255,0.14);
  --azul:#4c8dff; --violeta:#a78bfa; --cyan:#2dd4bf;
  --radius:16px; --shadow:0 10px 34px rgba(0,0,0,0.42);
}

/* Fondo con resplandores radiales para dar profundidad */
.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 620px at 12% -12%, rgba(76,141,255,0.16), transparent 58%),
    radial-gradient(980px 560px at 104% -4%, rgba(167,139,250,0.13), transparent 55%),
    radial-gradient(820px 620px at 60% 118%, rgba(45,212,191,0.08), transparent 60%),
    var(--bg);
}
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1420px;
  animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from {opacity:0; transform: translateY(8px);} to {opacity:1; transform:none;} }

h1,h2,h3,h4 { letter-spacing:-0.015em; color: var(--ink); }
a { color: var(--azul) !important; }

/* --- Tarjetas de vidrio: metricas, graficos y tablas --- */
[data-testid="stMetric"],
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(180deg, rgba(26,33,48,0.72), rgba(18,23,33,0.72));
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
}
[data-testid="stMetric"] { padding: 15px 17px 13px; }
[data-testid="stPlotlyChart"] { padding: 10px 12px 6px; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size:0.72rem;
  text-transform:uppercase; letter-spacing:0.05em; font-weight:700; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; color: var(--ink);
  font-weight:750; }

/* --- Barra lateral --- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0e131d, #0b0e14);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
[data-testid="stSidebarNav"] a { border-radius: 10px; }
[data-testid="stSidebarNav"] a:hover { background: rgba(76,141,255,0.10); }
[data-testid="stSidebarNav"] [aria-current="page"] {
  background: linear-gradient(90deg, rgba(76,141,255,0.22), rgba(76,141,255,0.05));
  box-shadow: inset 3px 0 0 var(--azul);
}

/* --- Banner (hero) --- */
.hero {
  position: relative; overflow: hidden;
  background: linear-gradient(120deg, #12275f 0%, #2c5fd0 42%, #6d5bd0 74%, #1f8fb0 100%);
  background-size: 220% 220%;
  animation: heroShift 16s ease infinite;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 22px; padding: 26px 30px; margin-bottom: 20px;
  box-shadow: 0 18px 50px rgba(43,95,208,0.32), inset 0 1px 0 rgba(255,255,255,0.18);
}
@keyframes heroShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.hero::after {  /* malla de puntos sutil */
  content:""; position:absolute; inset:0; opacity:0.5;
  background-image: radial-gradient(rgba(255,255,255,0.10) 1px, transparent 1px);
  background-size: 22px 22px; pointer-events:none;
}
.hero .h-title { font-size:1.85rem; font-weight:800; line-height:1.12; color:#fff;
  position:relative; text-shadow:0 2px 14px rgba(0,0,0,0.28); }
.hero .h-sub { font-size:1rem; color:rgba(255,255,255,0.92); margin-top:6px; position:relative; }
.hero .h-badges { margin-top:14px; display:flex; flex-wrap:wrap; gap:9px; position:relative; }
.hero .badge { background:rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.30);
  padding:5px 13px; border-radius:999px; font-size:0.8rem; font-weight:600; color:#fff;
  backdrop-filter: blur(4px); }

/* --- Encabezado de seccion --- */
.section-h { display:flex; align-items:center; gap:11px; margin:26px 0 6px; }
.section-h .bar { width:5px; height:23px; border-radius:4px;
  background:linear-gradient(180deg, var(--azul), var(--violeta));
  box-shadow:0 0 12px rgba(76,141,255,0.6); }
.section-h .t { font-size:1.22rem; font-weight:750; color:var(--ink); }
.section-desc { color:var(--ink-2); font-size:0.9rem; margin:0 0 10px 16px; }

/* --- Tira de KPIs --- */
.kpi-strip { display:flex; gap:13px; flex-wrap:wrap; margin:6px 0 4px; }
.kpi { flex:1 1 165px; position:relative; overflow:hidden;
  background: linear-gradient(180deg, rgba(26,33,48,0.85), rgba(16,21,31,0.85));
  border:1px solid var(--line); border-radius:var(--radius);
  padding:15px 16px; display:flex; align-items:center; gap:13px;
  box-shadow:var(--shadow); transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
.kpi::before { content:""; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg, var(--accent,var(--azul)), transparent 85%); }
.kpi:hover { transform: translateY(-3px);
  border-color: color-mix(in srgb, var(--accent,var(--azul)) 45%, var(--line));
  box-shadow: 0 14px 40px rgba(0,0,0,0.5), 0 0 0 1px color-mix(in srgb, var(--accent,var(--azul)) 25%, transparent); }
.kpi .ico { width:42px; height:42px; flex:0 0 42px; border-radius:12px;
  display:flex; align-items:center; justify-content:center; font-size:1.25rem;
  background: color-mix(in srgb, var(--accent,var(--azul)) 18%, transparent);
  border:1px solid color-mix(in srgb, var(--accent,var(--azul)) 35%, transparent);
  box-shadow: 0 0 16px color-mix(in srgb, var(--accent,var(--azul)) 22%, transparent); }
.kpi .lbl { color:var(--muted); font-size:0.68rem; text-transform:uppercase;
  letter-spacing:0.05em; font-weight:800; }
.kpi .val { color:var(--ink); font-size:1.38rem; font-weight:800;
  font-variant-numeric:tabular-nums; line-height:1.18; }
.kpi .sub { color:var(--ink-2); font-size:0.74rem; }

/* --- Pestañas --- */
[data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--line); }
[data-baseweb="tab"] { font-weight:650; color:var(--ink-2); }
[data-baseweb="tab"][aria-selected="true"] { color:var(--ink); }
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg,var(--azul),var(--violeta)) !important; height:3px; }

/* --- Botones --- */
.stButton button, [data-testid="stFormSubmitButton"] button {
  border-radius:11px; font-weight:650; border:1px solid var(--line-2); }
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
  background: linear-gradient(120deg, var(--azul), #6d7bff);
  border:none; box-shadow:0 6px 20px rgba(76,141,255,0.35); }
.stButton button[kind="primary"]:hover { box-shadow:0 8px 26px rgba(76,141,255,0.5); }

/* --- Widgets de entrada --- */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
  background: var(--surface-2) !important; border-color: var(--line) !important; }
[data-testid="stForm"] { border:1px solid var(--line); border-radius:var(--radius);
  background: rgba(20,26,36,0.5); backdrop-filter: blur(6px); }

/* --- Alertas --- */
[data-testid="stAlert"] { border-radius:12px; border:1px solid var(--line); }

/* --- Barra de scroll --- */
::-webkit-scrollbar { width:11px; height:11px; }
::-webkit-scrollbar-thumb { background:#2a3444; border-radius:8px; border:2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background:#38455c; }
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
    """Banner superior con degradado animado."""
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
