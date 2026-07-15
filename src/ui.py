"""Componentes y estilos de interfaz — tema CLARO refinado (estetica ChurnSense).

Superficies claras, acento turquesa, etiquetas monoespaciadas y tarjetas con
mucho aire. Paleta de datos accesible en src/visualizations.
"""

from __future__ import annotations

import streamlit as st

from src import visualizations as viz

PALETTE = viz.PALETTE
TEAL = viz.TEAL
CORAL = viz.CORAL

_MONO = 'ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace'

# ---------------------------------------------------------------------------
# CSS global (se inyecta una vez por ejecucion de pagina)
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
:root {{
  --bg:#f4f7fa; --card:#ffffff; --card-2:#fbfcfe;
  --ink:#0f172a; --ink-2:#475569; --muted:#94a3b8;
  --line:#e6ebf1; --line-2:#dbe2ea;
  --teal:{TEAL}; --teal-d:#0d7d72; --coral:{CORAL};
  --radius:14px;
  --shadow:0 1px 2px rgba(15,23,42,0.05), 0 10px 26px rgba(15,23,42,0.05);
  --mono:{_MONO};
}}

.stApp, [data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(900px 480px at 100% -8%, rgba(14,165,164,0.06), transparent 60%),
    radial-gradient(700px 420px at -6% 4%, rgba(59,130,246,0.05), transparent 58%),
    var(--bg);
}}
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1440px;
  animation: fadeIn .45s ease; }}
@keyframes fadeIn {{ from{{opacity:0; transform:translateY(6px);}} to{{opacity:1; transform:none;}} }}

h1,h2,h3,h4 {{ letter-spacing:-0.015em; color:var(--ink); }}
a {{ color:var(--teal) !important; }}
.mono {{ font-family:var(--mono); text-transform:uppercase; letter-spacing:0.09em; }}

/* --- Tarjetas: metricas, graficos, tablas, contenedores --- */
[data-testid="stMetric"],
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stVerticalBlockBorderWrapper"] {{
  background:var(--card); border:1px solid var(--line);
  border-radius:var(--radius); box-shadow:var(--shadow);
}}
[data-testid="stMetric"] {{ padding:15px 17px 13px; }}
[data-testid="stPlotlyChart"] {{ padding:10px 12px 6px; }}
[data-testid="stMetricLabel"] p {{ color:var(--muted); font-size:0.7rem;
  font-family:var(--mono); text-transform:uppercase; letter-spacing:0.07em; font-weight:600; }}
[data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; color:var(--ink); font-weight:700; }}

/* --- Barra lateral --- */
[data-testid="stSidebar"] {{ background:#ffffff; border-right:1px solid var(--line); }}
[data-testid="stSidebar"] .block-container {{ padding-top:1rem; }}
[data-testid="stSidebarNav"] a {{ border-radius:9px; }}
[data-testid="stSidebarNav"] a:hover {{ background:rgba(14,165,164,0.08); }}
[data-testid="stSidebarNav"] [aria-current="page"] {{
  background:linear-gradient(90deg, rgba(14,165,164,0.16), rgba(14,165,164,0.03));
  box-shadow:inset 3px 0 0 var(--teal); }}

/* --- Cabecera (hero) refinada --- */
.hero {{ position:relative; overflow:hidden; background:var(--card);
  border:1px solid var(--line); border-radius:16px; padding:20px 24px; margin-bottom:18px;
  box-shadow:var(--shadow); }}
.hero::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:5px;
  background:linear-gradient(180deg, var(--teal), #22d3ee); }}
.hero::after {{ content:""; position:absolute; right:-40px; top:-60px; width:240px; height:240px;
  background:radial-gradient(circle, rgba(14,165,164,0.10), transparent 70%); pointer-events:none; }}
.hero .h-title {{ font-size:1.55rem; font-weight:800; line-height:1.14; color:var(--ink);
  position:relative; }}
.hero .h-sub {{ font-size:0.96rem; color:var(--ink-2); margin-top:4px; position:relative; }}
.hero .h-badges {{ margin-top:13px; display:flex; flex-wrap:wrap; gap:8px; position:relative; }}
.hero .badge {{ font-family:var(--mono); text-transform:uppercase; letter-spacing:0.06em;
  font-size:0.68rem; font-weight:600; color:var(--teal-d);
  background:rgba(14,165,164,0.09); border:1px solid rgba(14,165,164,0.25);
  padding:4px 10px; border-radius:7px; }}

/* --- Chips de estadistica (mono, tipo ChurnSense) --- */
.stat-strip {{ display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:8px 14px; min-width:96px; box-shadow:var(--shadow); }}
.stat .k {{ font-family:var(--mono); text-transform:uppercase; letter-spacing:0.08em;
  font-size:0.62rem; color:var(--muted); font-weight:600; }}
.stat .v {{ font-family:var(--mono); font-size:0.98rem; color:var(--ink); font-weight:700;
  margin-top:2px; }}

/* --- Encabezado de seccion --- */
.section-h {{ display:flex; align-items:center; gap:10px; margin:24px 0 6px; }}
.section-h .bar {{ width:4px; height:20px; border-radius:3px;
  background:linear-gradient(180deg, var(--teal), #22d3ee); }}
.section-h .t {{ font-size:1.14rem; font-weight:750; color:var(--ink); }}
.section-h .tag {{ margin-left:auto; font-family:var(--mono); text-transform:uppercase;
  letter-spacing:0.08em; font-size:0.64rem; color:var(--muted); font-weight:600; }}
.section-desc {{ color:var(--ink-2); font-size:0.89rem; margin:0 0 10px 14px; }}

/* --- Tira de KPIs --- */
.kpi-strip {{ display:flex; gap:12px; flex-wrap:wrap; margin:6px 0 4px; }}
.kpi {{ flex:1 1 168px; position:relative; overflow:hidden; background:var(--card);
  border:1px solid var(--line); border-radius:var(--radius); padding:14px 16px;
  display:flex; align-items:center; gap:12px; box-shadow:var(--shadow);
  transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease; }}
.kpi::before {{ content:""; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg, var(--accent,var(--teal)), transparent 88%); }}
.kpi:hover {{ transform:translateY(-2px);
  border-color:color-mix(in srgb, var(--accent,var(--teal)) 45%, var(--line));
  box-shadow:0 10px 30px rgba(15,23,42,0.10); }}
.kpi .ico {{ width:40px; height:40px; flex:0 0 40px; border-radius:11px;
  display:flex; align-items:center; justify-content:center; font-size:1.2rem;
  background:color-mix(in srgb, var(--accent,var(--teal)) 12%, white);
  border:1px solid color-mix(in srgb, var(--accent,var(--teal)) 28%, transparent); }}
.kpi .lbl {{ color:var(--muted); font-family:var(--mono); font-size:0.64rem;
  text-transform:uppercase; letter-spacing:0.07em; font-weight:600; }}
.kpi .val {{ color:var(--ink); font-size:1.34rem; font-weight:800;
  font-variant-numeric:tabular-nums; line-height:1.18; }}
.kpi .sub {{ color:var(--ink-2); font-size:0.73rem; }}

/* --- Callout (recomendacion) --- */
.callout {{ background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.28);
  border-left:4px solid #f59e0b; border-radius:10px; padding:12px 15px; color:var(--ink);
  font-size:0.92rem; }}

/* --- Pestanas --- */
[data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--line); }}
[data-baseweb="tab"] {{ font-weight:650; color:var(--ink-2); }}
[data-baseweb="tab"][aria-selected="true"] {{ color:var(--teal-d); }}
[data-baseweb="tab-highlight"] {{ background:var(--teal) !important; height:3px; }}

/* --- Botones --- */
.stButton button, [data-testid="stFormSubmitButton"] button {{ border-radius:10px; font-weight:650; }}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {{
  background:linear-gradient(120deg, var(--teal), #14b8a6); border:none;
  box-shadow:0 5px 16px rgba(14,165,164,0.30); }}

/* --- Formulario / inputs --- */
[data-testid="stForm"] {{ border:1px solid var(--line); border-radius:var(--radius);
  background:var(--card); box-shadow:var(--shadow); }}
[data-testid="stAlert"] {{ border-radius:11px; border:1px solid var(--line); }}

/* --- Scroll --- */
::-webkit-scrollbar {{ width:11px; height:11px; }}
::-webkit-scrollbar-thumb {{ background:#cbd5e1; border-radius:8px; border:2px solid var(--bg); }}
::-webkit-scrollbar-thumb:hover {{ background:#94a3b8; }}
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
    """Cabecera refinada con acento turquesa."""
    badges_html = ""
    if badges:
        chips = "".join(f'<span class="badge">{b}</span>' for b in badges)
        badges_html = f'<div class="h-badges">{chips}</div>'
    st.markdown(
        f'<div class="hero"><div class="h-title">{title}</div>'
        f'<div class="h-sub">{subtitle}</div>{badges_html}</div>',
        unsafe_allow_html=True,
    )


def stat_chips(items: list[dict]):
    """Fila de chips de estadistica monoespaciados (label + value)."""
    chips = "".join(
        f'<div class="stat"><div class="k">{it["label"]}</div>'
        f'<div class="v">{it["value"]}</div></div>' for it in items
    )
    st.markdown(f'<div class="stat-strip">{chips}</div>', unsafe_allow_html=True)


def section(title: str, desc: str = "", tag: str = ""):
    """Encabezado de seccion; 'tag' es una etiqueta monoespaciada a la derecha."""
    tag_html = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(f'<div class="section-h"><span class="bar"></span>'
                f'<span class="t">{title}</span>{tag_html}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="section-desc">{desc}</div>', unsafe_allow_html=True)


def callout(html: str):
    """Bloque de recomendacion con acento ambar."""
    st.markdown(f'<div class="callout">{html}</div>', unsafe_allow_html=True)


def kpi_row(items: list[dict]):
    """Tira de tarjetas KPI. Cada item: {label, value, icon?, sub?, accent?}."""
    cards = []
    for it in items:
        accent = it.get("accent", TEAL)
        ico = f'<div class="ico">{it["icon"]}</div>' if it.get("icon") else ""
        sub = f'<div class="sub">{it["sub"]}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="kpi" style="--accent:{accent}">{ico}'
            f'<div><div class="lbl">{it["label"]}</div>'
            f'<div class="val">{it["value"]}</div>{sub}</div></div>'
        )
    st.markdown(f'<div class="kpi-strip">{"".join(cards)}</div>', unsafe_allow_html=True)
