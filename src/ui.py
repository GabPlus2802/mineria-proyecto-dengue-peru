"""Componentes y estilos de interfaz — "sala de vigilancia epidemiologica".

Superficie oscura azulada, acento cian de marca, etiquetas monoespaciadas y
tarjetas con mucho aire. El acento cian es SOLO cromo de interfaz: los colores
de datos viven en src/visualizations.py y estan validados para daltonismo.
"""

from __future__ import annotations

import streamlit as st

from src import visualizations as viz

PALETTE = viz.SERIE
ACENTO = viz.ACENTO
CRITICO = viz.CRITICO
BUENO = viz.BUENO
AVISO = viz.AVISO

_MONO = 'ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace'

# ---------------------------------------------------------------------------
# CSS global (se inyecta una vez por ejecucion de pagina)
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
:root {{
  --plano:{viz.PLANO}; --card:{viz.SUPERFICIE}; --card-2:#131e33;
  --ink:{viz.TINTA}; --ink-2:{viz.TINTA_2}; --muted:{viz.MUTED};
  --line:rgba(255,255,255,0.08); --line-2:rgba(255,255,255,0.14);
  --acento:{ACENTO}; --acento-d:#0e7490;
  --bueno:{BUENO}; --aviso:{AVISO}; --critico:{CRITICO};
  --radius:14px;
  --shadow:0 1px 2px rgba(0,0,0,0.4), 0 12px 34px rgba(0,0,0,0.34);
  --mono:{_MONO};
}}

/* --- Plano de pagina: dos halos y una retícula muy tenue --- */
.stApp, [data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1000px 520px at 92% -10%, rgba(34,211,238,0.10), transparent 62%),
    radial-gradient(760px 460px at -8% 2%, rgba(57,135,229,0.10), transparent 60%),
    linear-gradient(rgba(255,255,255,0.014) 1px, transparent 1px) 0 0/100% 34px,
    var(--plano);
  color:var(--ink);
}}
.block-container {{ padding-top:1.2rem; padding-bottom:3.2rem; max-width:1460px;
  animation:fadeIn .42s ease; }}
@keyframes fadeIn {{ from{{opacity:0; transform:translateY(7px);}} to{{opacity:1; transform:none;}} }}

h1,h2,h3,h4 {{ letter-spacing:-0.015em; color:var(--ink); }}
a {{ color:var(--acento) !important; }}
hr, [data-testid="stDivider"] {{ border-color:var(--line) !important; }}
code {{ background:rgba(34,211,238,0.10); color:#7dd3fc; border-radius:5px;
  padding:1px 5px; }}

/* --- Navegacion superior --- */
[data-testid="stNavigationMenu"] {{ border-bottom:1px solid var(--line);
  padding-bottom:2px; }}
[data-testid="stNavigationMenu"] a, header [role="tablist"] a {{
  border-radius:9px 9px 0 0; font-weight:650; color:var(--ink-2); }}
[data-testid="stNavigationMenu"] a:hover {{ background:rgba(34,211,238,0.09);
  color:var(--ink); }}
[data-testid="stNavigationMenu"] a[aria-current="page"] {{
  color:var(--acento) !important;
  background:linear-gradient(180deg, rgba(34,211,238,0.14), rgba(34,211,238,0.02));
  box-shadow:inset 0 -2px 0 var(--acento); }}

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
  font-family:var(--mono); text-transform:uppercase; letter-spacing:0.08em;
  font-weight:600; }}
[data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; color:var(--ink);
  font-weight:700; }}

/* --- Barra lateral --- */
[data-testid="stSidebar"] {{ background:linear-gradient(180deg, #0c1526, #0a1120);
  border-right:1px solid var(--line); }}
[data-testid="stSidebar"] .block-container {{ padding-top:1rem; }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ font-size:0.98rem; }}

/* --- Cabecera (hero) --- */
.hero {{ position:relative; overflow:hidden;
  background:linear-gradient(135deg, #12203a 0%, var(--card) 52%);
  border:1px solid var(--line); border-radius:16px; padding:21px 26px;
  margin-bottom:18px; box-shadow:var(--shadow); }}
.hero::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:5px;
  background:linear-gradient(180deg, var(--acento), {viz.SERIE[0]}); }}
.hero::after {{ content:""; position:absolute; right:-50px; top:-70px;
  width:270px; height:270px;
  background:radial-gradient(circle, rgba(34,211,238,0.16), transparent 70%);
  pointer-events:none; }}
.hero .h-title {{ font-size:1.56rem; font-weight:800; line-height:1.14;
  color:var(--ink); position:relative; }}
.hero .h-sub {{ font-size:0.95rem; color:var(--ink-2); margin-top:5px;
  position:relative; max-width:74ch; }}
.hero .h-badges {{ margin-top:14px; display:flex; flex-wrap:wrap; gap:8px;
  position:relative; }}
.hero .badge {{ font-family:var(--mono); text-transform:uppercase;
  letter-spacing:0.07em; font-size:0.67rem; font-weight:600; color:#67e8f9;
  background:rgba(34,211,238,0.10); border:1px solid rgba(34,211,238,0.28);
  padding:4px 10px; border-radius:7px; }}

/* --- Chips de estadistica --- */
.stat-strip {{ display:flex; gap:10px; flex-wrap:wrap; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:8px 14px; min-width:98px; box-shadow:var(--shadow); }}
.stat .k {{ font-family:var(--mono); text-transform:uppercase; letter-spacing:0.08em;
  font-size:0.61rem; color:var(--muted); font-weight:600; }}
.stat .v {{ font-family:var(--mono); font-size:0.98rem; color:var(--ink);
  font-weight:700; margin-top:2px; }}

/* --- Encabezado de seccion --- */
.section-h {{ display:flex; align-items:center; gap:10px; margin:26px 0 6px; }}
.section-h .bar {{ width:4px; height:20px; border-radius:3px;
  background:linear-gradient(180deg, var(--acento), {viz.SERIE[0]}); }}
.section-h .t {{ font-size:1.13rem; font-weight:750; color:var(--ink); }}
.section-h .tag {{ margin-left:auto; font-family:var(--mono); text-transform:uppercase;
  letter-spacing:0.08em; font-size:0.63rem; color:var(--muted); font-weight:600; }}
.section-desc {{ color:var(--ink-2); font-size:0.88rem; margin:0 0 10px 14px;
  max-width:96ch; }}

/* --- Tira de KPIs --- */
.kpi-strip {{ display:flex; gap:12px; flex-wrap:wrap; margin:6px 0 4px; }}
.kpi {{ flex:1 1 168px; position:relative; overflow:hidden; background:var(--card);
  border:1px solid var(--line); border-radius:var(--radius); padding:14px 16px;
  display:flex; align-items:center; gap:12px; box-shadow:var(--shadow);
  transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease; }}
.kpi::before {{ content:""; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg, var(--accent,var(--acento)), transparent 88%); }}
.kpi:hover {{ transform:translateY(-2px); border-color:var(--line-2);
  box-shadow:0 14px 36px rgba(0,0,0,0.5); }}
.kpi .ico {{ width:40px; height:40px; flex:0 0 40px; border-radius:11px;
  display:flex; align-items:center; justify-content:center; font-size:1.18rem;
  background:color-mix(in srgb, var(--accent,var(--acento)) 18%, transparent);
  border:1px solid color-mix(in srgb, var(--accent,var(--acento)) 34%, transparent); }}
.kpi .lbl {{ color:var(--muted); font-family:var(--mono); font-size:0.63rem;
  text-transform:uppercase; letter-spacing:0.08em; font-weight:600; }}
.kpi .val {{ color:var(--ink); font-size:1.33rem; font-weight:800;
  font-variant-numeric:tabular-nums; line-height:1.18; }}
.kpi .sub {{ color:var(--ink-2); font-size:0.72rem; }}

/* --- Callout --- */
.callout {{ background:rgba(250,178,25,0.09); border:1px solid rgba(250,178,25,0.3);
  border-left:4px solid var(--aviso); border-radius:10px; padding:12px 15px;
  color:var(--ink); font-size:0.91rem; }}

/* --- Aviso de datos simulados --- */
.sim-banner {{ display:flex; align-items:flex-start; gap:12px;
  background:linear-gradient(90deg, rgba(250,178,25,0.13), rgba(250,178,25,0.03));
  border:1px solid rgba(250,178,25,0.34); border-left:4px solid var(--aviso);
  border-radius:11px; padding:12px 16px; margin:6px 0 14px; }}
.sim-banner .ico {{ font-size:1.15rem; line-height:1.3; }}
.sim-banner .t {{ font-family:var(--mono); text-transform:uppercase;
  letter-spacing:0.08em; font-size:0.64rem; font-weight:700; color:var(--aviso); }}
.sim-banner .d {{ color:var(--ink-2); font-size:0.87rem; margin-top:3px; }}

/* --- Pestanas --- */
[data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--line); }}
[data-baseweb="tab"] {{ font-weight:650; color:var(--ink-2); }}
[data-baseweb="tab"][aria-selected="true"] {{ color:var(--acento); }}
[data-baseweb="tab-highlight"] {{ background:var(--acento) !important; height:3px; }}

/* --- Botones --- */
.stButton button, [data-testid="stFormSubmitButton"] button {{ border-radius:10px;
  font-weight:650; border-color:var(--line-2); }}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {{
  background:linear-gradient(120deg, var(--acento), {viz.SERIE[0]}); border:none;
  color:#05121c; box-shadow:0 6px 18px rgba(34,211,238,0.26); }}

/* --- Sliders: el control principal del panel de prediccion --- */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
  box-shadow:0 0 0 3px rgba(34,211,238,0.22); }}
[data-testid="stSlider"] label, [data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {{ font-weight:600; color:var(--ink-2); }}

/* --- Formulario / inputs / alertas --- */
[data-testid="stForm"] {{ border:1px solid var(--line); border-radius:var(--radius);
  background:var(--card); box-shadow:var(--shadow); }}
[data-testid="stAlert"] {{ border-radius:11px; border:1px solid var(--line); }}
[data-testid="stExpander"] details {{ background:var(--card);
  border:1px solid var(--line); border-radius:var(--radius); }}

/* --- Scroll --- */
::-webkit-scrollbar {{ width:11px; height:11px; }}
::-webkit-scrollbar-track {{ background:var(--plano); }}
::-webkit-scrollbar-thumb {{ background:#243248; border-radius:8px;
  border:2px solid var(--plano); }}
::-webkit-scrollbar-thumb:hover {{ background:#33455f; }}
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
    """Cabecera de panel con acento de marca."""
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
        accent = it.get("accent", ACENTO)
        ico = f'<div class="ico">{it["icon"]}</div>' if it.get("icon") else ""
        sub = f'<div class="sub">{it["sub"]}</div>' if it.get("sub") else ""
        cards.append(
            f'<div class="kpi" style="--accent:{accent}">{ico}'
            f'<div><div class="lbl">{it["label"]}</div>'
            f'<div class="val">{it["value"]}</div>{sub}</div></div>'
        )
    st.markdown(f'<div class="kpi-strip">{"".join(cards)}</div>', unsafe_allow_html=True)


def banner_simulacion(res: dict, detalle: str = ""):
    """Aviso permanente de que parte del periodo mostrado NO es dato real.

    'res' es lo que devuelve src.simulation.resumen(). Si no hay extension
    simulada no dibuja nada.
    """
    if not res.get("tiene_simulacion"):
        return
    texto = detalle or (
        f"Los registros desde <b>{res['desde']:%d/%m/%Y}</b> hasta "
        f"<b>{res['hasta']:%d/%m/%Y}</b> ({res['filas_simuladas']:,} filas de "
        f"{res['distritos']} distritos) fueron <b>generados por simulacion "
        f"estacional</b>, no son notificaciones del MINSA. La vigilancia real "
        f"publicada termina en {res['ultimo_ano_real']}."
    )
    st.markdown(
        f'<div class="sim-banner"><div class="ico">⚠️</div><div>'
        f'<div class="t">Datos simulados en el periodo reciente</div>'
        f'<div class="d">{texto}</div></div></div>',
        unsafe_allow_html=True,
    )
