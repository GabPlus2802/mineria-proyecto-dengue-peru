"""Panel 4: CRUD persistente de consultas (Supabase o modo local SQLite)."""

import pandas as pd
import streamlit as st

from src import database, loaders

st.set_page_config(page_title="CRUD de consultas", page_icon="🗂️", layout="wide")
st.title("🗂️ Panel 4 — Registro de consultas (CRUD)")


@st.cache_resource(show_spinner=False)
def _get_db():
    return database.get_db()


db = _get_db()
modo = getattr(db, "modo", "local")
if modo == "supabase":
    st.success("Conectado a **Supabase** (persistencia en produccion).", icon="✅")
else:
    st.warning(
        "Modo **local** (SQLite) — solo para desarrollo. El despliegue final debe "
        "configurar credenciales de Supabase en `.streamlit/secrets.toml`.", icon="⚠️",
    )

df = loaders.load_master() if loaders.artefactos_listos() else None

# ---------------------------------------------------------------------------
# CREAR
# ---------------------------------------------------------------------------
st.subheader("Crear consulta")
prefill = st.session_state.get("ultima_prediccion", {})
if prefill:
    st.caption("Prellenado con la ultima prediccion del Panel 2.")

with st.form("crear"):
    c = st.columns(4)
    departamento = c[0].text_input("Departamento", prefill.get("departamento", ""))
    distrito = c[1].text_input("Distrito", prefill.get("distrito", ""))
    semana = c[2].number_input("Semana", 1, 53, int(prefill.get("semana", 1)))
    modelo = c[3].selectbox("Modelo", ["random_forest", "xgboost"],
                            index=0 if prefill.get("modelo") != "xgboost" else 1)
    c2 = st.columns(2)
    prediccion = c2[0].selectbox("Prediccion", ["alta", "baja"],
                                 index=0 if prefill.get("prediccion") != "baja" else 1)
    probabilidad = c2[1].number_input("Probabilidad", 0.0, 1.0,
                                      float(prefill.get("probabilidad", 0.5)), 0.01)
    crear = st.form_submit_button("Crear registro", type="primary")

if crear:
    registro = {
        "departamento": departamento, "distrito": distrito, "semana": int(semana),
        "datos_entrada": prefill.get("datos_entrada", {"semana": int(semana)}),
        "modelo": modelo, "prediccion": prediccion, "probabilidad": float(probabilidad),
    }
    try:
        db.create(registro)
        st.success("Consulta creada.")
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo crear la consulta: {e}")

# ---------------------------------------------------------------------------
# LISTAR
# ---------------------------------------------------------------------------
st.subheader("Consultas registradas")
try:
    registros = db.list_all()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudieron listar las consultas: {e}")
    registros = []

if not registros:
    st.info("Aun no hay consultas registradas.")
    st.stop()

tabla = pd.DataFrame(registros)
cols_mostrar = [c for c in ["id", "departamento", "distrito", "semana", "modelo",
                            "prediccion", "probabilidad", "created_at", "updated_at"]
                if c in tabla.columns]
st.dataframe(tabla[cols_mostrar], width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# EDITAR / ELIMINAR
# ---------------------------------------------------------------------------
st.subheader("Editar o eliminar")
ids = tabla["id"].tolist()
id_sel = st.selectbox("Selecciona el ID", ids)
fila = tabla[tabla["id"] == id_sel].iloc[0]

with st.form("editar"):
    c = st.columns(4)
    e_dep = c[0].text_input("Departamento", str(fila.get("departamento", "")))
    e_dist = c[1].text_input("Distrito", str(fila.get("distrito", "")))
    e_sem = c[2].number_input("Semana", 1, 53, int(fila.get("semana", 1) or 1))
    e_pred = c[3].selectbox("Prediccion", ["alta", "baja"],
                            index=0 if fila.get("prediccion") != "baja" else 1)
    e_prob = st.number_input("Probabilidad", 0.0, 1.0,
                             float(fila.get("probabilidad", 0.5) or 0.5), 0.01)
    cols_btn = st.columns(2)
    editar = cols_btn[0].form_submit_button("Guardar cambios", type="primary")
    eliminar = cols_btn[1].form_submit_button("Eliminar registro")

if editar:
    try:
        db.update(int(id_sel), {
            "departamento": e_dep, "distrito": e_dist, "semana": int(e_sem),
            "prediccion": e_pred, "probabilidad": float(e_prob),
        })
        st.success(f"Consulta {id_sel} actualizada.")
        st.rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo actualizar: {e}")

if eliminar:
    try:
        db.delete(int(id_sel))
        st.success(f"Consulta {id_sel} eliminada.")
        st.rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo eliminar: {e}")
