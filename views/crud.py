"""Registro de datos: CRUD completo sobre los registros de vigilancia y sobre
las consultas hechas al modelo (Supabase o base local)."""

import pandas as pd
import streamlit as st

from src import database, loaders, modeling, ui

ui.hero("🗂️ Gestion de datos",
        "Crear, consultar, editar y eliminar registros de vigilancia y consultas "
        "al modelo, con persistencia.",
        badges=["Create", "Read", "Update", "Delete"])


@st.cache_resource(show_spinner=False)
def _db(tabla: str):
    return database.get_db(tabla)


db_registros = _db(database.TABLA_REGISTROS)
db_consultas = _db(database.TABLA_CONSULTAS)
modo = getattr(db_registros, "modo", "local")
st.sidebar.caption("Almacenamiento: **Supabase**" if modo == "supabase"
                   else "Almacenamiento: **base local**")

df = loaders.load_master() if loaders.artefactos_listos() else None

tab_reg, tab_cons = st.tabs(["📋 Registros de vigilancia", "🤖 Consultas al modelo"])


# ===========================================================================
# TAB 1 — CRUD sobre los registros de vigilancia (distrito-semana)
# ===========================================================================
with tab_reg:
    if df is None:
        st.error("Faltan artefactos. Ejecuta `python train.py --rebuild`.")
        st.stop()

    st.markdown(
        "Las cuatro operaciones sobre la unidad de analisis del proyecto: "
        "**un distrito en una semana epidemiologica**. Los registros se importan "
        "del dataset y desde ahi se pueden consultar, editar, anadir y eliminar."
    )
    ui.nota(
        "El CSV maestro no se modifica nunca: es la fuente de origen y debe quedar "
        "intacta para que los modelos sigan siendo reproducibles. Las ediciones "
        "viven en la tabla <code>registros</code> de la base de datos."
    )

    # --- READ (del dataset) + importacion ---------------------------------
    ui.section("Consultar el dataset e importar", tag="read")
    c = st.columns([1.2, 1.5, 1, 1])
    dep = c[0].selectbox("Departamento", sorted(df["departamento"].unique()),
                         key="reg_dep")
    dist_df = loaders.distritos_de(df, dep)
    dist = c[1].selectbox("Distrito", dist_df["distrito"].tolist(), key="reg_dist")
    ubigeo = dist_df.loc[dist_df["distrito"] == dist, "ubigeo"].iloc[0]
    anos = sorted(df.loc[df["ubigeo"].astype(str) == str(ubigeo), "ano"].unique())
    ano = c[2].selectbox("Anio", anos, index=len(anos) - 1, key="reg_ano")
    limite = c[3].number_input("Maximo a importar", 5, 60, 15, key="reg_lim")

    seleccion = df[(df["ubigeo"].astype(str) == str(ubigeo)) & (df["ano"] == ano)]
    seleccion = seleccion.sort_values("semana")

    st.dataframe(
        seleccion[["departamento", "provincia", "distrito", "ano", "semana", "casos"]],
        width='stretch', hide_index=True, height=240)
    st.caption(f"{len(seleccion)} semanas de **{dist.title()}** en {ano}.")

    b = st.columns([1, 1, 2])
    if b[0].button(f"⬇️ Importar {int(limite)} registros", type="primary",
                   width='stretch'):
        filas = [
            {"ubigeo": str(r.ubigeo), "departamento": r.departamento,
             "provincia": r.provincia, "distrito": r.distrito,
             "ano": int(r.ano), "semana": int(r.semana), "casos": int(r.casos),
             "observacion": "importado del dataset"}
            for r in seleccion.head(int(limite)).itertuples()
        ]
        try:
            n = db_registros.create_many(filas)
            st.success(f"{n} registros importados.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudieron importar: {e}")
    if b[1].button("🗑️ Vaciar la tabla", width='stretch'):
        try:
            db_registros.delete_all()
            st.success("Tabla vaciada.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo vaciar: {e}")

    # --- CREATE -----------------------------------------------------------
    ui.section("Anadir un registro nuevo", tag="create")
    with st.form("crear_registro"):
        f1 = st.columns(4)
        n_dep = f1[0].text_input("Departamento", dep)
        n_dist = f1[1].text_input("Distrito", dist)
        n_ano = f1[2].number_input("Anio", 2000, 2030, int(ano))
        n_sem = f1[3].number_input("Semana", 1, 53, 1)
        f2 = st.columns([1, 3])
        n_casos = f2[0].number_input("Casos", 0, 100000, 0)
        n_obs = f2[1].text_input("Observacion", "registro manual")
        if st.form_submit_button("Crear registro", type="primary"):
            try:
                db_registros.create({
                    "ubigeo": str(ubigeo), "departamento": n_dep, "provincia": "",
                    "distrito": n_dist, "ano": int(n_ano), "semana": int(n_sem),
                    "casos": int(n_casos), "observacion": n_obs,
                })
                st.success("Registro creado.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo crear: {e}")

    # --- READ (de la tabla) -----------------------------------------------
    ui.section("Registros almacenados", tag="read")
    try:
        almacenados = db_registros.list_all()
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudieron listar: {e}")
        almacenados = []

    if not almacenados:
        st.info("Todavia no hay registros. Importa algunos del dataset o crea uno.")
    else:
        tabla_reg = pd.DataFrame(almacenados)
        cols = [c for c in ["id", "departamento", "distrito", "ano", "semana",
                            "casos", "observacion", "updated_at"] if c in tabla_reg.columns]
        st.dataframe(tabla_reg[cols], width='stretch', hide_index=True)

        m = st.columns(4)
        m[0].metric("Registros", len(tabla_reg))
        m[1].metric("Casos acumulados", f"{int(tabla_reg['casos'].sum()):,}")
        m[2].metric("Distritos", tabla_reg["distrito"].nunique())
        m[3].metric("Anios", tabla_reg["ano"].nunique())

        # --- UPDATE / DELETE ----------------------------------------------
        ui.section("Editar o eliminar", tag="update · delete")
        ids = tabla_reg["id"].tolist()
        id_sel = st.selectbox(
            "Registro", ids,
            format_func=lambda i: (
                f"#{i} · {tabla_reg.loc[tabla_reg['id'] == i, 'distrito'].iloc[0]} · "
                f"{tabla_reg.loc[tabla_reg['id'] == i, 'ano'].iloc[0]}"
                f"-S{tabla_reg.loc[tabla_reg['id'] == i, 'semana'].iloc[0]}"),
            key="reg_id")
        fila = tabla_reg[tabla_reg["id"] == id_sel].iloc[0]

        with st.form("editar_registro"):
            e1 = st.columns(4)
            e_dep = e1[0].text_input("Departamento", str(fila.get("departamento", "")))
            e_dist = e1[1].text_input("Distrito", str(fila.get("distrito", "")))
            e_ano = e1[2].number_input("Anio", 2000, 2030, int(fila.get("ano") or 2024))
            e_sem = e1[3].number_input("Semana", 1, 53, int(fila.get("semana") or 1))
            e2 = st.columns([1, 3])
            e_casos = e2[0].number_input("Casos", 0, 100000, int(fila.get("casos") or 0))
            e_obs = e2[1].text_input("Observacion", str(fila.get("observacion") or ""))
            bt = st.columns(2)
            guardar = bt[0].form_submit_button("Guardar cambios", type="primary")
            borrar = bt[1].form_submit_button("Eliminar registro")

        if guardar:
            try:
                db_registros.update(int(id_sel), {
                    "departamento": e_dep, "distrito": e_dist, "ano": int(e_ano),
                    "semana": int(e_sem), "casos": int(e_casos), "observacion": e_obs,
                })
                st.success(f"Registro {id_sel} actualizado.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo actualizar: {e}")

        if borrar:
            try:
                db_registros.delete(int(id_sel))
                st.success(f"Registro {id_sel} eliminado.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo eliminar: {e}")


# ===========================================================================
# TAB 2 — CRUD sobre las consultas al modelo
# ===========================================================================
with tab_cons:
    st.markdown("Predicciones guardadas desde **Modelo Predictivo**, con su "
                "escenario de entrada, el modelo usado y la probabilidad obtenida.")

    ui.section("Registrar una consulta", tag="create")
    prefill = st.session_state.get("ultima_prediccion", {})
    if prefill:
        st.caption("Prellenado con el ultimo escenario de **Modelo Predictivo**.")

    MODELOS = list(modeling.MODEL_LABELS)
    with st.form("crear_consulta"):
        c = st.columns(4)
        departamento = c[0].text_input("Departamento", prefill.get("departamento", ""))
        distrito = c[1].text_input("Distrito", prefill.get("distrito", ""))
        semana = c[2].number_input("Semana", 1, 53, int(prefill.get("semana", 1)))
        modelo = c[3].selectbox(
            "Modelo", MODELOS,
            index=MODELOS.index(prefill["modelo"]) if prefill.get("modelo") in MODELOS else 0,
            format_func=lambda k: modeling.MODEL_LABELS.get(k, k))
        c2 = st.columns(2)
        prediccion = c2[0].selectbox("Prediccion", ["alta", "baja"],
                                     index=0 if prefill.get("prediccion") != "baja" else 1)
        probabilidad = c2[1].number_input("Probabilidad", 0.0, 1.0,
                                          float(prefill.get("probabilidad", 0.5)), 0.01)
        if st.form_submit_button("Registrar consulta", type="primary"):
            try:
                db_consultas.create({
                    "departamento": departamento, "distrito": distrito,
                    "semana": int(semana),
                    "datos_entrada": prefill.get("datos_entrada", {"semana": int(semana)}),
                    "modelo": modelo, "prediccion": prediccion,
                    "probabilidad": float(probabilidad),
                })
                st.success("Consulta registrada.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo registrar: {e}")

    ui.section("Consultas registradas", tag="read")
    try:
        registros = db_consultas.list_all()
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudieron listar: {e}")
        registros = []

    if not registros:
        st.info("Aun no hay consultas registradas.")
    else:
        tabla = pd.DataFrame(registros)
        cols_mostrar = [c for c in ["id", "departamento", "distrito", "semana", "modelo",
                                    "prediccion", "probabilidad", "updated_at"]
                        if c in tabla.columns]
        st.dataframe(tabla[cols_mostrar], width='stretch', hide_index=True)

        ui.section("Editar o eliminar", tag="update · delete")
        ids = tabla["id"].tolist()
        id_sel = st.selectbox(
            "Consulta", ids,
            format_func=lambda i: (
                f"#{i} · {tabla.loc[tabla['id'] == i, 'distrito'].iloc[0]} · "
                f"{tabla.loc[tabla['id'] == i, 'prediccion'].iloc[0]}"),
            key="cons_id")
        fila = tabla[tabla["id"] == id_sel].iloc[0]

        with st.form("editar_consulta"):
            c = st.columns(4)
            e_dep = c[0].text_input("Departamento", str(fila.get("departamento", "")))
            e_dist = c[1].text_input("Distrito", str(fila.get("distrito", "")))
            e_sem = c[2].number_input("Semana", 1, 53, int(fila.get("semana", 1) or 1))
            e_pred = c[3].selectbox("Prediccion", ["alta", "baja"],
                                    index=0 if fila.get("prediccion") != "baja" else 1)
            e_prob = st.number_input("Probabilidad", 0.0, 1.0,
                                     float(fila.get("probabilidad", 0.5) or 0.5), 0.01)
            bt = st.columns(2)
            editar = bt[0].form_submit_button("Guardar cambios", type="primary")
            eliminar = bt[1].form_submit_button("Eliminar consulta")

        if editar:
            try:
                db_consultas.update(int(id_sel), {
                    "departamento": e_dep, "distrito": e_dist, "semana": int(e_sem),
                    "prediccion": e_pred, "probabilidad": float(e_prob),
                })
                st.success(f"Consulta {id_sel} actualizada.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo actualizar: {e}")

        if eliminar:
            try:
                db_consultas.delete(int(id_sel))
                st.success(f"Consulta {id_sel} eliminada.")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo eliminar: {e}")
