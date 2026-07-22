"""Pruebas del CRUD: las cuatro operaciones sobre ambas tablas.

Se ejecutan contra el backend local en una base temporal, sin tocar la del
proyecto ni necesitar credenciales de Supabase.
"""

from __future__ import annotations

import pytest

from src import database


@pytest.fixture
def db_registros(tmp_path):
    return database._SQLiteBackend(path=tmp_path / "t.db",
                                   tabla=database.TABLA_REGISTROS)


@pytest.fixture
def db_consultas(tmp_path):
    return database._SQLiteBackend(path=tmp_path / "t.db",
                                   tabla=database.TABLA_CONSULTAS)


REGISTRO = {
    "ubigeo": "010101", "departamento": "PIURA", "provincia": "SULLANA",
    "distrito": "BELLAVISTA", "ano": 2026, "semana": 10, "casos": 42,
    "observacion": "importado del dataset",
}


# ---------------------------------------------------------------------------
# Registros de vigilancia
# ---------------------------------------------------------------------------
def test_create_devuelve_el_registro_con_id(db_registros):
    creado = db_registros.create(REGISTRO)
    assert creado["id"] > 0
    assert creado["casos"] == 42
    assert creado["created_at"] and creado["updated_at"]


def test_read_lista_lo_creado(db_registros):
    db_registros.create(REGISTRO)
    db_registros.create({**REGISTRO, "semana": 11, "casos": 7})
    filas = db_registros.list_all()
    assert len(filas) == 2
    assert sum(f["casos"] for f in filas) == 49


def test_update_modifica_solo_el_registro_indicado(db_registros):
    a = db_registros.create(REGISTRO)
    b = db_registros.create({**REGISTRO, "semana": 11, "casos": 7})
    db_registros.update(a["id"], {"casos": 100, "observacion": "corregido"})

    por_id = {f["id"]: f for f in db_registros.list_all()}
    assert por_id[a["id"]]["casos"] == 100
    assert por_id[a["id"]]["observacion"] == "corregido"
    assert por_id[b["id"]]["casos"] == 7, "no debe tocar los demas registros"


def test_delete_elimina_solo_ese_registro(db_registros):
    a = db_registros.create(REGISTRO)
    db_registros.create({**REGISTRO, "semana": 11})
    db_registros.delete(a["id"])
    restantes = db_registros.list_all()
    assert len(restantes) == 1
    assert restantes[0]["semana"] == 11


def test_create_many_inserta_en_bloque(db_registros):
    filas = [{**REGISTRO, "semana": s, "casos": s * 2} for s in range(1, 16)]
    assert db_registros.create_many(filas) == 15
    assert len(db_registros.list_all()) == 15


def test_create_many_vacio_no_falla(db_registros):
    assert db_registros.create_many([]) == 0


def test_delete_all_vacia_la_tabla(db_registros):
    db_registros.create_many([{**REGISTRO, "semana": s} for s in range(1, 6)])
    db_registros.delete_all()
    assert db_registros.list_all() == []


def test_se_ignoran_columnas_inexistentes(db_registros):
    """Una clave que no existe en la tabla no debe romper la insercion."""
    creado = db_registros.create({**REGISTRO, "columna_inventada": "x"})
    assert creado["casos"] == 42
    assert "columna_inventada" not in creado


# ---------------------------------------------------------------------------
# Consultas al modelo
# ---------------------------------------------------------------------------
def test_datos_entrada_sobrevive_como_diccionario(db_consultas):
    """El escenario del simulador se guarda como JSON y vuelve como dict."""
    entrada = {"casos": 7.0, "casos_lag_1": 3.0, "semana": 5}
    db_consultas.create({
        "departamento": "PIURA", "distrito": "BELLAVISTA", "semana": 5,
        "datos_entrada": entrada, "modelo": "xgboost",
        "prediccion": "alta", "probabilidad": 0.83,
    })
    guardado = db_consultas.list_all()[0]
    assert guardado["datos_entrada"] == entrada
    assert isinstance(guardado["datos_entrada"], dict)


def test_las_dos_tablas_conviven_en_la_misma_base(tmp_path):
    ruta = tmp_path / "t.db"
    reg = database._SQLiteBackend(path=ruta, tabla=database.TABLA_REGISTROS)
    con = database._SQLiteBackend(path=ruta, tabla=database.TABLA_CONSULTAS)
    reg.create(REGISTRO)
    con.create({"departamento": "PIURA", "distrito": "B", "semana": 1,
                "datos_entrada": {}, "modelo": "rf", "prediccion": "baja",
                "probabilidad": 0.1})
    assert len(reg.list_all()) == 1
    assert len(con.list_all()) == 1
