"""Capa de persistencia del CRUD.

Desacoplada del backend: usa Supabase si hay credenciales y, si no, una base
SQLite local. La interfaz publica es identica en ambos modos, de modo que la
pagina de Streamlit no necesita saber cual esta activo.

Gestiona dos tablas con el mismo contrato (create / list_all / update / delete):

  - ``consultas``  : predicciones guardadas desde el panel del modelo.
  - ``registros``  : registros de vigilancia distrito-semana. Se pueden importar
                     del dataset y luego crear, leer, editar y eliminar. El CSV
                     maestro NUNCA se modifica: es la fuente de origen y debe
                     permanecer intacto para que los modelos sean reproducibles.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

TABLA_CONSULTAS = "consultas"
TABLA_REGISTROS = "registros"
TABLA = TABLA_CONSULTAS  # alias historico
SQLITE_PATH = config.BASE_DIR / "data" / "consultas_local.db"

# Esquema de cada tabla en el backend local (Supabase se crea desde su consola)
ESQUEMAS = {
    TABLA_CONSULTAS: """
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        departamento TEXT,
        distrito TEXT,
        semana INTEGER,
        datos_entrada TEXT,
        modelo TEXT,
        prediccion TEXT,
        probabilidad REAL,
        created_at TEXT,
        updated_at TEXT
    """,
    TABLA_REGISTROS: """
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ubigeo TEXT,
        departamento TEXT,
        provincia TEXT,
        distrito TEXT,
        ano INTEGER,
        semana INTEGER,
        casos INTEGER,
        observacion TEXT,
        created_at TEXT,
        updated_at TEXT
    """,
}

# Columnas que acepta cada tabla (evita insertar claves inexistentes)
COLUMNAS = {
    tabla: [linea.strip().split()[0] for linea in esquema.strip().splitlines()]
    for tabla, esquema in ESQUEMAS.items()
}

# Campos que se guardan serializados como JSON
CAMPOS_JSON = {"datos_entrada"}


# ---------------------------------------------------------------------------
# Deteccion de credenciales
# ---------------------------------------------------------------------------
def _leer_credenciales():
    """Busca SUPABASE_URL/KEY en st.secrets, luego en variables de entorno."""
    url = key = None
    try:
        import streamlit as st

        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
    except Exception:
        pass
    if not url or not key:
        import os

        url = url or os.environ.get("SUPABASE_URL")
        key = key or os.environ.get("SUPABASE_KEY")
    return url, key


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _filtrar_columnas(tabla: str, registro: dict) -> dict:
    """Descarta claves que no existen en la tabla destino."""
    validas = set(COLUMNAS.get(tabla, [])) - {"id", "created_at", "updated_at"}
    return {k: v for k, v in registro.items() if k in validas}


# ---------------------------------------------------------------------------
# Backend Supabase
# ---------------------------------------------------------------------------
class _SupabaseBackend:
    modo = "supabase"

    def __init__(self, url, key, tabla: str = TABLA_CONSULTAS):
        from supabase import create_client

        self.client = create_client(url, key)
        self.tabla = tabla

    def create(self, registro: dict) -> dict:
        r = self.client.table(self.tabla).insert(_filtrar_columnas(self.tabla, registro)).execute()
        return r.data[0] if r.data else {}

    def create_many(self, registros: list[dict]) -> int:
        if not registros:
            return 0
        limpios = [_filtrar_columnas(self.tabla, r) for r in registros]
        r = self.client.table(self.tabla).insert(limpios).execute()
        return len(r.data or [])

    def list_all(self) -> list[dict]:
        r = self.client.table(self.tabla).select("*").order("created_at", desc=True).execute()
        return r.data or []

    def update(self, id_: int, cambios: dict) -> dict:
        cambios = {**_filtrar_columnas(self.tabla, cambios), "updated_at": _ahora()}
        r = self.client.table(self.tabla).update(cambios).eq("id", id_).execute()
        return r.data[0] if r.data else {}

    def delete(self, id_: int) -> None:
        self.client.table(self.tabla).delete().eq("id", id_).execute()

    def delete_all(self) -> None:
        self.client.table(self.tabla).delete().neq("id", -1).execute()


# ---------------------------------------------------------------------------
# Backend local (SQLite)
# ---------------------------------------------------------------------------
class _SQLiteBackend:
    modo = "local"

    def __init__(self, path: Path = SQLITE_PATH, tabla: str = TABLA_CONSULTAS):
        self.path = Path(path)
        self.tabla = tabla
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._crear_tabla()

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _crear_tabla(self):
        with self._conn() as c:
            c.execute(f"CREATE TABLE IF NOT EXISTS {self.tabla} ({ESQUEMAS[self.tabla]})")

    @staticmethod
    def _fila_a_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for campo in CAMPOS_JSON & set(d):
            if isinstance(d[campo], str):
                try:
                    d[campo] = json.loads(d[campo])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    @staticmethod
    def _serializar(registro: dict) -> dict:
        r = dict(registro)
        for campo in CAMPOS_JSON & set(r):
            if not isinstance(r[campo], str):
                r[campo] = json.dumps(r[campo], ensure_ascii=False)
        return r

    def create(self, registro: dict) -> dict:
        datos = self._serializar(_filtrar_columnas(self.tabla, registro))
        ahora = _ahora()
        campos = list(datos) + ["created_at", "updated_at"]
        marcas = ",".join("?" * len(campos))
        with self._conn() as c:
            cur = c.execute(
                f"INSERT INTO {self.tabla} ({', '.join(campos)}) VALUES ({marcas})",
                (*datos.values(), ahora, ahora),
            )
            row = c.execute(f"SELECT * FROM {self.tabla} WHERE id=?",
                            (cur.lastrowid,)).fetchone()
        return self._fila_a_dict(row)

    def create_many(self, registros: list[dict]) -> int:
        """Insercion masiva: usada al importar registros del dataset."""
        if not registros:
            return 0
        ahora = _ahora()
        limpios = [self._serializar(_filtrar_columnas(self.tabla, r)) for r in registros]
        campos = list(limpios[0]) + ["created_at", "updated_at"]
        marcas = ",".join("?" * len(campos))
        filas = [(*r.values(), ahora, ahora) for r in limpios]
        with self._conn() as c:
            c.executemany(
                f"INSERT INTO {self.tabla} ({', '.join(campos)}) VALUES ({marcas})", filas)
        return len(filas)

    def list_all(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM {self.tabla} ORDER BY created_at DESC, id DESC").fetchall()
        return [self._fila_a_dict(r) for r in rows]

    def update(self, id_: int, cambios: dict) -> dict:
        cambios = self._serializar(_filtrar_columnas(self.tabla, cambios))
        cambios["updated_at"] = _ahora()
        columnas = ", ".join(f"{k}=?" for k in cambios)
        with self._conn() as c:
            c.execute(f"UPDATE {self.tabla} SET {columnas} WHERE id=?",
                      (*cambios.values(), id_))
            row = c.execute(f"SELECT * FROM {self.tabla} WHERE id=?", (id_,)).fetchone()
        return self._fila_a_dict(row) if row else {}

    def delete(self, id_: int) -> None:
        with self._conn() as c:
            c.execute(f"DELETE FROM {self.tabla} WHERE id=?", (id_,))

    def delete_all(self) -> None:
        with self._conn() as c:
            c.execute(f"DELETE FROM {self.tabla}")


# ---------------------------------------------------------------------------
# Fabrica
# ---------------------------------------------------------------------------
def get_db(tabla: str = TABLA_CONSULTAS):
    """Backend para la tabla indicada (Supabase si hay credenciales, si no local)."""
    url, key = _leer_credenciales()
    if url and key:
        try:
            return _SupabaseBackend(url, key, tabla)
        except Exception as e:  # noqa: BLE001
            print(f"[database] No se pudo conectar a Supabase ({e}); usando modo local.")
    return _SQLiteBackend(tabla=tabla)
