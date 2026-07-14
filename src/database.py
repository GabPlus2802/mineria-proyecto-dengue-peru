"""Capa de persistencia para el CRUD de consultas (Panel 4).

Desacoplada: usa Supabase si hay credenciales; si no, cae a una base SQLite
local para desarrollo. La interfaz publica es la misma en ambos modos, de modo
que la pagina de Streamlit no necesita saber cual esta activo.

IMPORTANTE: el despliegue final debe configurar Supabase (persistencia real).
El modo local es solo para desarrollo; no es el CRUD definitivo.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

TABLA = "consultas"
SQLITE_PATH = config.BASE_DIR / "data" / "consultas_local.db"


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


# ---------------------------------------------------------------------------
# Backend Supabase
# ---------------------------------------------------------------------------
class _SupabaseBackend:
    modo = "supabase"

    def __init__(self, url, key):
        from supabase import create_client

        self.client = create_client(url, key)

    def create(self, registro: dict) -> dict:
        r = self.client.table(TABLA).insert(registro).execute()
        return r.data[0] if r.data else {}

    def list_all(self) -> list[dict]:
        r = self.client.table(TABLA).select("*").order("created_at", desc=True).execute()
        return r.data or []

    def update(self, id_: int, cambios: dict) -> dict:
        cambios = {**cambios, "updated_at": datetime.now(timezone.utc).isoformat()}
        r = self.client.table(TABLA).update(cambios).eq("id", id_).execute()
        return r.data[0] if r.data else {}

    def delete(self, id_: int) -> None:
        self.client.table(TABLA).delete().eq("id", id_).execute()


# ---------------------------------------------------------------------------
# Backend local (SQLite)
# ---------------------------------------------------------------------------
class _SQLiteBackend:
    modo = "local"

    def __init__(self, path: Path = SQLITE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._crear_tabla()

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _crear_tabla(self):
        with self._conn() as c:
            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLA} (
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
                )
                """
            )

    @staticmethod
    def _fila_a_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        if isinstance(d.get("datos_entrada"), str):
            try:
                d["datos_entrada"] = json.loads(d["datos_entrada"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def create(self, registro: dict) -> dict:
        ahora = datetime.now(timezone.utc).isoformat()
        datos = registro.get("datos_entrada")
        if not isinstance(datos, str):
            datos = json.dumps(datos, ensure_ascii=False)
        with self._conn() as c:
            cur = c.execute(
                f"""INSERT INTO {TABLA}
                    (departamento, distrito, semana, datos_entrada, modelo,
                     prediccion, probabilidad, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    registro.get("departamento"),
                    registro.get("distrito"),
                    registro.get("semana"),
                    datos,
                    registro.get("modelo"),
                    registro.get("prediccion"),
                    registro.get("probabilidad"),
                    ahora,
                    ahora,
                ),
            )
            new_id = cur.lastrowid
            row = c.execute(f"SELECT * FROM {TABLA} WHERE id=?", (new_id,)).fetchone()
        return self._fila_a_dict(row)

    def list_all(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM {TABLA} ORDER BY created_at DESC").fetchall()
        return [self._fila_a_dict(r) for r in rows]

    def update(self, id_: int, cambios: dict) -> dict:
        cambios = dict(cambios)
        if "datos_entrada" in cambios and not isinstance(cambios["datos_entrada"], str):
            cambios["datos_entrada"] = json.dumps(cambios["datos_entrada"], ensure_ascii=False)
        cambios["updated_at"] = datetime.now(timezone.utc).isoformat()
        columnas = ", ".join(f"{k}=?" for k in cambios)
        with self._conn() as c:
            c.execute(f"UPDATE {TABLA} SET {columnas} WHERE id=?",
                      (*cambios.values(), id_))
            row = c.execute(f"SELECT * FROM {TABLA} WHERE id=?", (id_,)).fetchone()
        return self._fila_a_dict(row) if row else {}

    def delete(self, id_: int) -> None:
        with self._conn() as c:
            c.execute(f"DELETE FROM {TABLA} WHERE id=?", (id_,))


# ---------------------------------------------------------------------------
# Fabrica
# ---------------------------------------------------------------------------
def get_db():
    """Devuelve el backend disponible (Supabase si hay credenciales, si no local)."""
    url, key = _leer_credenciales()
    if url and key:
        try:
            return _SupabaseBackend(url, key)
        except Exception as e:  # noqa: BLE001
            print(f"[database] No se pudo conectar a Supabase ({e}); usando modo local.")
    return _SQLiteBackend()
