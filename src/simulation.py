"""Extension SIMULADA del dataset hasta la actualidad (2025 - mayo 2026).

La vigilancia del MINSA publicada llega hasta 2024. Para que el tablero muestre
pronosticos con fechas vigentes, este modulo genera registros distrito-semana
para el periodo posterior mediante un **bootstrap estacional por distrito**.

    NO SON DATOS REALES.

Cada fila generada queda marcada con origen = "simulado" y con split =
"simulado", de modo que:
  - las metricas de clasificacion se siguen calculando SOLO con datos reales
    (train <= 2022, val = 2023, test = 2024);
  - el clustering se sigue construyendo SOLO con datos reales;
  - unicamente el pronostico y la exploracion temporal usan la extension.

Metodo (bootstrap estacional con recencia y factor de intensidad):
  1. Para cada distrito activo en los ultimos anios reales, y para cada semana
     epidemiologica objetivo w, se toma el conjunto de casos historicos de ese
     mismo distrito en las semanas [w - V, w + V] (ventana circular).
  2. Se muestrea un valor de ese conjunto con probabilidad decreciente segun la
     antiguedad del anio (los anios recientes pesan mas).
  3. El valor se multiplica por un factor de intensidad anual y por un ruido
     multiplicativo lognormal, y se redondea a entero no negativo.

Esto conserva la estacionalidad y la magnitud tipica de cada distrito sin
inventar una tendencia arbitraria. Todo es reproducible (config.RANDOM_STATE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import preprocessing

# Columnas base (antes de la ingenieria de caracteristicas)
COLS_BASE = [
    "ubigeo", "departamento", "provincia", "distrito", "ano", "semana",
    "fecha", "week_id", "casos", "edad_promedio", "prop_con_signos",
    "prop_grave", "prop_femenino",
]

AUX_COLS = ["edad_promedio", "prop_con_signos", "prop_grave", "prop_femenino"]

SEMANAS_POR_ANIO = 52


# ---------------------------------------------------------------------------
# Calendario objetivo
# ---------------------------------------------------------------------------
def _distancia_circular(semanas: np.ndarray, w: int) -> np.ndarray:
    """Distancia entre semanas epidemiologicas tratando el anio como un ciclo."""
    d = np.abs(semanas - w)
    return np.minimum(d, SEMANAS_POR_ANIO - d)


def calendario_objetivo(master: pd.DataFrame) -> pd.DataFrame:
    """Grid (ano, semana, fecha, week_id) desde el final de los datos reales
    hasta config.SIM_END_ANO / config.SIM_END_SEMANA."""
    ultimo_ano = int(master["ano"].max())
    ultima_semana = int(master.loc[master["ano"] == ultimo_ano, "semana"].max())
    max_week_id = int(master["week_id"].max())

    pares: list[tuple[int, int]] = []
    ano, semana = ultimo_ano, ultima_semana
    while (ano, semana) < (config.SIM_END_ANO, config.SIM_END_SEMANA):
        semana += 1
        if semana > SEMANAS_POR_ANIO:
            ano, semana = ano + 1, 1
        pares.append((ano, semana))

    cal = pd.DataFrame(pares, columns=["ano", "semana"])
    cal["week_id"] = np.arange(max_week_id + 1, max_week_id + 1 + len(cal))
    cal["fecha"] = pd.to_datetime(
        [preprocessing._epi_week_to_date(a, s) for a, s in zip(cal["ano"], cal["semana"])]
    )
    return cal


# ---------------------------------------------------------------------------
# Bootstrap estacional por distrito
# ---------------------------------------------------------------------------
def _distritos_activos(master: pd.DataFrame) -> set[str]:
    """Distritos con notificaciones en los ultimos anios reales.

    Un distrito que dejo de reportar hace anios no se 'reactiva': solo se
    extienden los que seguian activos al final del periodo real.
    """
    ultimo_ano = int(master["ano"].max())
    corte = ultimo_ano - config.SIM_ANIOS_ACTIVIDAD + 1
    return set(master.loc[master["ano"] >= corte, "ubigeo"].unique())


def simular_filas(master: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Genera las filas distrito-semana simuladas (columnas base)."""
    cal = calendario_objetivo(master)
    if cal.empty:
        return pd.DataFrame(columns=COLS_BASE + ["origen"])

    activos = _distritos_activos(master)
    ultimo_ano = int(master["ano"].max())
    ano_corte_pool = ultimo_ano - config.SIM_LOOKBACK_ANIOS + 1

    rng = np.random.default_rng(config.RANDOM_STATE)
    partes: list[pd.DataFrame] = []

    for ubigeo, sub in master[master["ubigeo"].isin(activos)].groupby("ubigeo", sort=True):
        sub = sub.sort_values("week_id")
        ultimo_week_id = int(sub["week_id"].max())

        # Semanas a generar para este distrito: desde la siguiente a su ultimo
        # registro (asi se cierra tambien el hueco de los distritos que dejaron
        # de reportar antes del cierre del anio) hasta el fin del horizonte.
        objetivo = cal[cal["week_id"] > ultimo_week_id]
        if objetivo.empty:
            continue

        pool = sub[sub["ano"] >= ano_corte_pool]
        if pool.empty:
            pool = sub
        p_semanas = pool["semana"].to_numpy()
        p_casos = pool["casos"].to_numpy(dtype=float)
        p_anios = pool["ano"].to_numpy(dtype=float)
        p_aux = pool[AUX_COLS].to_numpy(dtype=float)
        # Peso por recencia: los anios recientes son mas representativos
        p_peso = config.SIM_DECAIMIENTO ** (ultimo_ano - p_anios)

        # --- 1) Nivel estacional base: un muestreo por semana objetivo --------
        bases, auxs = [], []
        for ano, semana in objetivo[["ano", "semana"]].itertuples(index=False):
            mask = _distancia_circular(p_semanas, semana) <= config.SIM_VENTANA_SEMANAS
            if not mask.any():
                mask = np.ones(len(p_semanas), dtype=bool)

            pesos = p_peso[mask]
            total = pesos.sum()
            probas = pesos / total if total > 0 else None
            idx = rng.choice(np.flatnonzero(mask), p=probas)

            bases.append(p_casos[idx] * config.SIM_INTENSIDAD.get(int(ano), 1.0))
            auxs.append(p_aux[idx])

        base = pd.Series(bases, dtype=float)
        # Una curva epidemica real es suave: se promedia el muestreo para quitar
        # el salto artificial semana a semana que introduce el bootstrap.
        base = base.rolling(config.SIM_SUAVIZADO, center=True, min_periods=1).mean()

        # --- 2) Intensidad persistente: paseo AR(1) en escala logaritmica -----
        # La intensidad de un brote persiste varias semanas; un ruido
        # independiente por semana no lo representa.
        rho = config.SIM_PERSISTENCIA
        sigma = config.SIM_RUIDO_SIGMA
        eps = rng.normal(0.0, sigma * np.sqrt(1 - rho ** 2), size=len(base))
        log_f = np.empty(len(base))
        anterior = rng.normal(0.0, sigma)
        for i, e in enumerate(eps):
            anterior = rho * anterior + e
            log_f[i] = anterior

        casos = np.maximum(0, np.rint(base.to_numpy() * np.exp(log_f))).astype(int)

        parte = objetivo[["ano", "semana", "week_id", "fecha"]].reset_index(drop=True)
        parte.insert(0, "ubigeo", ubigeo)
        parte["casos"] = casos
        for j, col in enumerate(AUX_COLS):
            parte[col] = [a[j] for a in auxs]
        for col in ["departamento", "provincia", "distrito"]:
            parte[col] = sub[col].iloc[0]
        partes.append(parte)

    if not partes:
        return pd.DataFrame(columns=COLS_BASE + ["origen"])

    out = pd.concat(partes, ignore_index=True)
    out["origen"] = "simulado"
    # Sin casos no hay perfil de pacientes que reportar
    out.loc[out["casos"] == 0, AUX_COLS] = np.nan
    if verbose:
        print(f"      filas simuladas: {len(out):,} "
              f"({out['ubigeo'].nunique()} distritos, "
              f"hasta {config.SIM_END_ANO}-S{config.SIM_END_SEMANA})")
    return out[COLS_BASE + ["origen"]]


# ---------------------------------------------------------------------------
# Extension del maestro
# ---------------------------------------------------------------------------
def extender_master(master: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Devuelve el maestro con la extension simulada y las features recalculadas.

    Las columnas derivadas (lags, medias moviles, objetivo, split) se recalculan
    sobre la serie completa para que no queden discontinuidades en el empalme.
    """
    def log(*a):
        if verbose:
            print(*a)

    master = master.copy()
    master["fecha"] = pd.to_datetime(master["fecha"])
    if "origen" not in master.columns:
        master["origen"] = "real"
    real = master[master["origen"] == "real"][COLS_BASE + ["origen"]]

    log("      generando extension estacional...")
    simulado = simular_filas(real, verbose=verbose)
    if simulado.empty:
        log("      nada que simular: el maestro ya cubre el horizonte.")
        return master

    completo = pd.concat([real, simulado], ignore_index=True)
    completo = completo.sort_values(["ubigeo", "week_id"]).reset_index(drop=True)

    log("      recalculando features y variable objetivo...")
    conf = preprocessing.add_features(completo)
    final = preprocessing.add_target_and_split(conf)

    cols = (
        ["ubigeo", "departamento", "provincia", "distrito", "ano", "semana",
         "fecha", "week_id", "casos", "edad_promedio", "prop_con_signos",
         "prop_grave", "prop_femenino"]
        + [c for c in preprocessing.FEATURE_NUMERIC if c != "casos"]
        + ["umbral_incidencia", "split", preprocessing.TARGET, "origen"]
    )
    return final[cols]


def resumen(df: pd.DataFrame) -> dict:
    """Cifras de la extension, para mostrarlas en el dashboard."""
    if "origen" not in df.columns:
        return {"tiene_simulacion": False}
    sim = df[df["origen"] == "simulado"]
    real = df[df["origen"] == "real"]
    if sim.empty:
        return {"tiene_simulacion": False}
    return {
        "tiene_simulacion": True,
        "filas_simuladas": len(sim),
        "filas_reales": len(real),
        "distritos": int(sim["ubigeo"].nunique()),
        "casos_simulados": int(sim["casos"].sum()),
        "desde": pd.to_datetime(sim["fecha"]).min(),
        "hasta": pd.to_datetime(sim["fecha"]).max(),
        "ultimo_ano_real": int(real["ano"].max()),
    }
