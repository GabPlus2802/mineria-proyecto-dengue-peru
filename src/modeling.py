"""Clasificacion de alta incidencia de dengue en la semana siguiente.

Compara Random Forest y XGBoost sobre la unidad distrito-semana, con un
pipeline de preprocesamiento que se ajusta solo con entrenamiento. Incluye
metricas completas, barrido de umbral y utilidades para SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config
from src.preprocessing import FEATURE_CATEGORICAL, FEATURE_NUMERIC, TARGET

# Nombres legibles y orden de presentacion de los modelos
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting",
    "logistic_regression": "Regresion Logistica",
    "decision_tree": "Arbol de Decision",
}
# Modelos con soporte fiable de shap.TreeExplainer (HistGradientBoosting no se incluye)
MODELOS_ARBOL = {"random_forest", "xgboost", "decision_tree"}


# ---------------------------------------------------------------------------
# Preparacion de datos
# ---------------------------------------------------------------------------
def get_modeling_frame(df: pd.DataFrame):
    """Divide el dataset maestro en train/val/test segun la columna 'split'.

    Devuelve un dict con X_/y_ por particion. Descarta filas sin objetivo.
    """
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    cols = FEATURE_NUMERIC + FEATURE_CATEGORICAL

    out = {}
    for parte in ["train", "val", "test"]:
        sub = df[df["split"] == parte]
        out[f"X_{parte}"] = sub[cols].copy()
        out[f"y_{parte}"] = sub[TARGET].copy()
    return out


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: imputa+escala numericas, OneHot para categoricas.

    handle_unknown='ignore' permite departamentos no vistos en produccion.
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        [
            ("num", numeric, FEATURE_NUMERIC),
            ("cat", categorical, FEATURE_CATEGORICAL),
        ]
    )


def _imbalance_ratio(y) -> float:
    """Proporcion de la clase mayoritaria (0..1)."""
    vc = pd.Series(y).value_counts(normalize=True)
    return float(vc.max())


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def build_random_forest(balanced: bool) -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=getattr(config, "RF_MIN_SAMPLES_LEAF", 1),
        class_weight="balanced" if balanced else None,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def build_xgboost(scale_pos_weight: float) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        learning_rate=config.XGB_LEARNING_RATE,
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def build_gradient_boosting(balanced: bool) -> Pipeline:
    clf = HistGradientBoostingClassifier(
        max_iter=config.XGB_N_ESTIMATORS,
        learning_rate=config.XGB_LEARNING_RATE,
        class_weight="balanced" if balanced else None,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def build_logistic_regression(balanced: bool) -> Pipeline:
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced" if balanced else None,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def build_decision_tree(balanced: bool) -> Pipeline:
    clf = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=getattr(config, "RF_MIN_SAMPLES_LEAF", 20),
        class_weight="balanced" if balanced else None,
        random_state=config.RANDOM_STATE,
    )
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def train_models(data: dict):
    """Entrena 5 modelos (lineal, arbol simple y 3 ensembles).

    Aplica balanceo de clases si el desbalance supera 80/20.
    """
    y_train = data["y_train"]
    mayoria = _imbalance_ratio(y_train)
    balanced = mayoria > 0.80
    n_neg = int((y_train == 0).sum())
    n_pos = max(int((y_train == 1).sum()), 1)
    spw = n_neg / n_pos if balanced else 1.0

    modelos = {
        "random_forest": build_random_forest(balanced),
        "xgboost": build_xgboost(spw),
        "gradient_boosting": build_gradient_boosting(balanced),
        "logistic_regression": build_logistic_regression(balanced),
        "decision_tree": build_decision_tree(balanced),
    }
    for pipe in modelos.values():
        pipe.fit(data["X_train"], y_train)

    info = {"clase_mayoritaria": round(mayoria, 3), "balanceo_aplicado": balanced,
            "scale_pos_weight": round(spw, 2)}
    return modelos, info


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def evaluate(model: Pipeline, X, y, threshold=None) -> dict:
    """Metricas de clasificacion para un umbral dado (por defecto config)."""
    threshold = threshold if threshold is not None else config.CLASSIFICATION_THRESHOLD
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def metrics_table(models: dict, X, y, threshold=None) -> pd.DataFrame:
    filas = []
    for nombre, modelo in models.items():
        m = evaluate(modelo, X, y, threshold)
        m = {"modelo": nombre, **m}
        filas.append(m)
    df = pd.DataFrame(filas).set_index("modelo")
    return df.round(4)


def per_class_metrics(model: Pipeline, X, y, threshold=None) -> pd.DataFrame:
    """Precision, recall, F1 y soporte POR CLASE (0 = baja, 1 = alta incidencia)."""
    from sklearn.metrics import classification_report

    threshold = threshold if threshold is not None else config.CLASSIFICATION_THRESHOLD
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    rep = classification_report(y, pred, labels=[0, 1],
                                target_names=["Clase 0 (baja)", "Clase 1 (alta)"],
                                output_dict=True, zero_division=0)
    filas = {}
    for clase in ["Clase 0 (baja)", "Clase 1 (alta)"]:
        filas[clase] = {
            "precision": rep[clase]["precision"],
            "recall": rep[clase]["recall"],
            "f1-score": rep[clase]["f1-score"],
            "soporte": int(rep[clase]["support"]),
        }
    df = pd.DataFrame(filas).T
    df["soporte"] = df["soporte"].astype(int)
    return df.round(4)


def comparar_balanceo(data: dict, threshold=None) -> pd.DataFrame:
    """Efecto del balanceo de clases en el recall de la clase minoritaria (clase 1).

    Entrena cada modelo SIN balanceo y CON balanceo, y reporta recall y F1 de la
    clase 1 sobre el conjunto de prueba. Cumple el requisito de la rubrica de
    mostrar el efecto del balanceo cuando hay desbalance > 80/20.
    """
    threshold = threshold if threshold is not None else config.CLASSIFICATION_THRESHOLD
    y_train = data["y_train"]
    n_neg, n_pos = int((y_train == 0).sum()), max(int((y_train == 1).sum()), 1)
    spw = n_neg / n_pos

    variantes = {
        ("random_forest", "sin balanceo"): build_random_forest(balanced=False),
        ("random_forest", "con balanceo"): build_random_forest(balanced=True),
        ("xgboost", "sin balanceo"): build_xgboost(scale_pos_weight=1.0),
        ("xgboost", "con balanceo"): build_xgboost(scale_pos_weight=spw),
    }
    filas = []
    for (modelo, estado), pipe in variantes.items():
        pipe.fit(data["X_train"], y_train)
        m = evaluate(pipe, data["X_test"], data["y_test"], threshold)
        filas.append({"modelo": modelo, "estado": estado,
                      "recall_clase_1": round(m["recall"], 4),
                      "f1_clase_1": round(m["f1"], 4)})
    return pd.DataFrame(filas)


def threshold_sweep(model: Pipeline, X, y, umbrales=None) -> pd.DataFrame:
    """Efecto del umbral de decision sobre precision/recall/F1 y FP/FN."""
    if umbrales is None:
        umbrales = np.round(np.arange(0.1, 0.91, 0.1), 2)
    proba = model.predict_proba(X)[:, 1]
    filas = []
    for t in umbrales:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        filas.append({
            "umbral": t,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
        })
    return pd.DataFrame(filas).round(4)


def elegir_mejor_modelo(tabla: pd.DataFrame) -> str:
    """Elige por F1 (equilibrio recall/precision), no solo por accuracy."""
    return tabla["f1"].astype(float).idxmax()


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Nombres de las columnas tras el ColumnTransformer (para SHAP)."""
    prep = pipeline.named_steps["prep"]
    return list(prep.get_feature_names_out())


def transform_X(pipeline: Pipeline, X) -> np.ndarray:
    return pipeline.named_steps["prep"].transform(X)


def shap_explainer(pipeline: Pipeline):
    """TreeExplainer sobre el clasificador de arboles del pipeline."""
    import shap

    return shap.TreeExplainer(pipeline.named_steps["clf"])


# ---------------------------------------------------------------------------
# Construccion de features para una prediccion desde el navegador (Panel 2)
# ---------------------------------------------------------------------------
def construir_fila_prediccion(hist: pd.DataFrame, casos_actual: float, semana: int,
                              departamento: str) -> pd.DataFrame:
    """Arma la fila de features para predecir a partir del historial del distrito.

    hist: filas del distrito ordenadas cronologicamente (columna 'casos').
    casos_actual: casos observados en la semana desde la que se predice.
    Las variables derivadas se calculan automaticamente (no se piden al usuario).
    """
    casos_prev = hist["casos"].tail(8).tolist()

    def ultimo(n):
        return float(casos_prev[-n]) if len(casos_prev) >= n else 0.0

    ult4 = casos_prev[-4:] if len(casos_prev) >= 1 else [0.0]
    ult8 = casos_prev[-8:] if len(casos_prev) >= 1 else [0.0]
    lag1 = ultimo(1)

    fila = {
        "casos": float(casos_actual),
        "casos_lag_1": lag1,
        "casos_lag_2": ultimo(2),
        "casos_lag_4": ultimo(4),
        "promedio_movil_4": float(np.mean(ult4)),
        "promedio_movil_8": float(np.mean(ult8)),
        "desviacion_movil_4": float(np.std(ult4, ddof=1)) if len(ult4) > 1 else 0.0,
        "crecimiento_semanal": (float(casos_actual) - lag1) / (lag1 + 1.0),
        "semana_sen": float(np.sin(2 * np.pi * semana / 52.0)),
        "semana_cos": float(np.cos(2 * np.pi * semana / 52.0)),
        "departamento": departamento,
    }
    return pd.DataFrame([fila])[FEATURE_NUMERIC + FEATURE_CATEGORICAL]
