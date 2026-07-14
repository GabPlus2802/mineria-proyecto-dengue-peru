"""Clasificacion de alta incidencia de dengue en la semana siguiente.

Compara Random Forest y XGBoost sobre la unidad distrito-semana, con un
pipeline de preprocesamiento que se ajusta solo con entrenamiento. Incluye
metricas completas, barrido de umbral y utilidades para SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from xgboost import XGBClassifier

import config
from src.preprocessing import FEATURE_CATEGORICAL, FEATURE_NUMERIC, TARGET


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
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
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


def train_models(data: dict):
    """Entrena RF y XGBoost. Aplica balanceo si el desbalance supera 80/20."""
    y_train = data["y_train"]
    mayoria = _imbalance_ratio(y_train)
    balanced = mayoria > 0.80
    n_neg = int((y_train == 0).sum())
    n_pos = max(int((y_train == 1).sum()), 1)
    spw = n_neg / n_pos if balanced else 1.0

    rf = build_random_forest(balanced)
    xgb = build_xgboost(spw)
    rf.fit(data["X_train"], y_train)
    xgb.fit(data["X_train"], y_train)

    info = {"clase_mayoritaria": round(mayoria, 3), "balanceo_aplicado": balanced,
            "scale_pos_weight": round(spw, 2)}
    return {"random_forest": rf, "xgboost": xgb}, info


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
