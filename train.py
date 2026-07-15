"""Script unico de entrenamiento.

Genera TODOS los artefactos que necesita el dashboard:
  - data/processed/dengue_semanal.csv        (dataset maestro)
  - data/processed/distritos_clusters.csv    (clusters por distrito)
  - data/processed/metricas_clasificacion.csv
  - data/processed/metricas_pronostico.csv
  - models/*.joblib                           (modelos y preprocesador)

Uso:
    python train.py            # usa el dataset maestro si existe
    python train.py --rebuild  # reconstruye el dataset maestro desde el CSV crudo
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

import config
from src import clustering, forecasting, modeling, preprocessing


def cargar_o_construir_maestro(rebuild: bool) -> pd.DataFrame:
    if rebuild or not config.DENGUE_SEMANAL.exists():
        print(">> Construyendo dataset maestro desde el CSV crudo...")
        df = preprocessing.build_master_dataset()
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.DENGUE_SEMANAL, index=False, encoding="utf-8")
    else:
        print(">> Cargando dataset maestro existente...")
        df = pd.read_csv(config.DENGUE_SEMANAL)
    print(f"   {len(df):,} filas | {df['ubigeo'].nunique()} distritos")
    return df


def entrenar_clustering(df: pd.DataFrame):
    print("\n=== CLUSTERING (K-means) ===")
    perfil = clustering.district_profile(df)
    res = clustering.run_kmeans(perfil)
    print(f"   k elegido: {res['k']} (mejor silueta sugiere k={res['k_auto_silueta']})")
    print(f"   silueta final: {res['silueta']:.3f}")
    print(f"   varianza PCA 2D: {[round(v, 3) for v in res['pca_var']]}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(res["kmeans"], config.PATH_KMEANS, compress=3)
    joblib.dump(res["scaler"], config.PATH_SCALER_CLUSTERING, compress=3)
    res["perfil"].reset_index().to_csv(config.DISTRITOS_CLUSTERS, index=False, encoding="utf-8")
    print(f"   guardado: {config.DISTRITOS_CLUSTERS.name}, kmeans.joblib, scaler_clustering.joblib")
    print("\n   Perfil de clusters:")
    print(clustering.resumen_clusters(res["perfil"]).to_string())
    return res


def entrenar_clasificacion(df: pd.DataFrame):
    print("\n=== CLASIFICACION (Random Forest vs XGBoost) ===")
    data = modeling.get_modeling_frame(df)
    print(f"   train={len(data['y_train']):,}  val={len(data['y_val']):,}  test={len(data['y_test']):,}")

    models, info = modeling.train_models(data)
    print(f"   clase mayoritaria train: {info['clase_mayoritaria']} | "
          f"balanceo: {info['balanceo_aplicado']} | scale_pos_weight: {info['scale_pos_weight']}")

    tabla = modeling.metrics_table(models, data["X_test"], data["y_test"])
    print("\n   Metricas en TEST (ultimo anio):")
    print(tabla.to_string())

    mejor = modeling.elegir_mejor_modelo(tabla)
    print(f"\n   Mejor modelo (por F1): {mejor}")

    # Guardar metricas con la particion indicada
    tabla_out = tabla.reset_index()
    tabla_out.insert(1, "particion", "test")
    tabla_out.to_csv(config.METRICAS_CLASIFICACION, index=False, encoding="utf-8")

    # Efecto del balanceo en el recall de la clase minoritaria (desbalance > 80/20)
    if info["balanceo_aplicado"]:
        bal = modeling.comparar_balanceo(data)
        bal.to_csv(config.METRICAS_BALANCEO, index=False, encoding="utf-8")
        print("\n   Efecto del balanceo (recall de la clase 1 en test):")
        print(bal.to_string(index=False))

    # Guardar modelos y preprocesador (ajustado con train)
    prep = modeling.build_preprocessor().fit(data["X_train"])
    joblib.dump(prep, config.PATH_PREPROCESSOR, compress=3)
    joblib.dump(models["random_forest"], config.PATH_RANDOM_FOREST, compress=3)
    joblib.dump(models["xgboost"], config.PATH_XGBOOST, compress=3)
    extra = {k: models[k] for k in ("gradient_boosting", "logistic_regression", "decision_tree")}
    joblib.dump(extra, config.PATH_MODELOS_EXTRA, compress=3)
    joblib.dump(
        {
            "mejor_modelo": mejor,
            "threshold": config.CLASSIFICATION_THRESHOLD,
            "features_numericas": preprocessing.FEATURE_NUMERIC,
            "features_categoricas": preprocessing.FEATURE_CATEGORICAL,
            "info_balanceo": info,
        },
        config.PATH_MODEL_META,
    )
    print(f"   guardado: metricas_clasificacion.csv, preprocessor/random_forest/xgboost.joblib")
    return models, mejor


def entrenar_pronostico(df: pd.DataFrame):
    print("\n=== PRONOSTICO (serie nacional) ===")
    serie = forecasting.build_series(df, nivel="nacional")
    ev = forecasting.evaluate_models(serie)
    filas = []
    for modelo, m in ev["resultados"].items():
        filas.append({"modelo": modelo, "mape": round(m["mape"], 3), "rmse": round(m["rmse"], 3)})
    tabla = pd.DataFrame(filas)
    tabla["elegido"] = tabla["modelo"] == ev["mejor_modelo"]
    print(tabla.to_string(index=False))
    tabla.to_csv(config.METRICAS_PRONOSTICO, index=False, encoding="utf-8")

    futuro = forecasting.forecast_future(serie)
    print(f"\n   Pronostico {config.FORECAST_PERIODS} semanas futuras:")
    print(futuro.round(1).to_string(index=False))
    print(f"   guardado: metricas_pronostico.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Reconstruir dataset maestro")
    args = parser.parse_args()

    df = cargar_o_construir_maestro(args.rebuild)
    entrenar_clustering(df)
    entrenar_clasificacion(df)
    entrenar_pronostico(df)
    print("\nOK: todos los artefactos fueron generados.")


if __name__ == "__main__":
    main()
