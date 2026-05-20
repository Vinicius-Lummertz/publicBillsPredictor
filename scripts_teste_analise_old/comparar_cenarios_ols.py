from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts_teste_analise_old.ols_valor_vencedor import (
    build_model_data,
    finite_population_sample_size,
    fit_ols,
    inverse_transform_target,
    regression_metrics,
    sample_indexes,
    train_test_indexes,
    transform_target,
)


TARGET = "valorTotalVencedor"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("reports/ols_comparacao")
SEED = 42
TEST_SIZE = 0.20


SCENARIOS = [
    {
        "cenario": "minima_raw_com_vazamento",
        "include_leakage": True,
        "target_transform": "raw",
        "model_sample": "recommended",
        "correlation_threshold": 0.30,
        "max_features": 25,
    },
    {
        "cenario": "minima_log_com_vazamento",
        "include_leakage": True,
        "target_transform": "log1p",
        "model_sample": "recommended",
        "correlation_threshold": 0.30,
        "max_features": 25,
    },
    {
        "cenario": "completa_raw_com_vazamento",
        "include_leakage": True,
        "target_transform": "raw",
        "model_sample": "all",
        "correlation_threshold": 0.30,
        "max_features": 25,
    },
    {
        "cenario": "completa_log_com_vazamento",
        "include_leakage": True,
        "target_transform": "log1p",
        "model_sample": "all",
        "correlation_threshold": 0.30,
        "max_features": 25,
    },
    {
        "cenario": "minima_raw_sem_vazamento",
        "include_leakage": False,
        "target_transform": "raw",
        "model_sample": "recommended",
        "correlation_threshold": 0.30,
        "max_features": 25,
    },
    {
        "cenario": "completa_raw_sem_vazamento",
        "include_leakage": False,
        "target_transform": "raw",
        "model_sample": "all",
        "correlation_threshold": 0.30,
        "max_features": 25,
    },
    {
        "cenario": "completa_log_sem_vazamento",
        "include_leakage": False,
        "target_transform": "log1p",
        "model_sample": "all",
        "correlation_threshold": 0.30,
        "max_features": 25,
    },
    {
        "cenario": "completa_raw_sem_vazamento_corr_010",
        "include_leakage": False,
        "target_transform": "raw",
        "model_sample": "all",
        "correlation_threshold": 0.10,
        "max_features": 25,
    },
    {
        "cenario": "completa_log_sem_vazamento_corr_010",
        "include_leakage": False,
        "target_transform": "log1p",
        "model_sample": "all",
        "correlation_threshold": 0.10,
        "max_features": 25,
    },
]


def evaluate_scenario(config: dict, cache: dict) -> dict:
    cache_key = (config["include_leakage"], config["correlation_threshold"], config["max_features"])
    if cache_key not in cache:
        print(
            "Reconstruindo features: "
            f"leakage={config['include_leakage']}, "
            f"threshold={config['correlation_threshold']}"
        )
        cache[cache_key] = build_model_data(
            data_dir=DATA_DIR,
            target=TARGET,
            threshold=config["correlation_threshold"],
            max_features=config["max_features"],
            include_leakage=config["include_leakage"],
        )

    df, matrix, feature_names, selected_features, _ = cache[cache_key]
    sample_info = finite_population_sample_size(
        population_size=len(df),
        confidence=0.95,
        margin_error=0.05,
        p=0.50,
    )
    sample_size = sample_info["sample_size_finite_population"]
    if config["model_sample"] == "all":
        sample_size = len(df)

    row_indexes = sample_indexes(len(df), sample_size, SEED)
    sampled_df = df.iloc[row_indexes].reset_index(drop=True)
    sampled_matrix = matrix[row_indexes].toarray()
    y_raw = sampled_df[TARGET].to_numpy(dtype="float64")
    y_model = transform_target(sampled_df[TARGET], config["target_transform"])

    train_idx, test_idx = train_test_indexes(len(sampled_df), TEST_SIZE, SEED)
    x_train = sampled_matrix[train_idx]
    x_test = sampled_matrix[test_idx]
    y_train_model = y_model[train_idx]

    ols = fit_ols(x_train, y_train_model, feature_names)
    train_pred_model = x_train @ ols["beta"][1:] + ols["beta"][0]
    test_pred_model = x_test @ ols["beta"][1:] + ols["beta"][0]

    y_train_raw = y_raw[train_idx]
    y_test_raw = y_raw[test_idx]
    train_pred_raw = np.clip(inverse_transform_target(train_pred_model, config["target_transform"]), 0, None)
    test_pred_raw = np.clip(inverse_transform_target(test_pred_model, config["target_transform"]), 0, None)

    train_metrics = regression_metrics(y_train_raw, train_pred_raw, "treino")
    test_metrics = regression_metrics(y_test_raw, test_pred_raw, "teste")

    return {
        "cenario": config["cenario"],
        "include_leakage": config["include_leakage"],
        "target_transform": config["target_transform"],
        "model_sample": config["model_sample"],
        "correlation_threshold": config["correlation_threshold"],
        "sample_size": int(len(sampled_df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "n_features": int(len(feature_names)),
        "n_features_leakage": int(selected_features["possivel_vazamento_target"].sum()),
        "ols_train_r2_model_scale": ols["metrics"]["r2"],
        "ols_train_adj_r2_model_scale": ols["metrics"]["r2_ajustado"],
        "condition_number": ols["metrics"]["condition_number"],
        **train_metrics,
        **test_metrics,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    rows = []

    for config in SCENARIOS:
        print(f"\nRodando cenario: {config['cenario']}")
        try:
            rows.append(evaluate_scenario(config, cache))
        except Exception as exc:
            rows.append({"cenario": config["cenario"], "erro": str(exc)})
            print(f"Erro no cenario {config['cenario']}: {exc}")

    result = pd.DataFrame(rows)
    result = result.sort_values(["teste_r2", "teste_rmse"], ascending=[False, True], na_position="last")
    result.to_csv(OUTPUT_DIR / "comparacao_cenarios_ols.csv", index=False, encoding="utf-8-sig")

    with (OUTPUT_DIR / "comparacao_cenarios_ols.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    columns = [
        "cenario",
        "sample_size",
        "n_features",
        "n_features_leakage",
        "target_transform",
        "teste_r2",
        "teste_mae",
        "teste_rmse",
        "ols_train_r2_model_scale",
    ]
    print("\n=== Comparacao dos cenarios ===")
    print(result[[column for column in columns if column in result.columns]].to_string(index=False))
    print(f"\nRelatorios salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
