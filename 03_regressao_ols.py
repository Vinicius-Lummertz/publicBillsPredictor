from pathlib import Path

import json
import numpy as np
import pandas as pd

from scripts_teste_analise_old.ols_valor_vencedor import (
    build_model_data,
    finite_population_sample_size,
    fit_ols,
    regression_metrics,
    sample_indexes,
    train_test_indexes,
)


TARGET = "valorTotalVencedor"
DATA_DIR = Path("data")
OUT_DIR = Path("reports/finais/03_regressao_ols")
CORRELATION_THRESHOLD = 0.30
MAX_FEATURES = 25
INCLUDE_LEAKAGE = True
USE_FULL_BASE = True
TEST_SIZE = 0.20
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, matrix, feature_names, selected_features, _ = build_model_data(
        data_dir=DATA_DIR,
        target=TARGET,
        threshold=CORRELATION_THRESHOLD,
        max_features=MAX_FEATURES,
        include_leakage=INCLUDE_LEAKAGE,
    )
    sample_info = finite_population_sample_size(
        population_size=len(df),
        confidence=0.95,
        margin_error=0.05,
        p=0.50,
    )

    sample_size = len(df) if USE_FULL_BASE else sample_info["sample_size_finite_population"]
    row_indexes = sample_indexes(len(df), sample_size, SEED)
    sampled_df = df.iloc[row_indexes].reset_index(drop=True)
    x = matrix[row_indexes].toarray()
    y = sampled_df[TARGET].to_numpy(dtype="float64")

    train_idx, test_idx = train_test_indexes(len(sampled_df), TEST_SIZE, SEED)
    ols = fit_ols(x[train_idx], y[train_idx], feature_names)

    train_pred = np.column_stack([np.ones(len(train_idx)), x[train_idx]]) @ ols["beta"]
    test_pred = np.column_stack([np.ones(len(test_idx)), x[test_idx]]) @ ols["beta"]
    train_pred = np.clip(train_pred, 0, None)
    test_pred = np.clip(test_pred, 0, None)

    train_metrics = regression_metrics(y[train_idx], train_pred, "treino")
    test_metrics = regression_metrics(y[test_idx], test_pred, "teste")

    coeficientes = ols["coefficients"].merge(
        selected_features[["feature", "correlacao_raw", "possivel_vazamento_target"]],
        left_on="variavel",
        right_on="feature",
        how="left",
    )
    previsoes = pd.DataFrame(
        {
            "valor_real": y[test_idx],
            "valor_previsto": test_pred,
            "residuo": y[test_idx] - test_pred,
        }
    )
    resumo = {
        "target": TARGET,
        "populacao": len(df),
        "amostra_indicada": sample_info["sample_size_finite_population"],
        "amostra_usada": len(sampled_df),
        "treino": len(train_idx),
        "teste": len(test_idx),
        "features": len(feature_names),
        "features_com_possivel_vazamento": int(selected_features["possivel_vazamento_target"].sum()),
        "r2_treino_ols": ols["metrics"]["r2"],
        "r2_ajustado_treino_ols": ols["metrics"]["r2_ajustado"],
        **train_metrics,
        **test_metrics,
    }

    selected_features.to_csv(OUT_DIR / "features_usadas.csv", index=False, encoding="utf-8-sig")
    coeficientes.to_csv(OUT_DIR / "coeficientes_ols.csv", index=False, encoding="utf-8-sig")
    previsoes.to_csv(OUT_DIR / "previsoes_teste.csv", index=False, encoding="utf-8-sig")
    with (OUT_DIR / "resumo_ols.json").open("w", encoding="utf-8") as handle:
        json.dump(resumo, handle, ensure_ascii=False, indent=2)

    print("=== OLS ===")
    print(f"Target prevista: {TARGET}")
    print(f"Populacao: {len(df):,}".replace(",", "."))
    print(f"Amostra indicada: {sample_info['sample_size_finite_population']}")
    print(f"Amostra usada: {len(sampled_df):,}".replace(",", "."))
    print(f"Features usadas: {len(feature_names)}")
    print(f"R2 treino: {ols['metrics']['r2']:.6f}")
    print(f"R2 ajustado treino: {ols['metrics']['r2_ajustado']:.6f}")
    print(f"R2 teste: {test_metrics['teste_r2']:.6f}")
    print(f"MAE teste: {test_metrics['teste_mae']:.2f}")
    print(f"RMSE teste: {test_metrics['teste_rmse']:.2f}")
    print(f"Arquivos salvos em: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
