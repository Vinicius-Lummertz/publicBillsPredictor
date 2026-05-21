from pathlib import Path

import json
import numpy as np
import pandas as pd
from scipy import stats

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


def calcular_estatisticas_ols(x_train, y_train, beta, feature_names):
    x_const = np.column_stack([np.ones(len(x_train)), x_train])
    y_pred = x_const @ beta

    n = len(y_train)
    k = x_const.shape[1] - 1

    residuos = y_train - y_pred
    sse = np.sum(residuos ** 2)
    ssr = np.sum((y_pred - np.mean(y_train)) ** 2)
    sst = np.sum((y_train - np.mean(y_train)) ** 2)

    df_resid = n - k - 1
    df_model = k

    mse = sse / df_resid

    x_inv = np.linalg.pinv(x_const.T @ x_const)
    erros_padrao = np.sqrt(np.diag(mse * x_inv))

    t_values = beta / erros_padrao
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=df_resid))

    msr = ssr / df_model
    f_statistic = msr / mse
    prob_f = 1 - stats.f.cdf(f_statistic, df_model, df_resid)

    tabela = pd.DataFrame({
        "variavel": ["constante"] + list(feature_names),
        "coeficiente": beta,
        "erro_padrao": erros_padrao,
        "t_value": t_values,
        "p_valor": p_values,
    })

    return tabela, f_statistic, prob_f


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

    estatisticas_coeficientes, f_statistic, prob_f = calcular_estatisticas_ols(
        x_train=x[train_idx],
        y_train=y[train_idx],
        beta=ols["beta"],
        feature_names=feature_names,
    )

    train_pred = np.column_stack([np.ones(len(train_idx)), x[train_idx]]) @ ols["beta"]
    test_pred = np.column_stack([np.ones(len(test_idx)), x[test_idx]]) @ ols["beta"]

    train_pred = np.clip(train_pred, 0, None)
    test_pred = np.clip(test_pred, 0, None)

    train_metrics = regression_metrics(y[train_idx], train_pred, "treino")
    test_metrics = regression_metrics(y[test_idx], test_pred, "teste")

    coeficientes = estatisticas_coeficientes.merge(
    selected_features[["feature", "correlacao_raw", "possivel_vazamento_target"]],
    left_on="variavel",
    right_on="feature",
    how="left",
)

    coeficientes = coeficientes.drop(columns=["feature"])
    
    coeficientes = coeficientes.rename(columns={
        "variavel": "Variável",
        "coeficiente": "Coeficiente",
        "erro_padrao": "Erro padrão",
        "t_value": "Valor t",
        "p_valor": "P-valor",
        "correlacao_raw": "Correlação com target",
        "possivel_vazamento_target": "Possível vazamento"
    })
    
    coeficientes["Coeficiente"] = coeficientes["Coeficiente"].round(6)
    coeficientes["Erro padrão"] = coeficientes["Erro padrão"].round(6)
    coeficientes["Valor t"] = coeficientes["Valor t"].round(6)
    coeficientes["P-valor"] = coeficientes["P-valor"].apply(lambda x: f"{x:.10f}")
    coeficientes["Correlação com target"] = coeficientes["Correlação com target"].round(6)
    
    coeficientes["Possível vazamento"] = coeficientes["Possível vazamento"].map({
        True: "Sim",
        False: "Não"
    })
    
    coeficientes = coeficientes[
        [
            "Variável",
            "Coeficiente",
            "P-valor",
            "Erro padrão",
            "Valor t",
            "Correlação com target",
            "Possível vazamento",
        ]
    ]

    previsoes = pd.DataFrame({
        "valor_real": y[test_idx],
        "valor_previsto": test_pred,
        "residuo": y[test_idx] - test_pred,
    })

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
        "f_statistic": float(f_statistic),
        "prob_f": float(prob_f),
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
    print(f"F-statistic: {f_statistic:.6f}")
    print(f"ProbF: {prob_f:.10f}")
    print(f"Arquivos salvos em: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()