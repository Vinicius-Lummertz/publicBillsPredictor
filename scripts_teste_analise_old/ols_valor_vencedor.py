from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts_teste_analise_old.analise_target_licitacoes import (
    analyze_correlations,
    build_feature_matrix,
    load_records,
    prepare_dataframe,
    select_feature_columns,
)


CONFIDENCE_Z = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula amostra indicada e aplica OLS para prever a target "
            "valorTotalVencedor usando variaveis bem correlacionadas."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Pasta com os arquivos JSON.")
    parser.add_argument("--target", default="valorTotalVencedor", help="Variavel alvo.")
    parser.add_argument("--output-dir", default="reports/ols", help="Pasta de saida.")
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.30,
        help="Valor minimo de |correlacao| para selecionar features.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=25,
        help="Quantidade maxima de variaveis explicativas usadas no OLS.",
    )
    parser.add_argument(
        "--include-leakage",
        action="store_true",
        help="Inclui variaveis muito proximas da target, como outros campos de valor.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["raw", "log1p"],
        default="log1p",
        help="Transformacao da target no OLS. log1p foi a melhor na etapa de normalidade.",
    )
    parser.add_argument("--confidence", type=float, default=0.95, choices=sorted(CONFIDENCE_Z))
    parser.add_argument("--margin-error", type=float, default=0.05, help="Margem de erro para amostragem.")
    parser.add_argument(
        "--population-proportion",
        type=float,
        default=0.50,
        help="p da formula amostral. Use 0.5 quando a proporcao esperada e desconhecida.",
    )
    parser.add_argument(
        "--model-sample",
        choices=["recommended", "all"],
        default="recommended",
        help="Usa a amostra indicada ou toda a base para ajustar o OLS.",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporcao reservada para teste.")
    parser.add_argument("--seed", type=int, default=42, help="Semente da amostra aleatoria.")
    return parser.parse_args()


def finite_population_sample_size(
    population_size: int,
    confidence: float,
    margin_error: float,
    p: float,
) -> dict[str, Any]:
    z = CONFIDENCE_Z[confidence]
    n0 = (z**2 * p * (1 - p)) / (margin_error**2)
    corrected = n0 / (1 + ((n0 - 1) / population_size))

    return {
        "population_size": population_size,
        "confidence": confidence,
        "z": z,
        "margin_error": margin_error,
        "population_proportion": p,
        "sample_size_infinite_population": math.ceil(n0),
        "sample_size_finite_population": math.ceil(corrected),
    }


def build_model_data(
    data_dir: Path,
    target: str,
    threshold: float,
    max_features: int,
    include_leakage: bool,
) -> tuple[pd.DataFrame, Any, list[str], pd.DataFrame, pd.DataFrame]:
    raw_df, _ = load_records(data_dir, target)
    df, date_columns = prepare_dataframe(raw_df, target)
    df = df[df[target].notna()].reset_index(drop=True)

    numeric_columns, categorical_columns, text_columns, _ = select_feature_columns(
        df=df,
        target=target,
        date_columns=date_columns,
        max_categories=40,
        max_text_columns=8,
    )
    matrix, feature_names, feature_sources = build_feature_matrix(
        df=df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        text_columns=text_columns,
        min_category_frequency=10,
        max_text_features=40,
    )
    correlations = analyze_correlations(
        matrix=matrix,
        feature_names=feature_names,
        feature_sources=feature_sources,
        target_values=df[target],
        main_target="raw",
    )

    selected = correlations[correlations["abs_correlacao_usada"] >= threshold].copy()
    if not include_leakage:
        selected = selected[~selected["possivel_vazamento_target"]].copy()

    selected = selected.sort_values("abs_correlacao_usada", ascending=False).head(max_features)
    if selected.empty:
        leakage_text = "incluindo" if include_leakage else "sem incluir"
        raise ValueError(
            f"Nenhuma feature com |correlacao| >= {threshold} foi encontrada {leakage_text} vazamento. "
            "Reduza --correlation-threshold ou use --include-leakage."
        )

    index_lookup = {feature: index for index, feature in enumerate(feature_names)}
    selected_features = [feature for feature in selected["feature"].tolist() if feature in index_lookup]
    selected_indexes = [index_lookup[feature] for feature in selected_features]
    selected_matrix = matrix[:, selected_indexes]

    return df, selected_matrix, selected_features, selected, correlations


def transform_target(values: pd.Series, transform: str) -> np.ndarray:
    y = values.to_numpy(dtype="float64")
    if transform == "raw":
        return y
    if transform == "log1p":
        return np.log1p(np.clip(y, a_min=0, a_max=None))
    raise ValueError(f"Transformacao nao suportada: {transform}")


def inverse_transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "log1p":
        return np.expm1(values)
    raise ValueError(f"Transformacao nao suportada: {transform}")


def train_test_indexes(n_rows: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indexes = np.arange(n_rows)
    rng.shuffle(indexes)
    n_test = max(1, int(round(n_rows * test_size)))
    test_idx = indexes[:n_test]
    train_idx = indexes[n_test:]
    return train_idx, test_idx


def sample_indexes(n_rows: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if sample_size >= n_rows:
        return np.arange(n_rows)
    return np.sort(rng.choice(np.arange(n_rows), size=sample_size, replace=False))


def fit_ols(x: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float = 0.05) -> dict[str, Any]:
    x_const = np.column_stack([np.ones(len(x)), x])
    names = ["constante"] + feature_names

    beta, _, rank, _ = np.linalg.lstsq(x_const, y, rcond=None)
    fitted = x_const @ beta
    residuals = y - fitted

    n = len(y)
    p = x_const.shape[1]
    df_model = p - 1
    df_resid = max(n - p, 1)
    sse = float(np.sum(residuals**2))
    y_centered = y - y.mean()
    sst = float(np.sum(y_centered**2))
    ssr = max(sst - sse, 0.0)
    mse_resid = sse / df_resid

    xtx_inv = np.linalg.pinv(x_const.T @ x_const)
    covariance = mse_resid * xtx_inv
    std_error = np.sqrt(np.maximum(np.diag(covariance), 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = beta / std_error
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=df_resid))
    t_critical = stats.t.ppf(1 - alpha / 2, df=df_resid)

    r2 = 1 - (sse / sst) if sst > 0 else np.nan
    adj_r2 = 1 - ((1 - r2) * (n - 1) / df_resid) if np.isfinite(r2) else np.nan
    f_stat = (ssr / df_model) / mse_resid if df_model > 0 and mse_resid > 0 else np.nan
    f_p_value = 1 - stats.f.cdf(f_stat, df_model, df_resid) if np.isfinite(f_stat) else np.nan

    coefficients = pd.DataFrame(
        {
            "variavel": names,
            "coeficiente": beta,
            "erro_padrao": std_error,
            "t": t_values,
            "p_value": p_values,
            "intervalo_95_inf": beta - t_critical * std_error,
            "intervalo_95_sup": beta + t_critical * std_error,
        }
    )

    return {
        "beta": beta,
        "coefficients": coefficients,
        "fitted": fitted,
        "residuals": residuals,
        "metrics": {
            "n_observacoes_treino": int(n),
            "n_parametros": int(p),
            "rank_matriz": int(rank),
            "graus_liberdade_modelo": int(df_model),
            "graus_liberdade_residuos": int(df_resid),
            "sse": sse,
            "mse_residuos": float(mse_resid),
            "r2": float(r2),
            "r2_ajustado": float(adj_r2),
            "f_statistic": float(f_stat),
            "f_p_value": float(f_p_value),
            "condition_number": float(np.linalg.cond(x_const)),
        },
    }


def predict_ols(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x_const = np.column_stack([np.ones(len(x)), x])
    return x_const @ beta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs(errors / denom)) * 100)
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    sse = float(np.sum(errors**2))
    r2 = float(1 - sse / sst) if sst > 0 else np.nan

    return {
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_mape_pct": mape,
        f"{prefix}_r2": r2,
    }


def save_plots(output_dir: Path, y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_true, y_pred, s=18, alpha=0.65, color="#2f6f73")
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    axes[0].plot([lower, upper], [lower, upper], color="#b33f3f", linewidth=2)
    axes[0].set_title("Real vs previsto")
    axes[0].set_xlabel("Valor real")
    axes[0].set_ylabel("Valor previsto")

    axes[1].scatter(y_pred, residuals, s=18, alpha=0.65, color="#554c9a")
    axes[1].axhline(0, color="#b33f3f", linewidth=2)
    axes[1].set_title("Residuos")
    axes[1].set_xlabel("Valor previsto")
    axes[1].set_ylabel("Erro")

    figure.tight_layout()
    figure.savefig(output_dir / "ols_real_vs_previsto_residuos.png", dpi=140)
    plt.close(figure)


def write_text_summary(
    output_path: Path,
    target: str,
    target_transform: str,
    sample_info: dict[str, Any],
    selected_features: pd.DataFrame,
    ols_metrics: dict[str, Any],
    train_metrics_raw: dict[str, float],
    test_metrics_raw: dict[str, float],
    include_leakage: bool,
) -> None:
    lines = [
        "OLS - Regressao por Minimos Quadrados Ordinarios",
        "",
        f"Target: {target}",
        f"Transformacao da target no ajuste: {target_transform}",
        f"Variaveis com vazamento incluidas: {include_leakage}",
        "",
        "Amostragem:",
        f"- Populacao: {sample_info['population_size']}",
        f"- Confianca: {sample_info['confidence']:.0%}",
        f"- Margem de erro: {sample_info['margin_error']:.0%}",
        f"- Amostra indicada com correcao finita: {sample_info['sample_size_finite_population']}",
        "",
        "Metricas OLS no treino:",
        f"- R2: {ols_metrics['r2']:.6f}",
        f"- R2 ajustado: {ols_metrics['r2_ajustado']:.6f}",
        f"- F-statistic: {ols_metrics['f_statistic']:.6f}",
        f"- p-value do F: {ols_metrics['f_p_value']:.6e}",
        f"- Observacoes de treino: {ols_metrics['n_observacoes_treino']}",
        "",
        "Metricas em reais, apos desfazer transformacao quando necessario:",
        f"- Treino MAE: {train_metrics_raw['treino_mae']:.2f}",
        f"- Treino RMSE: {train_metrics_raw['treino_rmse']:.2f}",
        f"- Treino R2: {train_metrics_raw['treino_r2']:.6f}",
        f"- Teste MAE: {test_metrics_raw['teste_mae']:.2f}",
        f"- Teste RMSE: {test_metrics_raw['teste_rmse']:.2f}",
        f"- Teste R2: {test_metrics_raw['teste_r2']:.6f}",
        "",
        "Features usadas:",
    ]

    for row in selected_features.itertuples(index=False):
        lines.append(
            f"- {row.feature}: correlacao={row.correlacao_usada:.6f}, "
            f"possivel_vazamento={row.possivel_vazamento_target}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando dados e reconstruindo matriz de features...")
    df, matrix, feature_names, selected_features, correlations = build_model_data(
        data_dir=Path(args.data_dir),
        target=args.target,
        threshold=args.correlation_threshold,
        max_features=args.max_features,
        include_leakage=args.include_leakage,
    )

    sample_info = finite_population_sample_size(
        population_size=len(df),
        confidence=args.confidence,
        margin_error=args.margin_error,
        p=args.population_proportion,
    )
    desired_sample = (
        sample_info["sample_size_finite_population"] if args.model_sample == "recommended" else len(df)
    )
    row_indexes = sample_indexes(len(df), desired_sample, args.seed)
    sampled_df = df.iloc[row_indexes].reset_index(drop=True)
    sampled_matrix = matrix[row_indexes].toarray()
    y_raw = sampled_df[args.target].to_numpy(dtype="float64")
    y_model = transform_target(sampled_df[args.target], args.target_transform)

    train_idx, test_idx = train_test_indexes(len(sampled_df), args.test_size, args.seed)
    x_train = sampled_matrix[train_idx]
    x_test = sampled_matrix[test_idx]
    y_train = y_model[train_idx]
    y_test = y_model[test_idx]

    print(f"Populacao: {len(df):,}".replace(",", "."))
    print(f"Amostra indicada: {sample_info['sample_size_finite_population']:,}".replace(",", "."))
    print(f"Amostra usada no modelo: {len(sampled_df):,}".replace(",", "."))
    print(f"Features selecionadas: {len(feature_names)}")

    ols = fit_ols(x_train, y_train, feature_names)
    y_train_pred_model = predict_ols(x_train, ols["beta"])
    y_test_pred_model = predict_ols(x_test, ols["beta"])

    y_train_raw = y_raw[train_idx]
    y_test_raw = y_raw[test_idx]
    y_train_pred_raw = np.clip(inverse_transform_target(y_train_pred_model, args.target_transform), 0, None)
    y_test_pred_raw = np.clip(inverse_transform_target(y_test_pred_model, args.target_transform), 0, None)

    train_metrics_raw = regression_metrics(y_train_raw, y_train_pred_raw, "treino")
    test_metrics_raw = regression_metrics(y_test_raw, y_test_pred_raw, "teste")

    coefficients = ols["coefficients"].merge(
        selected_features[["feature", "correlacao_raw", "correlacao_log", "possivel_vazamento_target"]],
        left_on="variavel",
        right_on="feature",
        how="left",
    )
    predictions = pd.DataFrame(
        {
            "conjunto": "teste",
            "valor_real": y_test_raw,
            "valor_previsto": y_test_pred_raw,
            "residuo": y_test_raw - y_test_pred_raw,
        }
    )

    selected_features.to_csv(output_dir / "features_ols_selecionadas.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(output_dir / "correlacoes_recalculadas_ols.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(output_dir / "coeficientes_ols.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "previsoes_ols_teste.csv", index=False, encoding="utf-8-sig")

    metrics_report = {
        "target": args.target,
        "target_transform": args.target_transform,
        "include_leakage": args.include_leakage,
        "sample_info": sample_info,
        "model_sample": args.model_sample,
        "sample_used": int(len(sampled_df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "ols_metrics_model_scale": ols["metrics"],
        "train_metrics_raw_scale": train_metrics_raw,
        "test_metrics_raw_scale": test_metrics_raw,
        "selected_features": feature_names,
    }
    with (output_dir / "metricas_ols.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_report, handle, ensure_ascii=False, indent=2)

    save_plots(output_dir, y_test_raw, y_test_pred_raw, y_test_raw - y_test_pred_raw)
    write_text_summary(
        output_path=output_dir / "resumo_ols.txt",
        target=args.target,
        target_transform=args.target_transform,
        sample_info=sample_info,
        selected_features=selected_features,
        ols_metrics=ols["metrics"],
        train_metrics_raw=train_metrics_raw,
        test_metrics_raw=test_metrics_raw,
        include_leakage=args.include_leakage,
    )

    print("\n=== Resultado OLS ===")
    print(f"R2 treino na escala do modelo: {ols['metrics']['r2']:.6f}")
    print(f"R2 ajustado treino na escala do modelo: {ols['metrics']['r2_ajustado']:.6f}")
    print(f"MAE teste em reais: {test_metrics_raw['teste_mae']:.2f}")
    print(f"RMSE teste em reais: {test_metrics_raw['teste_rmse']:.2f}")
    print(f"R2 teste em reais: {test_metrics_raw['teste_r2']:.6f}")
    print(f"Relatorios salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
