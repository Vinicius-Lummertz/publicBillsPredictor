from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts_teste_analise_old.analise_target_licitacoes import load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Testa normalidade da variavel alvo com Shapiro-Wilk e compara "
            "transformacoes de normalizacao vistas em aula."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Pasta com os arquivos JSON.")
    parser.add_argument("--target", default="valorTotalVencedor", help="Variavel alvo.")
    parser.add_argument(
        "--output-dir",
        default="reports/normalidade_target",
        help="Pasta onde os relatorios serao salvos.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Tamanho da amostra para o SW Teste. A aula recomenda ate 2000.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Quantidade de amostras aleatorias testadas por transformacao.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia.")
    parser.add_argument("--seed", type=int, default=42, help="Semente de aleatoriedade.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Nao gerar histogramas e QQ plots.",
    )
    return parser.parse_args()


def get_target_series(data_dir: Path, target: str) -> pd.Series:
    df, _ = load_records(data_dir, target)
    if df.empty or target not in df.columns:
        raise ValueError(f"Nenhum registro encontrado para a target '{target}'.")

    series = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError(f"A target '{target}' nao tem valores numericos validos.")

    return series.reset_index(drop=True)


def iqr_limits(values: pd.Series) -> tuple[float, float, float, float, float]:
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def describe_values(name: str, values: pd.Series) -> dict[str, Any]:
    q1, q3, iqr, lower, upper = iqr_limits(values)
    outliers = int(((values < lower) | (values > upper)).sum())

    return {
        "serie": name,
        "registros": int(values.count()),
        "media": float(values.mean()),
        "mediana": float(values.median()),
        "desvio_padrao": float(values.std(ddof=1)),
        "assimetria_skew": float(values.skew()),
        "curtose": float(values.kurtosis()),
        "min": float(values.min()),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior_iqr": lower,
        "limite_superior_iqr": upper,
        "max": float(values.max()),
        "outliers_iqr": outliers,
        "outliers_iqr_pct": float(outliers / len(values)),
    }


def shift_for_positive(values: pd.Series, minimum_positive: float = 1e-6) -> tuple[pd.Series, float]:
    min_value = float(values.min())
    offset = 0.0
    if min_value <= 0:
        offset = abs(min_value) + minimum_positive
    return values + offset, offset


def shift_for_non_negative(values: pd.Series) -> tuple[pd.Series, float]:
    min_value = float(values.min())
    offset = abs(min_value) if min_value < 0 else 0.0
    return values + offset, offset


def build_transformations(values: pd.Series) -> dict[str, dict[str, Any]]:
    q1, q3, iqr, lower, upper = iqr_limits(values)
    clipped = values.clip(lower=lower, upper=upper)

    non_negative, sqrt_offset = shift_for_non_negative(values)
    positive, boxcox_offset = shift_for_positive(values)
    clipped_positive, clipped_boxcox_offset = shift_for_positive(clipped)
    clipped_non_negative, clipped_sqrt_offset = shift_for_non_negative(clipped)

    boxcox_values, boxcox_lambda = stats.boxcox(positive.to_numpy(dtype="float64"))
    clipped_boxcox_values, clipped_boxcox_lambda = stats.boxcox(clipped_positive.to_numpy(dtype="float64"))

    return {
        "original": {
            "values": values,
            "metodo": "Sem transformacao.",
            "parametros": {},
        },
        "iqr_winsorizado": {
            "values": clipped,
            "metodo": "Substitui outliers pelos limites Q1 - 1.5*IQR e Q3 + 1.5*IQR.",
            "parametros": {"q1": q1, "q3": q3, "iqr": iqr, "limite_inferior": lower, "limite_superior": upper},
        },
        "log1p": {
            "values": np.log1p(non_negative),
            "metodo": "Aplica log(1 + x). Reduz a cauda de valores monetarios muito altos.",
            "parametros": {"offset": sqrt_offset},
        },
        "sqrt": {
            "values": np.sqrt(non_negative),
            "metodo": "Aplica raiz quadrada. Reduz dispersao de forma mais suave que log.",
            "parametros": {"offset": sqrt_offset},
        },
        "boxcox": {
            "values": pd.Series(boxcox_values),
            "metodo": "Transformacao Box-Cox; escolhe lambda para aproximar normalidade.",
            "parametros": {"offset": boxcox_offset, "lambda": float(boxcox_lambda)},
        },
        "iqr_log1p": {
            "values": np.log1p(clipped_non_negative),
            "metodo": "Primeiro trata outliers por IQR, depois aplica log(1 + x).",
            "parametros": {"offset": clipped_sqrt_offset, "limite_inferior": lower, "limite_superior": upper},
        },
        "iqr_sqrt": {
            "values": np.sqrt(clipped_non_negative),
            "metodo": "Primeiro trata outliers por IQR, depois aplica raiz quadrada.",
            "parametros": {"offset": clipped_sqrt_offset, "limite_inferior": lower, "limite_superior": upper},
        },
        "iqr_boxcox": {
            "values": pd.Series(clipped_boxcox_values),
            "metodo": "Primeiro trata outliers por IQR, depois aplica Box-Cox.",
            "parametros": {
                "offset": clipped_boxcox_offset,
                "lambda": float(clipped_boxcox_lambda),
                "limite_inferior": lower,
                "limite_superior": upper,
            },
        },
    }


def as_series(values: Any) -> pd.Series:
    return pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def shapiro_repeated(
    values: pd.Series,
    sample_size: int,
    repeats: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    values_array = values.to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(values_array))
    rows: list[dict[str, Any]] = []

    for repeat in range(1, repeats + 1):
        if sample_size < len(values_array):
            sample = rng.choice(values_array, size=sample_size, replace=False)
        else:
            sample = values_array

        if np.nanstd(sample) == 0:
            statistic, p_value = np.nan, np.nan
        else:
            statistic, p_value = stats.shapiro(sample)

        rows.append(
            {
                "repeticao": repeat,
                "n_amostra": int(sample_size),
                "w_statistic": float(statistic),
                "p_value": float(p_value),
                "normal_p_maior_que_alpha": bool(p_value > alpha) if np.isfinite(p_value) else False,
            }
        )

    result_df = pd.DataFrame(rows)
    return {
        "detalhes": result_df,
        "w_mediano": float(result_df["w_statistic"].median()),
        "p_value_mediano": float(result_df["p_value"].median()),
        "p_value_min": float(result_df["p_value"].min()),
        "p_value_max": float(result_df["p_value"].max()),
        "amostras_normais": int(result_df["normal_p_maior_que_alpha"].sum()),
        "amostras_testadas": int(len(result_df)),
        "normalidade_aprovada": bool(result_df["p_value"].median() > alpha),
    }


def plot_distribution(name: str, values: pd.Series, output_dir: Path, sample_seed: int) -> None:
    safe_name = name.replace(" ", "_").replace("/", "_")
    plot_values = values
    if len(plot_values) > 10_000:
        plot_values = plot_values.sample(n=10_000, random_state=sample_seed)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(plot_values, bins=60, color="#2f6f73", edgecolor="white", alpha=0.9)
    axes[0].set_title(f"Histograma - {name}")
    axes[0].set_xlabel("Valor transformado")
    axes[0].set_ylabel("Frequencia")

    stats.probplot(plot_values, dist="norm", plot=axes[1])
    axes[1].set_title(f"QQ plot - {name}")

    figure.tight_layout()
    figure.savefig(output_dir / f"{safe_name}.png", dpi=140)
    plt.close(figure)


def write_markdown_report(
    path: Path,
    target: str,
    alpha: float,
    sample_size: int,
    repeats: int,
    total_records: int,
    summary_df: pd.DataFrame,
) -> None:
    best = summary_df.iloc[0]
    original = summary_df[summary_df["serie"] == "original"].iloc[0]
    normal_rows = summary_df[summary_df["normalidade_aprovada_p_mediano_maior_que_alpha"]]

    if normal_rows.empty:
        conclusion = (
            "Nenhuma versao da target foi aprovada formalmente no Shapiro-Wilk "
            f"com alpha = {alpha}. Mesmo assim, a melhor aproximacao foi "
            f"`{best['serie']}`, pois apresentou o maior p-value mediano e W mediano."
        )
    else:
        approved = ", ".join(normal_rows["serie"].astype(str).tolist())
        conclusion = f"As seguintes versoes passaram no criterio p-value > {alpha}: {approved}."

    markdown = f"""# Teste de Normalidade da Target

Target analisada: `{target}`

Total de registros validos: `{total_records}`

O Shapiro-Wilk foi aplicado em `{repeats}` amostras aleatorias de ate `{sample_size}` registros, porque a base completa e grande demais para a faixa recomendada em aula para o SW Teste.

## Hipoteses

- H0: a serie segue uma distribuicao normal.
- H1: a serie nao segue uma distribuicao normal.
- Criterio: se `p-value > {alpha}`, nao rejeitamos H0; se `p-value <= {alpha}`, rejeitamos H0.

## Resultado

Target original:

- W mediano: `{original['shapiro_w_mediano']:.6f}`
- p-value mediano: `{original['shapiro_p_value_mediano']:.6e}`
- Assimetria: `{original['assimetria_skew']:.6f}`
- Outliers pelo IQR: `{int(original['outliers_iqr'])}`

Melhor transformacao encontrada:

- Serie: `{best['serie']}`
- Metodo: {best['metodo']}
- W mediano: `{best['shapiro_w_mediano']:.6f}`
- p-value mediano: `{best['shapiro_p_value_mediano']:.6e}`
- Assimetria: `{best['assimetria_skew']:.6f}`
- Outliers pelo IQR: `{int(best['outliers_iqr'])}`

## Conclusao

{conclusion}

Para o projeto, isso indica que `valorTotalVencedor` e uma variavel monetaria muito assimetrica, com cauda longa a direita. A transformacao `log1p` e a mais defensavel para a proxima etapa de modelagem, porque reduziu bastante a assimetria e aproximou mais a distribuicao de uma normal, mesmo sem passar formalmente no teste.
"""
    path.write_text(markdown, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo target '{args.target}' em: {data_dir.resolve()}")
    target_values = get_target_series(data_dir, args.target)
    print(f"Registros validos da target: {len(target_values):,}".replace(",", "."))

    transformations = build_transformations(target_values)
    summary_rows: list[dict[str, Any]] = []
    details_rows: list[pd.DataFrame] = []

    for index, (name, info) in enumerate(transformations.items()):
        values = as_series(info["values"])
        shapiro = shapiro_repeated(
            values=values,
            sample_size=args.sample_size,
            repeats=args.repeats,
            alpha=args.alpha,
            seed=args.seed + index,
        )

        description = describe_values(name, values)
        row = {
            **description,
            "metodo": info["metodo"],
            "parametros": json.dumps(info["parametros"], ensure_ascii=False),
            "shapiro_w_mediano": shapiro["w_mediano"],
            "shapiro_p_value_mediano": shapiro["p_value_mediano"],
            "shapiro_p_value_min": shapiro["p_value_min"],
            "shapiro_p_value_max": shapiro["p_value_max"],
            "amostras_normais": shapiro["amostras_normais"],
            "amostras_testadas": shapiro["amostras_testadas"],
            "normalidade_aprovada_p_mediano_maior_que_alpha": shapiro["normalidade_aprovada"],
        }
        summary_rows.append(row)

        detail_df = shapiro["detalhes"].copy()
        detail_df.insert(0, "serie", name)
        details_rows.append(detail_df)

        if not args.no_plots:
            plot_distribution(name, values, output_dir, args.seed)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["normalidade_aprovada_p_mediano_maior_que_alpha", "shapiro_p_value_mediano", "shapiro_w_mediano"],
        ascending=[False, False, False],
    )
    details_df = pd.concat(details_rows, ignore_index=True)

    summary_path = output_dir / f"normalidade_{args.target}.csv"
    details_path = output_dir / f"normalidade_{args.target}_amostras.csv"
    json_path = output_dir / f"normalidade_{args.target}_resumo.json"
    markdown_path = output_dir / f"normalidade_{args.target}_relatorio.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    best = summary_df.iloc[0].to_dict()
    final_report = {
        "target": args.target,
        "alpha": args.alpha,
        "sample_size": args.sample_size,
        "repeats": args.repeats,
        "total_registros_validos": int(len(target_values)),
        "melhor_transformacao_por_p_value_mediano": {
            "serie": best["serie"],
            "shapiro_w_mediano": best["shapiro_w_mediano"],
            "shapiro_p_value_mediano": best["shapiro_p_value_mediano"],
            "normalidade_aprovada": best["normalidade_aprovada_p_mediano_maior_que_alpha"],
            "metodo": best["metodo"],
            "parametros": json.loads(best["parametros"]),
        },
        "observacao": (
            "Como a base tem muitos registros, o SW Teste foi aplicado em amostras "
            "aleatorias repetidas. Isso segue a faixa recomendada em aula para o Shapiro-Wilk."
        ),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(final_report, handle, ensure_ascii=False, indent=2)

    write_markdown_report(
        path=markdown_path,
        target=args.target,
        alpha=args.alpha,
        sample_size=args.sample_size,
        repeats=args.repeats,
        total_records=len(target_values),
        summary_df=summary_df,
    )

    display_columns = [
        "serie",
        "registros",
        "media",
        "mediana",
        "assimetria_skew",
        "outliers_iqr",
        "shapiro_w_mediano",
        "shapiro_p_value_mediano",
        "amostras_normais",
        "amostras_testadas",
        "normalidade_aprovada_p_mediano_maior_que_alpha",
    ]

    print("\n=== Resultado Shapiro-Wilk ===")
    print(summary_df[display_columns].to_string(index=False))
    print(f"\nMelhor transformacao: {final_report['melhor_transformacao_por_p_value_mediano']['serie']}")
    print(f"Relatorios salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
