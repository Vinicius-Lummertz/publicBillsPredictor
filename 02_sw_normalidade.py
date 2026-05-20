from pathlib import Path

import pandas as pd

from scripts_teste_analise_old.teste_normalidade_target import (
    as_series,
    build_transformations,
    describe_values,
    get_target_series,
    shapiro_repeated,
    write_markdown_report,
)


TARGET = "valorTotalVencedor"
DATA_DIR = Path("data")
OUT_DIR = Path("reports/finais/02_sw_normalidade")
SAMPLE_SIZE = 2000
REPEATS = 10
ALPHA = 0.05
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_values = get_target_series(DATA_DIR, TARGET)
    transformations = build_transformations(target_values)
    rows = []

    for index, (name, info) in enumerate(transformations.items()):
        values = as_series(info["values"])
        shapiro = shapiro_repeated(
            values=values,
            sample_size=SAMPLE_SIZE,
            repeats=REPEATS,
            alpha=ALPHA,
            seed=SEED + index,
        )
        rows.append(
            {
                **describe_values(name, values),
                "metodo": info["metodo"],
                "shapiro_w_mediano": shapiro["w_mediano"],
                "shapiro_p_value_mediano": shapiro["p_value_mediano"],
                "amostras_normais": shapiro["amostras_normais"],
                "amostras_testadas": shapiro["amostras_testadas"],
                "normalidade_aprovada": shapiro["normalidade_aprovada"],
            }
        )

    resultado = pd.DataFrame(rows).sort_values(
        ["normalidade_aprovada", "shapiro_p_value_mediano", "shapiro_w_mediano"],
        ascending=[False, False, False],
    )
    resultado.to_csv(OUT_DIR / "resultado_sw.csv", index=False, encoding="utf-8-sig")
    write_markdown_report(
        path=OUT_DIR / "relatorio_sw.md",
        target=TARGET,
        alpha=ALPHA,
        sample_size=SAMPLE_SIZE,
        repeats=REPEATS,
        total_records=len(target_values),
        summary_df=resultado.rename(
            columns={"normalidade_aprovada": "normalidade_aprovada_p_mediano_maior_que_alpha"}
        ),
    )

    print("=== Shapiro-Wilk ===")
    print(f"Target analisada: {TARGET}")
    print(f"Registros validos: {len(target_values):,}".replace(",", "."))
    print(f"Amostras por transformacao: {REPEATS} x {SAMPLE_SIZE}")
    print("\nResultado:")
    print(
        resultado[
            [
                "serie",
                "shapiro_w_mediano",
                "shapiro_p_value_mediano",
                "assimetria_skew",
                "normalidade_aprovada",
            ]
        ].to_string(index=False)
    )
    print(f"\nMelhor aproximacao: {resultado.iloc[0]['serie']}")
    print(f"Arquivos salvos em: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
