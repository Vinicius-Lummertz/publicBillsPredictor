from pathlib import Path

from scripts_teste_analise_old.analise_target_licitacoes import (
    analyze_correlations,
    build_feature_matrix,
    load_records,
    prepare_dataframe,
    select_feature_columns,
)


TARGET = "valorTotalVencedor"
DATA_DIR = Path("data")
OUT_DIR = Path("reports/finais/01_target_relacoes")
CORRELATION_THRESHOLD = 0.30


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw, candidatas = load_records(DATA_DIR, TARGET)
    df, date_columns = prepare_dataframe(df_raw, TARGET)
    df = df[df[TARGET].notna()].reset_index(drop=True)

    numeric_cols, categorical_cols, text_cols, _ = select_feature_columns(
        df=df,
        target=TARGET,
        date_columns=date_columns,
        max_categories=40,
        max_text_columns=8,
    )
    matrix, feature_names, feature_sources = build_feature_matrix(
        df=df,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        text_columns=text_cols,
        min_category_frequency=10,
        max_text_features=40,
    )
    correlacoes = analyze_correlations(
        matrix=matrix,
        feature_names=feature_names,
        feature_sources=feature_sources,
        target_values=df[TARGET],
        main_target="raw",
    )

    fortes = correlacoes[correlacoes["abs_correlacao_usada"] >= CORRELATION_THRESHOLD]

    candidatas.to_csv(OUT_DIR / "candidatas_valor.csv", index=False, encoding="utf-8-sig")
    correlacoes.to_csv(OUT_DIR / "correlacoes_target.csv", index=False, encoding="utf-8-sig")
    fortes.to_csv(OUT_DIR / "correlacoes_fortes.csv", index=False, encoding="utf-8-sig")

    print("=== Target e relacoes ===")
    print(f"Target escolhida: {TARGET}")
    print(f"Registros validos da target: {len(df):,}".replace(",", "."))
    print(f"Features geradas: {len(correlacoes):,}".replace(",", "."))
    print(f"Features com |correlacao| >= {CORRELATION_THRESHOLD}: {len(fortes)}")
    print("\nTop 10 correlacoes:")
    print(
        correlacoes[
            ["feature", "correlacao_usada", "possivel_vazamento_target"]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nArquivos salvos em: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
