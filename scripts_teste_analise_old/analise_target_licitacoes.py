""" analisa possivel target, quantidade de registros e relação com outras variaveis """

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


TARGET_REQUIREMENTS = {
    "min_target_records": 20_000,
    "min_features": 25,
    "min_influential_features": 15,
    "correlation_threshold": 0.30,
}

LIST_FIELD_NAMES = {
    "itensVencedores",
    "participantes",
    "documentosRelacionados",
    "contratos",
    "empenhos",
    "despesas",
}

TEXT_HINTS = (
    "objeto",
    "descricao",
    "fundamento",
    "motivo",
    "endereco",
    "nome",
    "participante",
    "fornecedor",
    "contratado",
)

DATE_HINTS = ("data", "inicio", "termino", "vigencia")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Limpa os JSONs de processos licitatorios, expande itens vencedores "
            "quando necessario e verifica se uma target atende aos requisitos."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Pasta com os arquivos JSON.")
    parser.add_argument(
        "--target",
        default="valorTotalVencedor",
        help="Variavel target. Ex.: valorTotalVencedor ou valorHomologado.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/target_analysis",
        help="Pasta onde os relatorios serao salvos.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=40,
        help="Maximo de categorias distintas para aplicar one-hot em uma coluna.",
    )
    parser.add_argument(
        "--min-category-frequency",
        type=int,
        default=10,
        help="Frequencia minima para uma categoria virar coluna no one-hot.",
    )
    parser.add_argument(
        "--max-text-features",
        type=int,
        default=40,
        help="Maximo de termos TF-IDF por coluna textual selecionada.",
    )
    parser.add_argument(
        "--max-text-columns",
        type=int,
        default=8,
        help="Maximo de colunas textuais usadas em TF-IDF.",
    )
    parser.add_argument(
        "--correlation-target",
        choices=["raw", "log"],
        default="raw",
        help="Base principal para o requisito de influencia: raw ou log.",
    )
    parser.add_argument(
        "--save-clean-csv",
        action="store_true",
        help="Tambem salva a base limpa antes do encoding. Pode gerar arquivo grande.",
    )
    return parser.parse_args()


def infer_year_from_name(path: Path) -> int | None:
    match = re.search(r"(20\d{2})", path.stem)
    return int(match.group(1)) if match else None


def bad_text_score(value: str) -> int:
    return sum(value.count(token) for token in ("Ã", "Â", "�", "ƒ", "€", "œ"))


def repair_text(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value

    best = value
    best_score = bad_text_score(value)
    current = value

    for _ in range(2):
        improved = False
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = current.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue

            candidate_score = bad_text_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
                current = candidate
                improved = True
                break

        if not improved:
            break

    return best.strip() if isinstance(best, str) else best


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return repair_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if not isinstance(value, str):
        return None

    cleaned = repair_text(value).strip()
    cleaned = re.sub(r"^(R\$|BRL)\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", "", cleaned)

    if not re.fullmatch(r"[-+]?(?:\d+|\d{1,3}(?:\.\d{3})+)(?:[,.]\d+)?(?:[eE][-+]?\d+)?", cleaned):
        return None

    if cleaned in {"", "-", ".", ","}:
        return None

    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def flatten_scalar_dict(data: dict[str, Any], prefix: str, skip_keys: set[str] | None = None) -> dict[str, Any]:
    skip_keys = skip_keys or set()
    flattened: dict[str, Any] = {}

    for key, value in data.items():
        if key in skip_keys:
            continue
        column = f"{prefix}__{key}"
        if is_scalar(value):
            flattened[column] = normalize_scalar(value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if is_scalar(nested_value):
                    flattened[f"{column}__{nested_key}"] = normalize_scalar(nested_value)
        elif isinstance(value, list):
            continue
        else:
            flattened[column] = normalize_scalar(value)

    return flattened


def collect_numeric_values(items: list[dict[str, Any]], skip_field: str | None) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)

    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if key == skip_field or isinstance(value, (dict, list)):
                continue
            parsed = to_float(value)
            if parsed is not None:
                values[key].append(parsed)

    return values


def aggregate_list(items: Any, prefix: str, skip_numeric_field: str | None = None) -> dict[str, Any]:
    if not isinstance(items, list):
        return {f"{prefix}__count": 0}

    result: dict[str, Any] = {f"{prefix}__count": len(items)}
    dict_items = [item for item in items if isinstance(item, dict)]
    numeric_values = collect_numeric_values(dict_items, skip_numeric_field)

    for key, values in numeric_values.items():
        series = pd.Series(values, dtype="float64")
        safe_key = f"{prefix}__{key}"
        result[f"{safe_key}__sum"] = float(series.sum())
        result[f"{safe_key}__mean"] = float(series.mean())
        result[f"{safe_key}__max"] = float(series.max())
        result[f"{safe_key}__min"] = float(series.min())

    categorical_values: dict[str, list[str]] = defaultdict(list)
    nested_counts: Counter[str] = Counter()

    for item in dict_items:
        for key, value in item.items():
            if isinstance(value, str):
                cleaned = repair_text(value)
                if cleaned:
                    categorical_values[key].append(cleaned)
            elif isinstance(value, list):
                nested_counts[f"{key}__count"] += len(value)
                nested_dicts = [child for child in value if isinstance(child, dict)]
                nested_numeric = collect_numeric_values(nested_dicts, skip_numeric_field)
                for nested_key, nested_values in nested_numeric.items():
                    metric_key = f"{prefix}__{key}__{nested_key}"
                    result[f"{metric_key}__sum"] = float(np.sum(nested_values))
                    result[f"{metric_key}__mean"] = float(np.mean(nested_values))
                    result[f"{metric_key}__max"] = float(np.max(nested_values))

                for child in nested_dicts:
                    status = child.get("situacao")
                    if isinstance(status, str) and status:
                        status_key = repair_text(status).lower().replace(" ", "_")
                        result[f"{prefix}__{key}__situacao__{status_key}__count"] = (
                            result.get(f"{prefix}__{key}__situacao__{status_key}__count", 0) + 1
                        )

    for key, values in categorical_values.items():
        result[f"{prefix}__{key}__nunique"] = len(set(values))

    for key, count in nested_counts.items():
        result[f"{prefix}__{key}"] = count

    return result


def update_candidate_stats(stats: Counter[str], container: dict[str, Any], prefix: str = "") -> None:
    for key, value in container.items():
        path = f"{prefix}.{key}" if prefix else key
        if "valor" in key.lower() and to_float(value) is not None:
            stats[path] += 1

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    update_candidate_stats(stats, item, path)


def build_base_row(process: dict[str, Any], source_file: Path, target: str) -> dict[str, Any]:
    base = flatten_scalar_dict(process, "proc", skip_keys={target})
    base["anoArquivo"] = infer_year_from_name(source_file)
    base["source_file"] = source_file.name

    for field_name in LIST_FIELD_NAMES:
        aggregate_prefix = f"agg__{field_name}"
        skip_field = target if field_name == "itensVencedores" else None
        base.update(aggregate_list(process.get(field_name), aggregate_prefix, skip_field))

    return base


def load_records(data_dir: Path, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    candidate_stats: Counter[str] = Counter()
    files = sorted(data_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"Nenhum JSON encontrado em {data_dir.resolve()}")

    for file_path in files:
        with file_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError(f"O arquivo {file_path} nao tem uma lista JSON no topo.")

        for process in data:
            if not isinstance(process, dict):
                continue

            update_candidate_stats(candidate_stats, process)
            base = build_base_row(process, file_path, target)
            items = process.get("itensVencedores")
            process_target = to_float(process.get(target))

            item_rows_created = False
            if isinstance(items, list):
                for item_index, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    item_target = to_float(item.get(target))
                    if item_target is None:
                        continue

                    row = dict(base)
                    row["item__index"] = item_index
                    row.update(flatten_scalar_dict(item, "item", skip_keys={target}))
                    row[target] = item_target
                    rows.append(row)
                    item_rows_created = True

            if not item_rows_created and process_target is not None:
                row = dict(base)
                row[target] = process_target
                rows.append(row)

    candidate_df = (
        pd.DataFrame(
            [
                {"campo": field_path, "registros_nao_nulos": count}
                for field_path, count in candidate_stats.items()
            ]
        )
        .sort_values("registros_nao_nulos", ascending=False)
        .reset_index(drop=True)
    )

    return pd.DataFrame(rows), candidate_df


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    converted = series.map(to_float)
    non_null_original = series.notna().sum()
    non_null_converted = converted.notna().sum()

    if non_null_original and non_null_converted / non_null_original >= 0.85:
        return pd.to_numeric(converted, errors="coerce")

    return series


def add_date_features(df: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    date_columns: set[str] = set()
    enriched = df.copy()

    for column in list(enriched.columns):
        lowered = column.lower()
        if not any(hint in lowered for hint in DATE_HINTS):
            continue

        parsed = pd.to_datetime(enriched[column], errors="coerce", utc=True)
        if parsed.notna().sum() < max(20, len(enriched) * 0.05):
            continue

        date_columns.add(column)
        parsed = parsed.dt.tz_convert(None)
        enriched[f"{column}__year"] = parsed.dt.year
        enriched[f"{column}__month"] = parsed.dt.month
        enriched[f"{column}__quarter"] = parsed.dt.quarter
        enriched[f"{column}__dayofweek"] = parsed.dt.dayofweek
        enriched[f"{column}__ordinal"] = parsed.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    date_pairs = [
        ("proc__dataHomologacao", "proc__dataCriacao", "dias_criacao_ate_homologacao"),
        ("proc__dataHomologacao", "proc__dataAberturaEnvelopes", "dias_abertura_ate_homologacao"),
        ("proc__dataJulgamento", "proc__dataCriacao", "dias_criacao_ate_julgamento"),
        ("proc__terminoRecebimentoEnvelopes", "proc__inicioRecebimentoEnvelopes", "dias_recebimento_envelopes"),
        ("proc__dataPublicacao", "proc__dataCriacao", "dias_criacao_ate_publicacao"),
    ]

    for end_column, start_column, output_column in date_pairs:
        if end_column not in enriched.columns or start_column not in enriched.columns:
            continue
        end_date = pd.to_datetime(enriched[end_column], errors="coerce", utc=True).dt.tz_convert(None)
        start_date = pd.to_datetime(enriched[start_column], errors="coerce", utc=True).dt.tz_convert(None)
        enriched[output_column] = (end_date - start_date).dt.total_seconds() / 86_400

    return enriched, date_columns


def prepare_dataframe(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, set[str]]:
    prepared = df.copy()

    for column in list(prepared.columns):
        if column == target:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            continue
        prepared[column] = coerce_numeric_series(prepared[column])

    prepared, date_columns = add_date_features(prepared)
    return prepared, date_columns


def profile_columns(df: pd.DataFrame, target: str, output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    total_rows = len(df)

    for column in df.columns:
        sample_value = df[column].dropna().astype(str).head(1)
        rows.append(
            {
                "coluna": column,
                "dtype": str(df[column].dtype),
                "target": column == target,
                "nao_nulos": int(df[column].notna().sum()),
                "missing_pct": round(float(df[column].isna().mean()), 4) if total_rows else 0,
                "unicos": int(df[column].nunique(dropna=True)),
                "exemplo": sample_value.iloc[0][:120] if not sample_value.empty else "",
            }
        )

    pd.DataFrame(rows).sort_values(["target", "nao_nulos"], ascending=[False, False]).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )


def is_target_alias(column: str, target: str) -> bool:
    normalized_column = column.lower().replace("_", "")
    normalized_target = target.lower().replace("_", "")
    return normalized_column.endswith(normalized_target) or normalized_column == normalized_target


def select_feature_columns(
    df: pd.DataFrame,
    target: str,
    date_columns: set[str],
    max_categories: int,
    max_text_columns: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    excluded = {target, "source_file"} | date_columns
    feature_candidates = [column for column in df.columns if column not in excluded and not is_target_alias(column, target)]

    numeric_columns = [
        column
        for column in feature_candidates
        if pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().sum() >= 20 and df[column].nunique(dropna=True) > 1
    ]

    object_columns = [
        column
        for column in feature_candidates
        if not pd.api.types.is_numeric_dtype(df[column]) and df[column].notna().sum() >= 20
    ]

    categorical_columns: list[str] = []
    text_scores: list[tuple[int, str]] = []
    skipped_columns: list[str] = []

    for column in object_columns:
        non_null = df[column].dropna().astype(str)
        unique_count = non_null.nunique()
        average_length = non_null.map(len).mean() if not non_null.empty else 0
        lowered = column.lower()
        has_text_hint = any(hint in lowered for hint in TEXT_HINTS)

        if 1 < unique_count <= max_categories and average_length <= 80:
            categorical_columns.append(column)
        elif has_text_hint or average_length > 40:
            score = int(has_text_hint) * 10_000 + int(non_null.notna().sum()) - int(unique_count > 2_000) * 1_000
            text_scores.append((score, column))
        else:
            skipped_columns.append(column)

    text_columns = [column for _, column in sorted(text_scores, reverse=True)[:max_text_columns]]
    return numeric_columns, categorical_columns, text_columns, skipped_columns


def build_feature_matrix(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    text_columns: list[str],
    min_category_frequency: int,
    max_text_features: int,
) -> tuple[Any, list[str], pd.DataFrame]:
    matrices = []
    feature_names: list[str] = []
    feature_sources: list[dict[str, str]] = []

    if numeric_columns:
        numeric_df = df[numeric_columns].copy()
        missing_indicators: dict[str, pd.Series] = {}
        for column in numeric_columns:
            if numeric_df[column].isna().mean() > 0.01:
                indicator = f"{column}__missing"
                missing_indicators[indicator] = numeric_df[column].isna().astype(int)
                feature_sources.append({"feature": indicator, "tipo": "missing_indicator", "coluna_origem": column})
            median = numeric_df[column].median()
            numeric_df[column] = numeric_df[column].fillna(0 if pd.isna(median) else median)

        if missing_indicators:
            numeric_df = pd.concat([numeric_df, pd.DataFrame(missing_indicators, index=numeric_df.index)], axis=1)

        numeric_feature_names = list(numeric_df.columns)
        matrices.append(csr_matrix(numeric_df.to_numpy(dtype="float64")))
        feature_names.extend(numeric_feature_names)
        for column in numeric_feature_names:
            if not column.endswith("__missing"):
                feature_sources.append({"feature": column, "tipo": "numerica", "coluna_origem": column})

    if categorical_columns:
        categorical_df = df[categorical_columns].fillna("__missing__").astype(str)
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_category_frequency,
            sparse_output=True,
        )
        encoded = encoder.fit_transform(categorical_df)
        encoded_names = encoder.get_feature_names_out(categorical_columns).tolist()
        matrices.append(encoded)
        feature_names.extend(encoded_names)
        for feature in encoded_names:
            origin = next((column for column in categorical_columns if feature.startswith(f"{column}_")), feature)
            feature_sources.append({"feature": feature, "tipo": "one_hot", "coluna_origem": origin})

    for column in text_columns:
        text = df[column].fillna("").astype(str)
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            min_df=5,
            max_df=0.95,
            max_features=max_text_features,
            ngram_range=(1, 2),
        )
        try:
            matrix = vectorizer.fit_transform(text)
        except ValueError:
            continue

        names = [f"tfidf__{column}__{term}" for term in vectorizer.get_feature_names_out()]
        matrices.append(matrix)
        feature_names.extend(names)
        for feature in names:
            feature_sources.append({"feature": feature, "tipo": "tfidf", "coluna_origem": column})

    if not matrices:
        raise ValueError("Nenhuma feature valida foi gerada para comparar com a target.")

    return hstack(matrices, format="csr"), feature_names, pd.DataFrame(feature_sources)


def sparse_correlations(matrix: Any, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    valid_y = np.isfinite(y)
    matrix = matrix[valid_y]
    y = y[valid_y]

    if len(y) < 2 or np.nanstd(y) == 0:
        return np.full(matrix.shape[1], np.nan)

    y_centered = y - y.mean()
    y_std = np.sqrt(np.mean(y_centered**2))
    x_mean = np.asarray(matrix.mean(axis=0)).ravel()
    x_sq_mean = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    x_var = np.maximum(x_sq_mean - x_mean**2, 0)
    x_std = np.sqrt(x_var)
    covariance = np.asarray(matrix.T.dot(y_centered)).ravel() / len(y)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = covariance / (x_std * y_std)

    correlations[~np.isfinite(correlations)] = np.nan
    return correlations


def possible_leakage(feature: str, target: str) -> bool:
    lowered = feature.lower()
    target_lower = target.lower()
    leakage_terms = ("vencedor", "homologado", "referencia", "valorunitario", "valortotal")
    return any(term in lowered for term in leakage_terms) and target_lower.startswith("valor")


def analyze_correlations(
    matrix: Any,
    feature_names: list[str],
    feature_sources: pd.DataFrame,
    target_values: pd.Series,
    main_target: str,
) -> pd.DataFrame:
    y_raw = target_values.to_numpy(dtype="float64")
    y_log = np.log1p(np.clip(y_raw, a_min=0, a_max=None))

    corr_raw = sparse_correlations(matrix, y_raw)
    corr_log = sparse_correlations(matrix, y_log)

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "correlacao_raw": corr_raw,
            "correlacao_log": corr_log,
        }
    )
    result = result.merge(feature_sources, on="feature", how="left")
    chosen_column = "correlacao_log" if main_target == "log" else "correlacao_raw"
    result["correlacao_usada"] = result[chosen_column]
    result["abs_correlacao_usada"] = result["correlacao_usada"].abs()
    result["possivel_vazamento_target"] = result["feature"].map(lambda feature: possible_leakage(feature, target_values.name))

    return result.sort_values("abs_correlacao_usada", ascending=False).reset_index(drop=True)


def build_requirement_summary(
    df: pd.DataFrame,
    target: str,
    correlations: pd.DataFrame,
    main_target: str,
) -> dict[str, Any]:
    threshold = TARGET_REQUIREMENTS["correlation_threshold"]
    usable_correlations = correlations["abs_correlacao_usada"].dropna()

    return {
        "target": target,
        "correlation_target": main_target,
        "total_registros_modelagem": int(len(df)),
        "registros_target_nao_nulos": int(df[target].notna().sum()),
        "features_geradas": int(len(correlations)),
        "features_com_abs_correlacao_maior_igual_0_30": int((usable_correlations >= threshold).sum()),
        "requisitos": TARGET_REQUIREMENTS,
        "atende_minimo_20k_registros": bool(df[target].notna().sum() >= TARGET_REQUIREMENTS["min_target_records"]),
        "atende_minimo_25_variaveis": bool(len(correlations) >= TARGET_REQUIREMENTS["min_features"]),
        "atende_minimo_15_influentes": bool(
            (usable_correlations >= threshold).sum() >= TARGET_REQUIREMENTS["min_influential_features"]
        ),
    }


def print_summary(summary: dict[str, Any], correlations: pd.DataFrame, candidate_df: pd.DataFrame) -> None:
    print("\n=== Resumo da target ===")
    print(f"Target: {summary['target']}")
    print(f"Registros com target: {summary['registros_target_nao_nulos']:,}".replace(",", "."))
    print(f"Features geradas para comparacao: {summary['features_geradas']:,}".replace(",", "."))
    print(
        "Features com |correlacao| >= 0.30: "
        f"{summary['features_com_abs_correlacao_maior_igual_0_30']:,}".replace(",", ".")
    )
    print(f"Atende 20k registros? {summary['atende_minimo_20k_registros']}")
    print(f"Atende 25 variaveis? {summary['atende_minimo_25_variaveis']}")
    print(f"Atende 15 influentes? {summary['atende_minimo_15_influentes']}")

    print("\n=== Top 15 correlacoes ===")
    top_columns = [
        "feature",
        "tipo",
        "coluna_origem",
        "correlacao_raw",
        "correlacao_log",
        "possivel_vazamento_target",
    ]
    print(correlations[top_columns].head(15).to_string(index=False))

    if not candidate_df.empty:
        print("\n=== Candidatas contendo 'valor' por quantidade de registros ===")
        print(candidate_df.head(15).to_string(index=False))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo JSONs em: {data_dir.resolve()}")
    raw_df, candidate_df = load_records(data_dir, args.target)
    if raw_df.empty:
        raise ValueError(
            f"Nenhum registro foi gerado para a target '{args.target}'. "
            "Confira se ela esta no processo ou em itensVencedores."
        )

    print(f"Registros brutos gerados para a target: {len(raw_df):,}".replace(",", "."))
    df, date_columns = prepare_dataframe(raw_df, args.target)
    df = df[df[args.target].notna()].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"A target '{args.target}' existe, mas nao tem valores numericos validos.")

    candidate_df.to_csv(output_dir / "candidatas_valor_contagem.csv", index=False, encoding="utf-8-sig")
    profile_columns(df, args.target, output_dir / f"perfil_colunas_{args.target}.csv")

    if args.save_clean_csv:
        df.to_csv(output_dir / f"base_limpa_{args.target}.csv", index=False, encoding="utf-8-sig")

    numeric_columns, categorical_columns, text_columns, skipped_columns = select_feature_columns(
        df=df,
        target=args.target,
        date_columns=date_columns,
        max_categories=args.max_categories,
        max_text_columns=args.max_text_columns,
    )

    selection_report = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "text_columns": text_columns,
        "skipped_object_columns": skipped_columns,
        "date_columns_raw_excluded": sorted(date_columns),
    }
    with (output_dir / f"selecao_features_{args.target}.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_report, handle, ensure_ascii=False, indent=2)

    matrix, feature_names, feature_sources = build_feature_matrix(
        df=df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        text_columns=text_columns,
        min_category_frequency=args.min_category_frequency,
        max_text_features=args.max_text_features,
    )

    correlations = analyze_correlations(
        matrix=matrix,
        feature_names=feature_names,
        feature_sources=feature_sources,
        target_values=df[args.target],
        main_target=args.correlation_target,
    )
    correlations.to_csv(output_dir / f"correlacoes_{args.target}.csv", index=False, encoding="utf-8-sig")

    summary = build_requirement_summary(df, args.target, correlations, args.correlation_target)
    with (output_dir / f"resumo_requisitos_{args.target}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print_summary(summary, correlations, candidate_df)
    print(f"\nRelatorios salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
