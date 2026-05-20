from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "finais"

STEPS = [
    {
        "name": "01 - Target e relacoes",
        "script": "01_target_relacoes.py",
        "summary": "summary_target_relacoes",
    },
    {
        "name": "02 - SW Teste de normalidade",
        "script": "02_sw_normalidade.py",
        "summary": "summary_sw_normalidade",
    },
    {
        "name": "03 - Regressao OLS",
        "script": "03_regressao_ols.py",
        "summary": "summary_ols",
    },
]


def line(char: str = "=", size: int = 78) -> str:
    return char * size


def title(text: str) -> None:
    print("\n" + line("="), flush=True)
    print(text, flush=True)
    print(line("="), flush=True)


def section(text: str) -> None:
    print("\n" + line("-"), flush=True)
    print(text, flush=True)
    print(line("-"), flush=True)


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", ".")


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def fmt_money(value: float) -> str:
    formatted = f"{float(value):,.2f}"
    return "R$ " + formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def run_script(script_name: str) -> None:
    script_path = ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script nao encontrado: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)


def summary_target_relacoes() -> None:
    out_dir = REPORTS / "01_target_relacoes"
    correlacoes = pd.read_csv(out_dir / "correlacoes_target.csv")
    fortes = pd.read_csv(out_dir / "correlacoes_fortes.csv")
    candidatas = pd.read_csv(out_dir / "candidatas_valor.csv")

    target_row = candidatas[candidatas["campo"].str.endswith("valorTotalVencedor", na=False)].head(1)
    registros_target = int(target_row["registros_nao_nulos"].iloc[0]) if not target_row.empty else 0
    top = correlacoes.iloc[0]
    vazamento = int(fortes["possivel_vazamento_target"].sum())

    section("Conclusao da etapa 01")
    print(f"Target escolhida: valorTotalVencedor")
    print(f"Registros validos da target: {fmt_int(registros_target)}")
    print(f"Features analisadas: {fmt_int(len(correlacoes))}")
    print(f"Features fortes, com |correlacao| >= 0,30: {fmt_int(len(fortes))}")
    print(f"Features fortes marcadas como possivel vazamento: {fmt_int(vazamento)}")
    print(
        "Maior correlacao encontrada: "
        f"{top['feature']} ({fmt_float(top['correlacao_usada'], 4)})"
    )
    print("Leitura: a target passa nos requisitos de volume e correlacao, mas parte das melhores variaveis sao valores muito proximos dela.")


def summary_sw_normalidade() -> None:
    out_dir = REPORTS / "02_sw_normalidade"
    resultado = pd.read_csv(out_dir / "resultado_sw.csv")
    best = resultado.iloc[0]
    original = resultado[resultado["serie"] == "original"].iloc[0]

    section("Conclusao da etapa 02")
    print("Teste usado: Shapiro-Wilk em 10 amostras de 2.000 registros")
    print(
        "Target original: "
        f"W={fmt_float(original['shapiro_w_mediano'], 6)}, "
        f"p-value={original['shapiro_p_value_mediano']:.3e}, "
        f"skew={fmt_float(original['assimetria_skew'], 2)}"
    )
    print(
        "Melhor transformacao: "
        f"{best['serie']} com W={fmt_float(best['shapiro_w_mediano'], 6)}, "
        f"p-value={best['shapiro_p_value_mediano']:.3e}, "
        f"skew={fmt_float(best['assimetria_skew'], 3)}"
    )
    if bool(best["normalidade_aprovada"]):
        print("Leitura: a melhor transformacao passou no criterio p-value > 0,05.")
    else:
        print("Leitura: nenhuma versao passou formalmente no SW, mas log1p aproximou muito melhor a distribuicao de uma normal.")


def summary_ols() -> None:
    out_dir = REPORTS / "03_regressao_ols"
    with (out_dir / "resumo_ols.json").open("r", encoding="utf-8") as handle:
        resumo = json.load(handle)

    section("Conclusao da etapa 03")
    print(f"Populacao: {fmt_int(resumo['populacao'])}")
    print(f"Amostra estatistica indicada: {fmt_int(resumo['amostra_indicada'])}")
    print(f"Amostra usada no OLS: {fmt_int(resumo['amostra_usada'])}")
    print(f"Treino/teste: {fmt_int(resumo['treino'])} / {fmt_int(resumo['teste'])}")
    print(f"Features usadas: {fmt_int(resumo['features'])}")
    print(f"Features com possivel vazamento: {fmt_int(resumo['features_com_possivel_vazamento'])}")
    print(f"R2 treino: {fmt_float(resumo['r2_treino_ols'], 4)}")
    print(f"R2 ajustado treino: {fmt_float(resumo['r2_ajustado_treino_ols'], 4)}")
    print(f"R2 teste: {fmt_float(resumo['teste_r2'], 4)}")
    print(f"MAE teste: {fmt_money(resumo['teste_mae'])}")
    print(f"RMSE teste: {fmt_money(resumo['teste_rmse'])}")
    print("Leitura: usar a base completa foi melhor para previsao; a amostra minima serve para justificar a parte estatistica.")


def final_summary() -> None:
    title("Resumo final")
    print("1. A target definida foi valorTotalVencedor, com volume suficiente para o projeto.")
    print("2. A distribuicao original nao e normal; log1p foi a melhor aproximacao no SW Teste.")
    print("3. O OLS teve melhor desempenho usando a base completa e as variaveis mais correlacionadas.")
    print("4. Ha possivel vazamento em varias variaveis de valor, entao isso deve ser citado como limitacao metodologica.")
    print(f"\nRelatorios finais em: {REPORTS}")


def main() -> None:
    title("Pipeline - Preditor de Contas Publicas")
    print("Este pipeline executa as 3 etapas finais do projeto em sequencia.", flush=True)

    for step in STEPS:
        title(f"Rodando {step['name']}")
        run_script(step["script"])
        globals()[step["summary"]]()

    final_summary()


if __name__ == "__main__":
    main()
