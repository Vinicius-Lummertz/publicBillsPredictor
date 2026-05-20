# Teste de Normalidade da Target

Target analisada: `valorTotalVencedor`

Total de registros validos: `110675`

O Shapiro-Wilk foi aplicado em `10` amostras aleatorias de ate `2000` registros, porque a base completa e grande demais para a faixa recomendada em aula para o SW Teste.

## Hipoteses

- H0: a serie segue uma distribuicao normal.
- H1: a serie nao segue uma distribuicao normal.
- Criterio: se `p-value > 0.05`, nao rejeitamos H0; se `p-value <= 0.05`, rejeitamos H0.

## Resultado

Target original:

- W mediano: `0.037156`
- p-value mediano: `4.987609e-72`
- Assimetria: `131.313275`
- Outliers pelo IQR: `17335`

Melhor transformacao encontrada:

- Serie: `log1p`
- Metodo: Aplica log(1 + x). Reduz a cauda de valores monetarios muito altos.
- W mediano: `0.992210`
- p-value mediano: `1.768837e-08`
- Assimetria: `0.312158`
- Outliers pelo IQR: `1501`

## Conclusao

Nenhuma versao da target foi aprovada formalmente no Shapiro-Wilk com alpha = 0.05. Mesmo assim, a melhor aproximacao foi `log1p`, pois apresentou o maior p-value mediano e W mediano.

Para o projeto, isso indica que `valorTotalVencedor` e uma variavel monetaria muito assimetrica, com cauda longa a direita. A transformacao `log1p` e a mais defensavel para a proxima etapa de modelagem, porque reduziu bastante a assimetria e aproximou mais a distribuicao de uma normal, mesmo sem passar formalmente no teste.
