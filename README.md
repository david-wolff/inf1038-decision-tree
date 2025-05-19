# INF1038 - Aprendizado Automático
### Professora: Karla Figueiredo

## 📝 Trabalho: Análise com Árvores de Decisão na base German Credit

Este projeto aplica um classificador **Decision Tree** na base de crédito alemão, testando os impactos de diferentes pré-processamentos e a seleção de atributos, conforme os métodos abordados em aula:

- Análise descritiva dos dados (tipo, escala e cardinalidade)
- Limpeza de dados (valores ausentes e possíveis outliers)
- Normalização
- Discretização supervisionada
- Seleção de variáveis com base em:
  - Chi²
  - Correlação de Pearson
  - Importância da Árvore
  - RFE com Árvore

## 🌳 Visualização da Árvore de Decisão

Abaixo, a árvore gerada pelo modelo após limpeza de dados:

![Árvore de Decisão](arvore_limpeza.png)

## 📊 Resultados

- Modelo base: **Acurácia = 0.66**
- Modelo com limpeza: **Acurácia = 0.71**
- Modelo com normalização: **Acurácia = 0.71**
- Modelo com discretização: **Acurácia = 0.63**
- Modelo com seleção de variáveis: **Acurácia = 0.703**

A seleção de variáveis resultou em um modelo mais simples, mantendo desempenho próximo do modelo completo.

## 📂 Estrutura dos Arquivos

- `main.py` — executa todos os testes
- `describe_base.py` — análise dos dados da base (tipos, estatísticas, frequências)
- `model_base.py` — modelo sem pré-processamento
- `model_limpeza.py` — modelo com tratamento de valores ausentes
- `model_normalizacao.py` — modelo com dados normalizados
- `model_discretizacao.py` — modelo com discretização
- `model_selecao.py` — análise de seleção de variáveis e modelo com atributos reduzidos
- `class_german_credit.csv` — base de dados
- `requirements.txt` — dependências
- `relatorio_resultados.txt` — resumo das análises (opcional)

## ▶️ Como Executar

```bash
pip install -r requirements.txt
python3 main.py
