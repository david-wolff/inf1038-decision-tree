# INF1038 - Aprendizado Automático
### Professora: Karla Figueiredo

## 📝 Trabalho: Análise com Árvores de Decisão na base German Credit

Este projeto aplica um classificador **Decision Tree** na base de crédito alemão, testando os impactos de pré-processamentos como:
- Limpeza de dados
- Normalização
- Discretização

## 🌳 Visualização da Árvore de Decisão

Abaixo, a árvore gerada pelo modelo com limpeza de dados:

![Árvore de Decisão](arvore_limpeza.png)
## 📂 Estrutura dos Arquivos

- `main.py` — executa todos os testes
- `model_base.py` — modelo sem pré-processamento
- `model_limpeza.py` — modelo com tratamento de valores ausentes
- `model_normalizacao.py` — modelo com dados normalizados
- `model_discretizacao.py` — modelo com discretização dos atributos
- `class_german_credit.csv` — base de dados
- `requirements.txt` — dependências
- `relatorio_resultados.txt` — análise final dos testes (opcional)

## ▶️ Como Executar

```bash
pip install -r requirements.txt
python3 main.py