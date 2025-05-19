import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from scipy.stats import pearsonr
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

def run_selection_analysis():
    # Carregar a base
    df = pd.read_csv("class_german_credit.csv")
    df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
    df["Checking account"] = df["Checking account"].fillna("unknown")

    X = df.drop("Risk", axis=1)
    y = df["Risk"]

    # Codificar atributos categóricos
    encoder = OrdinalEncoder()
    X_encoded = X.copy()
    X_encoded[X.select_dtypes(include="object").columns] = encoder.fit_transform(
        X.select_dtypes(include="object"))

    # Chi2 (Filter)
    chi2_selector = SelectKBest(score_func=chi2, k="all")
    chi2_selector.fit(X_encoded, y)
    chi2_scores = chi2_selector.scores_

    # Correlação de Pearson (Filter)
    pearson_scores = []
    y_numeric = y.apply(lambda val: 1 if val == "good" else 0)
    for col in X_encoded.columns:
        if np.issubdtype(X_encoded[col].dtype, np.number):
            corr, _ = pearsonr(X_encoded[col], y_numeric)
            pearson_scores.append(abs(corr))
        else:
            pearson_scores.append(0)

    # Importância da árvore (Embedded)
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_encoded, y)
    importances = tree.feature_importances_

    # RFE com árvore (Wrapper)
    rfe = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=1)
    rfe.fit(X_encoded, y)
    rfe_ranking = rfe.ranking_

    # Tabela final
    resultados = pd.DataFrame({
        "Atributo": X.columns,
        "Chi2": chi2_scores,
        "Pearson": pearson_scores,
        "Importância_Árvore": importances,
        "Ranking_RFE": rfe_ranking
    })

    print("=== Análise de Seleção de Variáveis ===")
    print(resultados.sort_values(by="Ranking_RFE").to_string(index=False))

    # Selecionar 5 melhores atributos
    melhores = resultados.sort_values(by="Ranking_RFE").head(5)["Atributo"].tolist()
    print("\nAtributos Selecionados:", melhores)

    # Treinar modelo com atributos selecionados
    X_sel = X_encoded[melhores]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print("\nAcurácia com atributos selecionados:", round(accuracy, 3))
