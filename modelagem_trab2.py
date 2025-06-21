# trabalho2_final.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")  # Remove avisos visuais de convergência e RFE

def executar_trabalho2():
    # === 1. Carregamento e limpeza ===
    df = pd.read_csv("class_german_credit.csv")
    df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
    df["Checking account"] = df["Checking account"].fillna("unknown")

    X = df.drop("Risk", axis=1)
    y = df["Risk"]

    # === 2. Codificação das variáveis categóricas ===
    encoder = OrdinalEncoder()
    X[X.select_dtypes(include="object").columns] = encoder.fit_transform(X.select_dtypes(include="object"))

    # === 3. Normalização para o RFE (LogisticRegression precisa) ===
    X_norm_rfe = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    # === 4. Seleção de atributos via RFE com validação de desempenho ===
    melhor_score = 0
    melhor_k = 0
    melhor_atributos = []

    for k in range(4, len(X.columns) + 1):  # Limita a no máximo o nº de colunas
        modelo_rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=k)
        X_sel_temp = modelo_rfe.fit_transform(X_norm_rfe, y)
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
            X_sel_temp, y, test_size=0.3, stratify=y, random_state=42
        )
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_temp, y_train_temp)
        y_pred_temp = rf.predict(X_test_temp)
        score = accuracy_score(y_test_temp, y_pred_temp)
        if score > melhor_score:
            melhor_score = score
            melhor_k = k
            melhor_atributos = X.columns[modelo_rfe.support_].tolist()

    print(f"\n>>> Atributos selecionados via RFE (k={melhor_k}): {melhor_atributos}\n")

    # === 5. Seleção final de atributos e divisão treino/teste ===
    X_sel = X[melhor_atributos]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.3, stratify=y, random_state=42
    )

    # === 6. Normalização para modelos que exigem (kNN e NB) ===
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # === 7. Definição dos modelos ===
    modelos = {
        "Decision Tree (balanced)": (
            DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=42),
            X_train, X_test),
        "Random Forest (balanced)": (
            RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
            X_train, X_test),
        "Naive Bayes": (
            GaussianNB(),
            X_train_norm, X_test_norm),
        "kNN (k=7)": (
            KNeighborsClassifier(n_neighbors=7),
            X_train_norm, X_test_norm),
    }

    resultados = []

    # === 8. Avaliação ===
    for nome, (modelo, X_tr, X_te) in modelos.items():
        modelo.fit(X_tr, y_train)
        y_pred = modelo.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print(f"--- {nome} ---")
        print(f"Acurácia: {round(acc, 3)}")
        print(f"Recall (bad): {round(report['bad']['recall'], 3)}")
        print(f"F1-score (bad): {round(report['bad']['f1-score'], 3)}\n")

        resultados.append({
            "Modelo": nome,
            "Acurácia": round(acc, 3),
            "Recall_good": round(report["good"]["recall"], 3),
            "Recall_bad": round(report["bad"]["recall"], 3),
            "F1_good": round(report["good"]["f1-score"], 3),
            "F1_bad": round(report["bad"]["f1-score"], 3),
        })

    # === 9. Resumo final ===
    print("========== RESUMO FINAL ==========")
    for r in resultados:
        print(f"{r['Modelo']}: Acurácia={r['Acurácia']} | Recall_bad={r['Recall_bad']} | F1_bad={r['F1_bad']}")

if __name__ == "__main__":
    executar_trabalho2()
