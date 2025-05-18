# model_limpeza.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def run_clean_model():
    df = pd.read_csv("class_german_credit.csv")
    X = df.drop("Risk", axis=1).copy()
    y = df["Risk"]

    # Tratar valores ausentes
    X["Saving accounts"] = X["Saving accounts"].fillna("unknown")
    X["Checking account"] = X["Checking account"].fillna("unknown")

    # Codificação
    encoder = OrdinalEncoder()
    X[X.select_dtypes(include="object").columns] = encoder.fit_transform(X.select_dtypes(include="object"))

    # Separar treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar modelo
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Avaliação
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Visualização e salvamento
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
    plt.title("Árvore de Decisão - Modelo com Limpeza")
    plt.savefig("arvore_limpeza.png")
    plt.show()