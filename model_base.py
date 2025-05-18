# model_base.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report

def run_base_model():
    df = pd.read_csv("class_german_credit.csv")
    X = df.drop("Risk", axis=1).fillna("missing")
    y = df["Risk"]

    encoder = OrdinalEncoder()
    X[X.select_dtypes(include="object").columns] = encoder.fit_transform(X.select_dtypes(include="object"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))