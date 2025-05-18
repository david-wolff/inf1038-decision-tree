import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def run_disc_model():
    df = pd.read_csv("class_german_credit.csv")
    X = df.drop("Risk", axis=1).copy()
    y = df["Risk"]

    X["Saving accounts"] = X["Saving accounts"].fillna("unknown")
    X["Checking account"] = X["Checking account"].fillna("unknown")

    encoder = OrdinalEncoder()
    X[X.select_dtypes(include="object").columns] = encoder.fit_transform(X.select_dtypes(include="object"))

    discretizer = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    X[X.select_dtypes(include="number").columns] = discretizer.fit_transform(X.select_dtypes(include="number"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))