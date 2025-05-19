# model_base.py
import pandas as pd

def run_describe_base():
    df = pd.read_csv("class_german_credit.csv")
    
    df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
    df["Checking account"] = df["Checking account"].fillna("unknown")


    atributos = []
    for coluna in df.columns:
        if coluna == "Risk":
            continue 

        tipo_dado = str(df[coluna].dtype)
        n_valores = df[coluna].nunique()

        if tipo_dado == "object":
            tipo = "Categórico"
            escala = "Nominal"
            cardinalidade = "Binária" if n_valores == 2 else "Discreta"
        else:
            tipo = "Numérico"
            escala = "Razão"
            cardinalidade = "Contínua" if n_valores > 10 else "Discreta"

        atributos.append({
            "Atributo": coluna,
            "Tipo de Dado": tipo,
            "Escala": escala,
            "Cardinalidade": cardinalidade
        })

    df_atributos = pd.DataFrame(atributos)
    print(df_atributos.to_string(index=False))

    print("\n\t=== Estatísticas para atributos numéricos ===")
    stats_numericas = df.select_dtypes(include=["number"]).describe().T.reset_index().rename(columns={"index": "Atributo"})
    print(stats_numericas.to_string(index=False))

    print("\n\t=== Frequência para atributos categóricos ===")
    freqs = []
    df_categorico = df.select_dtypes(include=["object"])
    for col in df_categorico.columns:
        valores = df_categorico[col].value_counts().reset_index()
        valores.columns = ["Valor", "Frequência"]
        valores["Atributo"] = col
        freqs.append(valores)
    
    freqs_df = pd.concat(freqs, ignore_index=True)
    for atributo in freqs_df["Atributo"].unique():
        print(f"\n\t-- {atributo} --")
        print(freqs_df[freqs_df["Atributo"] == atributo][["Valor", "Frequência"]].to_string(index=False))
