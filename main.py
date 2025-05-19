# main.py
from model_base import run_base_model
from model_limpeza import run_clean_model
from model_normalizacao import run_norm_model
from model_discretizacao import run_disc_model
from describe_base import run_describe_base
from model_selecao import run_selection_analysis

if __name__ == "__main__":

    print("=== DADOS DA BASE ===")
    run_describe_base()

    print("\n=== MODELO BASE ===")
    run_base_model()

    print("\n=== MODELO COM LIMPEZA ===")
    run_clean_model()

    print("\n=== MODELO COM NORMALIZACAO ===")
    run_norm_model()

    print("\n=== MODELO COM DISCRETIZACAO ===")
    run_disc_model()

    print("\n=== SELEÇÃO DE VARIÁVEIS ===")
    run_selection_analysis()
