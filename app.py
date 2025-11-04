
import streamlit as st

st.set_page_config(
    page_title="Análise Bayesiana dos Dados de Criminalidade no DF",
    page_icon="👮‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
        /* Diminui a largura da sidebar */
        [data-testid="stSidebar"] {
            width: 200px !important;      /* largura fixa */
            min-width: 200px !important;
            max-width: 200px !important;
        }

    </style>
""", unsafe_allow_html=True)

# =======================================================

st.markdown("### Análise Bayesiana dos dados de<br>Criminalidade no DF (2022-2024)", unsafe_allow_html=True)

st.markdown("<br>**Autor:** Robson Ricardo Leite da Silva <br> **Matrícula:** 22112120015 <br> **Curso:** Ciência de Dados e Inteligência Artificial<br> **Disciplina:** Inferência Bayesiana (2°/2025) <br> **Instituição:** IESB - Instituto de Educação Superior de Brasília", unsafe_allow_html=True)

st.divider()

