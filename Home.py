import streamlit as st
import json
import pandas as pd

# =====================================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================================

st.set_page_config(
    page_title="Análise Bayesiana - PMDF",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CABEÇALHO
# =====================================================================

st.markdown("<h3>Análise Bayesiana dos Dados <br>de Criminalidade no DF</h3>", unsafe_allow_html=True)
st.caption("Previsão de Ocorrências Policiais em 2025 usando Modelagem Bayesiana Hierárquica")
st.markdown("<br>", unsafe_allow_html=True)

st.warning(f"**Use o menu lateral para navegar entre as seções da análise**", icon="⬅️")

# =====================================================================
# INFORMAÇÕES DO TRABALHO
# =====================================================================

col1, col2 = st.columns([2, 1], border=True, gap="small")

with col1:
    st.markdown("<b>Informações do Trabalho</b>", unsafe_allow_html=True)
    st.markdown("""
    **Autor:** Robson Ricardo Leite da Silva  
    **Matrícula:** 22112120015  
    **Curso:** Ciência de Dados e Inteligência Artificial  
    **Disciplina:** Inferência Bayesiana (2°/2025)  
    **Instituição:** IESB - Instituto de Educação Superior de Brasília
    """)

with col2:
    st.markdown("<b>Objetivo</b>", unsafe_allow_html=True)
    st.markdown("""
    Aplicar **Modelagem Bayesiana Hierárquica** para prever 
    a taxa mensal esperada de ocorrências criminais atendidas 
    pela PMDF no Distrito Federal em **2025**.
    """)

# =====================================================================
# PROBLEMA DE PESQUISA
# =====================================================================

with st.container(border=True):
    st.markdown("<b>Problema de Pesquisa</b>", unsafe_allow_html=True)
    st.markdown("A Polícia Militar do Distrito Federal (PMDF) atende milhares de ocorrências mensalmente, variando de crimes graves a chamados de emergência diversos. O planejamento operacional eficiente depende de predições confiáveis da demanda futura. Neste trabalho, busca-se responder a seguinte pergunta de pesquisa:", unsafe_allow_html=True)

    st.info("""
    **Qual a taxa mensal esperada de ocorrências criminais atendidas pela PMDF 
    no Distrito Federal em 2025?**
    """)
    st.markdown("Para responder a essa pergunta, será utilizada uma abordagem de modelagem bayesiana hierárquica, incorporando dados históricos de ocorrências e possíveis covariáveis explicativas. O objetivo é fornecer previsões robustas que possam auxiliar na alocação de recursos e no planejamento estratégico da PMDF.", unsafe_allow_html=True)


# =====================================================================
# METODOLOGIA (SIMPLIFICADA)
# =====================================================================

with st.container(border=True):
    st.markdown("<b>Metodologia</b>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📊 Dados**
        - Período: 2022-2024
        - Fonte: PMDF (Dados Abertos DF)
        - Granularidade: Mensal
        """)

    with col2:
        st.markdown("""
        **🧮 Modelo**
        - Família: Binomial Negativa
        - Estrutura: GLM Hierárquico
        - Inferência: MCMC (NUTS)
        """)

    with col3:
        st.markdown("""
        **🎯 Resultado**
        - Taxa mensal prevista
        - Intervalos de credibilidade
        - Previsão out-of-sample 2025
        """)

with st.container(border=True):
    st.markdown("<b>Fonte dos Dados</b>", unsafe_allow_html=True)
    st.markdown("""
    Dados obtidos através do **Portal de Dados Abertos do Distrito Federal**, 
    dos Relatórios Mensais de Produtividade da PMDF (2022-2024).
    """)
    
    st.link_button(
        "🔗 Dados Abertos PMDF",
        "https://dados.df.gov.br/pt_BR/organization/pmdf-policia-militar-do-distrito-federal",
        use_container_width=True
    )


with st.container(border=True):
    st.markdown("<b>Ferramentas Utilizadas</b>", unsafe_allow_html=True)
    st.markdown("""
    - **Linguagem de Programação:** Python  
    - **Bibliotecas:** PyMC, ArviZ, Pandas, NumPy, Plotly, Streamlit  
    - **Ambiente de Desenvolvimento:** Jupyter Notebook, Visual Studio Code no Linux Manjaro 
    - **Plataforma de Visualização:** Streamlit Community Cloud 
    """)

