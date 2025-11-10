import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from functions import load_data
import warnings
warnings.filterwarnings('ignore')
import pickle


# =====================================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================================

st.set_page_config(
    page_title="Modelos Bayesianos - PMDF",
    page_icon="👮‍♂️",
    layout="wide"
)
# Cores padrão
colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}

# =====================================================================
# CARREGAMENTO DOS RESULTADOS
# =====================================================================

@st.cache_data
def carregar_resultados():
    """Carrega resultados salvos do notebook Jupyter"""
    with open('data/bayes/resultados_bayesianos_completos.pkl', 'rb') as f:
        return pickle.load(f)

# Carregar dados
resultados = carregar_resultados()

# Extrair componentes principais
info_geral = resultados['info_geral']
dados_originais = resultados['dados_originais']
power_prior = resultados['power_prior_params']
modelos = resultados['modelos']
comparacao = resultados['comparacao_modelos']
predicoes = resultados['predicoes_2025']
sensibilidade = resultados['analise_sensibilidade']

def formatar_numero_br(valor):
    return f"{valor:,.0f}".replace(",", ".")

# =====================================================================
# CABEÇALHO DA PÁGINA
# =====================================================================

st.markdown("### Inferência Bayesiana:<br>Predição de Ocorrências atendidas pela PMDF", unsafe_allow_html=True)
st.markdown("<b>Análise Preditiva com Modelos Bayesianos: 2022-2024 → 2025</b>", unsafe_allow_html=True)


# =====================================================================
# ESTRUTURA EM ABAS
# =====================================================================
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📖 Problema de Pesquisa",
    "🔧 Metodologia Bayesiana", 
    "📊 Modelos Implementados",
    "✅ Validação e Comparação",
    "🔮 Predições 2025",
    "📚 Conclusões"
])

# =====================================================================
# ABA 1: PROBLEMA DE PESQUISA
# =====================================================================

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<b>Contexto e Motivação</b>", unsafe_allow_html=True)

    st.markdown("""
    A Polícia Militar do Distrito Federal (PMDF) atende milhares de ocorrências mensalmente, variando de crimes graves a chamados de emergência diversos. O planejamento operacional eficiente depende de predições confiáveis da demanda futura.
    """)

    st.markdown("<b>Pergunta de Pesquisa</b>", unsafe_allow_html=True)

    st.markdown("""
    > **"Qual a taxa mensal esperada de ocorrências criminais atendidas pela PMDF no Distrito Federal em 2025?"**
    """)









    
    st.markdown("""
    
    ### Pergunta de Pesquisa Principal
    
    > **"Qual a taxa mensal esperada de ocorrências criminais atendidas pela PMDF no 
    > Distrito Federal em 2025, e qual a probabilidade de exceder limiares críticos 
    > de capacidade operacional?"**
    
    ### Objetivos Específicos
    
    1. **Modelar** a taxa de ocorrências usando inferência bayesiana com dados históricos
    2. **Incorporar incerteza** através de distribuições posteriori e intervalos de credibilidade
    3. **Comparar** diferentes especificações de modelos (Poisson vs Negative Binomial)
    4. **Predizer** valores mensais para 2025 com quantificação de incerteza
    5. **Avaliar riscos** de exceder limiares operacionais críticos
    
    ### Importância Prática
    
    - **Alocação de recursos:** Dimensionamento adequado de efetivo policial
    - **Planejamento orçamentário:** Estimativas confiáveis de demanda
    - **Gestão de risco:** Identificação de períodos de sobrecarga potencial
    - **Políticas públicas:** Embasamento para decisões estratégicas
    """)
    
    st.divider()
    
    # Estatísticas descritivas dos dados
    st.subheader("📊 Dados Observados (2022-2024)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats_df = pd.DataFrame({
            'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo'],
            'Valor': [
                f"{dados_originais['estatisticas_basicas']['media']:.0f}",
                f"{dados_originais['estatisticas_basicas']['mediana']:.0f}",
                f"{dados_originais['estatisticas_basicas']['std']:.0f}",
                f"{dados_originais['estatisticas_basicas']['min']:.0f}",
                f"{dados_originais['estatisticas_basicas']['max']:.0f}"
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown(f"""
        **Diagnóstico Inicial:**
        
        - **Overdispersion ratio:** {overdispersion:.1f}
        - **Interpretação:** Variância >> Média
        - **Implicação:** Modelo Poisson padrão inadequado
        - **Solução:** Negative Binomial ou modelo hierárquico
        
        ➡️ *A alta overdispersion (>1) indica necessidade de modelos que 
        capturem variabilidade extra além da distribuição Poisson.*
        """)

# =====================================================================
# ABA 2: METODOLOGIA BAYESIANA
# =====================================================================

with tab2:
    st.header("🔧 Metodologia Bayesiana Aplicada")
    
    st.markdown("""
    ## Fundamentação Teórica
    
    ### Teorema de Bayes
    
    A inferência bayesiana combina conhecimento prévio (priori) com dados observados 
    (verossimilhança) para obter conhecimento atualizado (posteriori):
    """)
    
    st.latex(r'''
    P(\\theta | y) = \\frac{P(y | \\theta) \\cdot P(\\theta)}{P(y)} \\propto P(y | \\theta) \\cdot P(\\theta)
    ''')
    
    st.markdown("""
    Onde:
    - **P(θ|y)**: Distribuição posteriori (conhecimento atualizado)
    - **P(y|θ)**: Verossimilhança (informação dos dados)
    - **P(θ)**: Distribuição priori (conhecimento prévio)
    - **P(y)**: Evidência (constante normalizadora)
    """)
    
    st.divider()
    
    # Power Prior
    st.subheader("🎯 Power Prior: Incorporando Dados Históricos")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        ### Parâmetros do Power Prior
        
        - **α (shape):** {power_prior['alpha_prior']:.1f}
        - **β (rate):** {power_prior['beta_prior']:.2f}
        - **Peso histórico:** {power_prior['power_weight']:.0%}
        
        ### Construção
        
        O Power Prior foi construído usando:
        - **Dados históricos:** {power_prior['dados_historicos_count']} meses (2022-2023)
        - **Dados atuais:** {power_prior['dados_2024_count']} meses (2024)
        - **Peso:** 70% de confiança nos dados históricos
        """)
    
    with col2:
        st.info("""
        **Por que Power Prior?**
        
        1. **Incorpora experiência prévia** de forma controlada
        2. **Reduz incerteza** nas estimativas posteriori
        3. **Flexibilidade** no peso dado ao conhecimento prévio
        4. **Justificável teoricamente** e empiricamente
        
        O peso de 70% representa um equilíbrio entre:
        - Confiança nos padrões históricos (2022-2023)
        - Adaptação a mudanças recentes (2024)
        """)
    
    st.divider()
    
    # Conjugação
    st.subheader("🔗 Conjugação Gamma-Poisson")
    
    st.markdown("""
    ### Propriedade de Conjugação
    
    A família Gamma é conjugada para a verossimilhança Poisson, permitindo cálculo 
    analítico da posteriori:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Priori:**
        """)
        st.latex(r"\\lambda \\sim \\text{Gamma}(\\alpha_0, \\beta_0)")
        
        st.markdown("""
        **Verossimilhança:**
        """)
        st.latex(r"y_i \\sim \\text{Poisson}(\\lambda)")
    
    with col2:
        st.markdown("""
        **Posteriori (conjugada):**
        """)
        st.latex(r"\\lambda | y \\sim \\text{Gamma}(\\alpha_0 + \\sum y_i, \\beta_0 + n)")
        
        st.markdown("""
        **Média Posteriori:**
        """)
        st.latex(r"E[\\lambda|y] = \\frac{\\alpha_0 + \\sum y_i}{\\beta_0 + n}")
    
    st.success("""
    **Vantagem Computacional:** A conjugação permite atualização analítica direta, 
    sem necessidade de métodos de Monte Carlo para este modelo base.
    """)

# =====================================================================
# ABA 3: MODELOS IMPLEMENTADOS
# =====================================================================

with tab3:
    st.header("📊 Modelos Bayesianos Implementados")
    
    st.markdown("""
    Três modelos foram implementados e comparados para lidar com a overdispersion 
    identificada nos dados:
    """)
    
    # Modelo A: Poisson
    with st.expander("🔴 Modelo A: Poisson com Power Prior (Baseline)", expanded=False):
        st.markdown(f"""
        ### Especificação do Modelo
        
        **Priori:**
        """)
        st.latex(r"\\lambda \\sim \\text{Gamma}(258420.0, 16.8)")
        
        st.markdown("""
        **Verossimilhança:**
        """)
        st.latex(r"y_i \\sim \\text{Poisson}(\\lambda), \\quad i = 1, ..., 36")
        
        st.markdown(f"""
        ### Resultados Posteriori
        
        - **Média:** {modelos['poisson']['parametros']['lambda_rate']['media']:.2f}
        - **Mediana:** {modelos['poisson']['parametros']['lambda_rate']['mediana']:.2f}
        - **IC 95%:** [{modelos['poisson']['parametros']['lambda_rate']['hdi_2_5']:.2f}, 
          {modelos['poisson']['parametros']['lambda_rate']['hdi_97_5']:.2f}]
        - **Desvio padrão:** {modelos['poisson']['parametros']['lambda_rate']['std']:.2f}
        
        ### Diagnósticos
        
        - **Convergência:** {modelos['poisson']['diagnosticos']['convergencia']}
        - **R̂ < 1.01:** {'✅' if modelos['poisson']['diagnosticos']['rhat_ok'] else '❌'}
        - **ESS > 400:** {'✅' if modelos['poisson']['diagnosticos']['ess_ok'] else '❌'}
        
        ### Validação
        
        - **RMSE:** {modelos['poisson']['validacao']['rmse']:.2f}
        - **Cobertura IC 95%:** {modelos['poisson']['validacao']['cobertura_ic95']:.1%}
        - **Pontos dentro IC:** {modelos['poisson']['validacao']['pontos_dentro_ic']}/{modelos['poisson']['validacao']['total_pontos']}
        """)
        
        st.error(f"""
        **Status: {modelos['poisson']['status']}**
        
        O modelo Poisson apresentou **cobertura IC 95% muito baixa** ({modelos['poisson']['validacao']['cobertura_ic95']:.1%}), 
        indicando que não captura adequadamente a variabilidade dos dados devido à overdispersion.
        """)
    
    # Modelo C: Negative Binomial
    with st.expander("🟢 Modelo C: Negative Binomial (Recomendado)", expanded=True):
        st.markdown(f"""
        ### Especificação do Modelo
        
        **Priori para média:**
        """)
        st.latex(r"\\mu \\sim \\text{Gamma}(258420.0, 16.8)")
        
        st.markdown("""
        **Priori para dispersão:**
        """)
        st.latex(r"\\alpha \\sim \\text{Exponential}(1.0)")
        
        st.markdown("""
        **Verossimilhança:**
        """)
        st.latex(r"y_i \\sim \\text{NegativeBinomial}(\\mu, \\alpha)")
        
        st.markdown(f"""
        ### Resultados Posteriori
        
        **Parâmetro μ (média):**
        - **Média:** {modelos['negative_binomial']['parametros']['mu_nb']['media']:.2f}
        - **IC 95%:** [{modelos['negative_binomial']['parametros']['mu_nb']['hdi_2_5']:.2f}, 
          {modelos['negative_binomial']['parametros']['mu_nb']['hdi_97_5']:.2f}]
        
        **Parâmetro α (dispersão):**
        - **Média:** {modelos['negative_binomial']['parametros']['alpha_nb']['media']:.2f}
        - **IC 95%:** [{modelos['negative_binomial']['parametros']['alpha_nb']['hdi_2_5']:.2f}, 
          {modelos['negative_binomial']['parametros']['alpha_nb']['hdi_97_5']:.2f}]
        
        ### Diagnósticos
        
        - **Convergência:** {modelos['negative_binomial']['diagnosticos']['convergencia']}
        - **R̂ < 1.01:** {'✅' if modelos['negative_binomial']['diagnosticos']['rhat_ok'] else '❌'}
        - **ESS > 400:** {'✅' if modelos['negative_binomial']['diagnosticos']['ess_ok'] else '❌'}
        
        ### Validação
        
        - **RMSE:** {modelos['negative_binomial']['validacao']['rmse']:.2f}
        - **Cobertura IC 95%:** {modelos['negative_binomial']['validacao']['cobertura_ic95']:.1%}
        - **Pontos dentro IC:** {modelos['negative_binomial']['validacao']['pontos_dentro_ic']}/{modelos['negative_binomial']['validacao']['total_pontos']}
        """)
        
        st.success(f"""
        **Status: {modelos['negative_binomial']['status']}**
        
        O modelo Negative Binomial apresentou **cobertura IC 95% adequada** ({modelos['negative_binomial']['validacao']['cobertura_ic95']:.1%}), 
        corrigindo completamente o problema de overdispersion através do parâmetro α.
        """)
    
    # Modelo D: Hierárquico
    with st.expander("🟡 Modelo D: Poisson Hierárquico", expanded=False):
        st.markdown(f"""
        ### Especificação do Modelo
        
        **Hyperprior global:**
        """)
        st.latex(r"\\mu_{global} \\sim \\text{Gamma}(258420.0, 16.8)")
        
        st.markdown("""
        **Hiperparâmetro de variabilidade:**
        """)
        st.latex(r"\\sigma_{temporal} \\sim \\text{HalfNormal}(0.5)")
        
        st.markdown("""
        **Efeitos aleatórios individuais:**
        """)
        st.latex(r"\\mu_i = \\mu_{global} \\cdot \\exp(\\text{offset}_i), \\quad \\text{offset}_i \\sim \\mathcal{N}(0, \\sigma_{temporal})")
        
        st.markdown(f"""
        ### Resultados Posteriori
        
        **μ_global:**
        - **Média:** {modelos['hierarquico']['parametros']['mu_global']['media']:.2f}
        - **IC 95%:** [{modelos['hierarquico']['parametros']['mu_global']['hdi_2_5']:.2f}, 
          {modelos['hierarquico']['parametros']['mu_global']['hdi_97_5']:.2f}]
        
        **σ_temporal:**
        - **Média:** {modelos['hierarquico']['parametros']['sigma_temporal']['media']:.3f}
        - **IC 95%:** [{modelos['hierarquico']['parametros']['sigma_temporal']['hdi_2_5']:.3f}, 
          {modelos['hierarquico']['parametros']['sigma_temporal']['hdi_97_5']:.3f}]
        
        ### Validação
        
        - **RMSE:** {modelos['hierarquico']['validacao']['rmse']:.2f}
        - **Cobertura IC 95%:** {modelos['hierarquico']['validacao']['cobertura_ic95']:.1%}
        - **Pontos dentro IC:** {modelos['hierarquico']['validacao']['pontos_dentro_ic']}/{modelos['hierarquico']['validacao']['total_pontos']}
        """)
        
        st.success(f"""
        **Status: {modelos['hierarquico']['status']}**
        
        O modelo hierárquico também apresentou **cobertura adequada** ({modelos['hierarquico']['validacao']['cobertura_ic95']:.1%}), 
        capturando heterogeneidade temporal através de efeitos aleatórios.
        """)

# =====================================================================
# ABA 4: VALIDAÇÃO E COMPARAÇÃO
# =====================================================================

with tab4:
    st.header("✅ Validação e Comparação de Modelos")
    
    # Tabela comparativa
    st.subheader("📊 Comparação Quantitativa")
    
    comp_df = pd.DataFrame({
        'Modelo': ['Poisson', 'Negative Binomial', 'Hierárquico'],
        'RMSE': [
            comparacao['criterios']['rmse']['poisson'],
            comparacao['criterios']['rmse']['negative_binomial'],
            comparacao['criterios']['rmse']['hierarquico']
        ],
        'Cobertura IC 95% (%)': [
            comparacao['criterios']['cobertura_ic95']['poisson'],
            comparacao['criterios']['cobertura_ic95']['negative_binomial'],
            comparacao['criterios']['cobertura_ic95']['hierarquico']
        ],
        'Overdispersion': [
            '❌ Não tratada',
            '✅ Tratada (α)',
            '✅ Tratada (hierárquico)'
        ],
        'Status': [
            modelos['poisson']['status'],
            modelos['negative_binomial']['status'],
            modelos['hierarquico']['status']
        ]
    })
    
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Visualização comparativa
    st.subheader("📈 Gráfico Comparativo de Cobertura")
    
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Bar(
        x=['Poisson', 'Negative Binomial', 'Hierárquico'],
        y=[
            comparacao['criterios']['cobertura_ic95']['poisson'],
            comparacao['criterios']['cobertura_ic95']['negative_binomial'],
            comparacao['criterios']['cobertura_ic95']['hierarquico']
        ],
        marker_color=['red', 'green', 'orange'],
        text=[f"{v:.1f}%" for v in [
            comparacao['criterios']['cobertura_ic95']['poisson'],
            comparacao['criterios']['cobertura_ic95']['negative_binomial'],
            comparacao['criterios']['cobertura_ic95']['hierarquico']
        ]],
        textposition='auto'
    ))
    
    fig_comp.add_hline(y=95, line_dash="dash", line_color="black", 
                      annotation_text="Meta (95%)")
    
    fig_comp.update_layout(
        title="Cobertura dos Intervalos de Credibilidade (95%)",
        xaxis_title="Modelo",
        yaxis_title="Cobertura (%)",
        yaxis_range=[0, 105],
        height=400
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.divider()
    
    # Análise de sensibilidade
    st.subheader("🔬 Análise de Sensibilidade às Prioris")
    
    sens_df = pd.DataFrame({
        'Priori': ['Power Prior', 'Não-Informativa', 'Vaga'],
        'Média λ': [
            sensibilidade['power_prior']['media'],
            sensibilidade['nao_informativo']['media'],
            sensibilidade['vago']['media']
        ],
        'Desvio Padrão': [
            sensibilidade['power_prior']['std'],
            sensibilidade['nao_informativo']['std'],
            sensibilidade['vago']['std']
        ],
        'Largura IC 95%': [
            sensibilidade['power_prior']['ic_width'],
            sensibilidade['nao_informativo']['ic_width'],
            sensibilidade['vago']['ic_width']
        ]
    })
    
    st.dataframe(sens_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Diferença Power vs Não-informativa",
            f"{sensibilidade['impacto_priori']['diferenca_media_power_vs_nao_inf']:.2f}",
            "ocorrências"
        )
    
    with col2:
        st.metric(
            "Redução de Incerteza",
            f"{sensibilidade['impacto_priori']['reducao_incerteza_power_vs_nao_inf']:.1f}%",
            "Power Prior vs Não-informativa"
        )
    
    st.info("""
    **Interpretação:** O Power Prior reduz a incerteza das estimativas mantendo 
    a média próxima das prioris não-informativas, demonstrando que o conhecimento 
    histórico é consistente com os dados atuais e melhora a precisão preditiva.
    """)

# =====================================================================
# ABA 5: PREDIÇÕES 2025
# =====================================================================

with tab5:
    st.header("🔮 Predições para 2025")
    
    st.markdown(f"""
    ### Modelo Selecionado: **{predicoes['modelo_usado']}**
    
    {comparacao['justificativa']}
    """)
    
    st.divider()
    
    # Predições mensais
    st.subheader("📅 Predições Mensais")
    
    predições_mensais = predicoes.get('predições_mensais', {})
    analise_risco = predicoes.get('analise_risco', {})

    pred_df = pd.DataFrame({
        'Mês': predições_mensais.get('meses', []),
        'Predição (Média)': predições_mensais.get('medias', []),
        'IC 2.5%': predições_mensais.get('ic_lower', []),
        'IC 97.5%': predições_mensais.get('ic_upper', []),
        'Prob. Exceder 15k': [f"{p:.1%}" for p in analise_risco.get('prob_mensal', [])]
    })
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    # Gráfico de predições
    fig_pred = go.Figure()
    
    # Área de incerteza
    fig_pred.add_trace(go.Scatter(
        x=list(range(12)) + list(range(12))[::-1],
        y=predicoes['predições_mensais']['ic_upper'] + predicoes['predições_mensais']['ic_lower'][::-1],
        fill='toself',
        fillcolor='rgba(0,100,200,0.2)',
        line_color='rgba(255,255,255,0)',
        name='IC 95%'
    ))
    
    # Linha de predição
    fig_pred.add_trace(go.Scatter(
        x=list(range(12)),
        y=predicoes['predições_mensais']['medias'],
        mode='lines+markers',
        name='Predição Média',
        line=dict(color='blue', width=3)
    ))
    
    # Limiar crítico
    fig_pred.add_hline(
        y=predicoes['analise_risco']['limiar_critico'], 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Limiar Crítico ({predicoes['analise_risco']['limiar_critico']:,})"
    )
    
    fig_pred.update_layout(
        title="Predições Mensais 2025 com Intervalos de Credibilidade",
        xaxis_title="Mês",
        xaxis=dict(ticktext=predicoes['predições_mensais']['meses'], tickvals=list(range(12))),
        yaxis_title="Ocorrências",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.divider()
    
    # Predição anual
    st.subheader("📊 Predição Anual 2025")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Média Anual",
            f"{predicoes['predicao_anual']['media']:.0f}",
            "ocorrências"
        )
    
    with col2:
        st.metric(
            "IC 95% Inferior",
            f"{predicoes['predicao_anual']['ic_lower']:.0f}",
            "limite inferior"
        )
    
    with col3:
        st.metric(
            "IC 95% Superior",
            f"{predicoes['predicao_anual']['ic_upper']:.0f}",
            "limite superior"
        )
    
    st.divider()
    
    # Análise de risco
    st.subheader("⚠️ Análise de Risco")
    
    st.markdown(f"""
    ### Probabilidade de Exceder Limiar Crítico ({predicoes['analise_risco']['limiar_critico']:,} ocorrências/mês)
    
    - **Probabilidade em algum mês de 2025:** {predicoes['analise_risco']['prob_algum_mes']:.1%}
    """)
    
    # Gráfico de probabilidades
    fig_risk = go.Figure()
    
    fig_risk.add_trace(go.Bar(
        x=predicoes['predições_mensais']['meses'],
        y=[p * 100 for p in predicoes['analise_risco']['prob_mensal']],
        marker_color=['red' if p > 0.5 else 'orange' if p > 0.1 else 'green' 
                      for p in predicoes['analise_risco']['prob_mensal']],
        text=[f"{p:.1%}" for p in predicoes['analise_risco']['prob_mensal']],
        textposition='auto'
    ))
    
    fig_risk.update_layout(
        title=f"Probabilidade Mensal de Exceder {predicoes['analise_risco']['limiar_critico']:,} Ocorrências",
        xaxis_title="Mês",
        yaxis_title="Probabilidade (%)",
        height=400
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Recomendações
    st.warning(f"""
    **Recomendações Operacionais:**
    
    Com {predicoes['analise_risco']['prob_algum_mes']:.1%} de probabilidade de exceder o limiar 
    crítico em algum mês de 2025, recomenda-se:
    
    1. **Planejamento preventivo** para meses de maior risco
    2. **Capacidade de resposta adicional** disponível sob demanda
    3. **Monitoramento contínuo** das ocorrências mensais
    4. **Revisão trimestral** das predições com dados atualizados
    """)

# =====================================================================
# ABA 6: CONCLUSÕES
# =====================================================================

with tab6:
    st.header("📚 Conclusões e Considerações Finais")
    
    st.markdown(f"""
    ## Síntese dos Resultados
    
    ### Problema de Pesquisa
    
    O objetivo deste trabalho foi desenvolver um modelo preditivo bayesiano para estimar 
    a taxa mensal de ocorrências atendidas pela PMDF no Distrito Federal em 2025, quantificando 
    incertezas e avaliando riscos operacionais.
    
    ### Principais Achados
    
    1. **Overdispersion Identificada**
       - Razão Variância/Média: {overdispersion:.1f}
       - Modelo Poisson padrão mostrou-se **inadequado**
       - Cobertura IC 95%: apenas {modelos['poisson']['validacao']['cobertura_ic95']:.1%}
    
    2. **Modelos Alternativos Bem-Sucedidos**
       - **Negative Binomial:** Cobertura {modelos['negative_binomial']['validacao']['cobertura_ic95']:.1%}
       - **Hierárquico:** Cobertura {modelos['hierarquico']['validacao']['cobertura_ic95']:.1%}
       - Ambos corrigiram completamente o problema de subdispersão
    
    3. **Predições para 2025**
       - Média anual: **{predicoes['predicao_anual']['media']:.0f}** ocorrências
       - IC 95%: [{predicoes['predicao_anual']['ic_lower']:.0f}, {predicoes['predicao_anual']['ic_upper']:.0f}]
       - Risco de exceder limiar crítico: **{predicoes['analise_risco']['prob_algum_mes']:.1%}**
    
    4. **Benefícios do Power Prior**
       - Redução de incerteza: {sensibilidade['impacto_priori']['reducao_incerteza_power_vs_nao_inf']:.1f}%
       - Incorporação sistemática de conhecimento histórico
       - Estimativas mais estáveis e confiáveis
    
    ### Algoritmo Implementado
    
    O modelo final ({predicoes['modelo_usado']}) utiliza:
    
    1. **Conjugação Gamma-Poisson/Negative Binomial** para atualização analítica eficiente
    2. **NUTS (No-U-Turn Sampler)** para exploração da posteriori
    3. **Múltiplas cadeias MCMC** (4 chains) para diagnóstico de convergência
    4. **Posterior Predictive Checks** para validação do ajuste
    
    ### Limitações e Trabalhos Futuros
    
    **Limitações:**
    - Dados mensais agregados (36 observações)
    - Não incorpora covariáveis explicativas (sazonalidade, políticas públicas, etc.)
    - Assume estacionariedade condicional
    
    **Extensões Sugeridas:**
    - **Modelos de séries temporais bayesianas** (BSTS, Prophet)
    - **Incorporação de covariáveis** (eventos, feriados, clima)
    - **Análise espacial** por região administrativa
    - **Atualização online** com novos dados mensais
    
    ### Contribuições Científicas
    
    1. **Metodológica:** Demonstração prática de correção de overdispersion via Negative Binomial
    2. **Aplicada:** Framework replicável para predição de demanda em segurança pública
    3. **Estatística:** Validação empírica da eficácia de Power Priors em dados reais
    
    ### Impacto Prático
    
    Os resultados deste trabalho fornecem à PMDF:
    - **Predições confiáveis** com quantificação de incerteza
    - **Análise de risco** para planejamento operacional
    - **Metodologia replicável** para atualizações futuras
    - **Base quantitativa** para decisões estratégicas
    
    ## Considerações Metodológicas
    
    A abordagem bayesiana adotada neste trabalho oferece vantagens sobre métodos frequentistas:
    
    - **Intervalos de credibilidade** mais interpretáveis que intervalos de confiança
    - **Incorporação natural** de conhecimento prévio via prioris
    - **Predições com incerteza** quantificada diretamente da posteriori
    - **Flexibilidade** para modelos hierárquicos complexos
    
    ## Referências Metodológicas
    
    - **Power Prior:** Chen & Ibrahim (2000), "The Power Prior: Theory and Applications"
    - **Negative Binomial Bayesiano:** Gelman et al. (2013), "Bayesian Data Analysis"
    - **MCMC Diagnostics:** Vehtari et al. (2021), "Rank-Normalization, Folding, and Localization"
    - **Posterior Predictive Checks:** Gelman et al. (1996), "Posterior Predictive Assessment"
    
    ---
    
    **Trabalho desenvolvido para a disciplina de Inferência Bayesiana**  
    **Dados:** PMDF - Polícia Militar do Distrito Federal  
    **Período:** 2022-2024 (36 meses de observações)  
    **Data de execução:** {info_geral['data_execucao']}
    """)
    
    st.success("""
    ✅ **Objetivo alcançado:** Modelo preditivo bayesiano robusto implementado com sucesso, 
    fornecendo predições confiáveis e análise de risco para planejamento operacional da PMDF em 2025.
    """)

# =====================================================================
# RODAPÉ
# =====================================================================

st.divider()
st.caption("Dashboard desenvolvido com Streamlit | Dados: PMDF | Modelo: Inferência Bayesiana")