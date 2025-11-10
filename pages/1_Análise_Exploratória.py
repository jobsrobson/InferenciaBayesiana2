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
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

df = load_data()
df['ano'] = df['ano'].astype(str)

df = df.sort_values(['ano', 'mes']).reset_index(drop=True)
df['tempo'] = range(len(df))
df_filtered = df.copy() 

st.set_page_config(
    page_title="Análise Bayesiana dos Dados de Criminalidade no DF",
    page_icon="👮‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cores padrão
colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}

# CSS customizado
st.markdown("""
    <style>
        /* Diminui a largura da sidebar */
        [data-testid="stSidebar"] {
            width: 300px !important;      /* largura fixa */
            min-width: 300px !important;
            max-width: 300px !important;
        }

    </style>
""", unsafe_allow_html=True)

def formatar_numero_br(valor):
    return f"{valor:,.0f}".replace(",", ".")


# =======================================================
# Filtros na sidebar

st.sidebar.markdown("### Filtros de Análise")
anos_disponiveis = df['ano'].unique().tolist()
anos_selecionados = st.sidebar.multiselect(
    "Selecione os anos para análise:",
    options=anos_disponiveis,
    default=anos_disponiveis
)

# Filtrar o dataframe baseado nos anos selecionados
if anos_selecionados:
    df_filtered = df[df['ano'].isin(anos_selecionados)].copy()
else:
    df_filtered = df.copy()  # Se nenhum ano selecionado, usar todos os dados
    st.warning("Nenhum ano selecionado. Mostrando dados de todos os anos disponíveis.", icon=":material/warning:")

# Filtrar por Tipo de Análise (Análise Exploratória, Análise de Correlações)
tipo_analise = st.sidebar.selectbox(
    "Selecione o tipo de análise:",
    options=["Análise Exploratória", "Análise de Correlações"]
)


# =======================================================
# Variáveis calculadas

# Cria novas colunas com total_furtos e total_roubos
df_filtered['total_furtos'] = total_furtos = df_filtered[['furt_trans', 'furt_cel', 'furt_veic', 'furt_com', 'furt_res']].sum(axis=1)
df_filtered['total_roubos'] = total_roubos = df_filtered[['roub_trans', 'roub_veic', 'roub_col', 'roub_res']].sum(axis=1)



# =======================================================
# CABEÇALHO DA PÁGINA
# =======================================================

st.markdown("### Análise Exploratória dos dados de Criminalidade<br> no Distrito Federal no Triênio 2022-2024", unsafe_allow_html=True)
st.markdown("<small><b>Fonte dos Dados:</b> Polícia Militar do Distrito Federal (PMDF)</small>", unsafe_allow_html=True)


if tipo_analise == "Análise Exploratória":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Análises Gerais", unsafe_allow_html=True)

    # =======================================================
    # SEÇÃO 1 - OCORRÊNCIAS GERAIS ATENDIDAS
    # =======================================================

    # MÉTRICAS
    col1, col2, col3, col4 = st.columns(4, border=True, gap="small")
    with col1:
        total_ocorrencias = df_filtered['ocor_atend'].sum() if not df_filtered.empty else 0
        st.metric("👮‍♂️ Ocorrências Atendidas", formatar_numero_br(total_ocorrencias))
    with col2:
        media_mensal = df_filtered['ocor_atend'].mean() if not df_filtered.empty else 0
        st.metric("📅 Média Mensal de Ocorrências", formatar_numero_br(media_mensal))
    with col3:
        max_ocorrencias = df_filtered['ocor_atend'].max() if not df_filtered.empty else 0
        st.metric("📈 Máximo Mensal de Ocorrências", formatar_numero_br(max_ocorrencias))
    with col4:
        min_ocorrencias = df_filtered['ocor_atend'].min() if not df_filtered.empty else 0
        st.metric("📉 Mínimo Mensal de Ocorrências", formatar_numero_br(min_ocorrencias))


    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.markdown("**Total de Ocorrências Atendidas por Ano**", unsafe_allow_html=True)

        if not df_filtered.empty:
            ocor_anual = df_filtered.groupby('ano')['ocor_atend'].sum().reset_index()
            ocor_anual['ano'] = ocor_anual['ano'].astype(int).astype(str)

            fig2 = px.bar(
                ocor_anual,
                x='ano',
                y='ocor_atend',
                labels={'ocor_atend': 'Total de Ocorrências', 'ano': 'Ano'},
                color='ano',
                text='ocor_atend',
                color_discrete_sequence=['#002156', '#ffbb3c', '#ec152f']
            )
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig2.update_xaxes(type='category')
            fig2.update_layout(
                margin=dict(t=20, b=50, l=50, r=50),
                showlegend=False,
                xaxis_title=''
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Nenhum dado disponível para os filtros selecionados.")


    with col2:
        st.markdown("**Evolução Mensal de Ocorrências Atendidas (2022-2024)**", unsafe_allow_html=True)
        # ========== 1. EVOLUÇÃO TEMPORAL DE OCORRÊNCIAS ATENDIDAS ==========
        if not df_filtered.empty:
            fig1 = px.line(df_filtered.sort_values('mes'), 
                    x='mes', y='ocor_atend', color='ano',
                    labels={'ocor_atend': 'Número de Ocorrências', 'mes': 'Mês', 'ano': 'Ano'},
                    markers=True,
                    color_discrete_map={'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
                    )
            fig1.update_layout(hovermode='x unified',margin=dict(t=0, b=0, l=0, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            # hide xaxis title
            xaxis_title=''
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Nenhum dado disponível para os filtros selecionados.")


    # =======================================================
    # SEÇÃO 2 - CRIMES VIOLENTOS CONTRA A VIDA
    # =======================================================


    # FILTRO PARA EXIBIR SOMENTE SE NÃO ESTIVER APENAS 2022
    if not (len(anos_selecionados) == 1 and '2022' in anos_selecionados):
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### Crimes Violentos contra a Vida (2023/2024)", unsafe_allow_html=True)
        st.info("A PMDF não divulgou dados do ano de 2022.", icon=":material/info:")

        # MÉTRICAS
        col1, col2, col3, col4, col5 = st.columns(5, border=True, gap="small")
        with col1:
            # Total de Homicídios (hom)
            total_homicidios = df_filtered['hom'].sum() if not df_filtered.empty else 0
            st.metric("💀 Homicídios", formatar_numero_br(total_homicidios))
        with col2:
            # Total de Tentativas de Homicídio (hom_tent)
            total_tentativas_homicidio = df_filtered['hom_tent'].sum() if not df_filtered.empty else 0
            st.metric("🤕 Tentativas de Homicídio", formatar_numero_br(total_tentativas_homicidio))
        with col3:
            # Total de Feminicídios (fem)
            total_feminicidios = df_filtered['fem'].sum() if not df_filtered.empty else 0
            st.metric("💀 Feminicídios", formatar_numero_br(total_feminicidios))
        with col4:
            # Total de Tentativas de Feminicídio (fem_tent)
            total_tentativas_feminicidio = df_filtered['fem_tent'].sum() if not df_filtered.empty else 0
            st.metric("🤕 Tentativas de Feminicídio", formatar_numero_br(total_tentativas_feminicidio))
        with col5:
            # Total de Homicídios Culposos
            total_homicidios_culposos = df_filtered['hom_culp'].sum() if not df_filtered.empty else 0
            st.metric("😭 Homicídios Culposos", formatar_numero_br(total_homicidios_culposos))

        # GRÁFICOS
        col1, col2 = st.columns(2, border=True, gap="small")
        with col1:
            st.markdown("**💀 Evolução Mensal de Homicídios (2023-2024)**", unsafe_allow_html=True)
            
            df_crimes_violentos = df_filtered[['mes', 'ano', 'hom']].copy()
            df_crimes_violentos = df_crimes_violentos.sort_values(['ano', 'mes'])

            fig3 = go.Figure()
            anos_crimes = df_crimes_violentos['ano'].unique()
            for ano in anos_crimes:
                if ano in ['2023', '2024']:  # Só mostrar anos com dados de crimes violentos
                    dados_ano = df_crimes_violentos[df_crimes_violentos['ano'] == ano]
                    colors = {'2023': '#ffbb3c', '2024': '#ec152f'}
                    fig3.add_trace(go.Scatter(
                        x=dados_ano['mes'], y=dados_ano['hom'],
                        name=f'{ano}',
                        mode='lines+markers',
                        # Cores
                        line=dict(color=colors.get(ano, '#000000'))
                    ))

            fig3.update_layout(
                yaxis_title='Número de Casos',
                hovermode='x unified',
                margin=dict(t=20, b=50, l=50, r=5),
                legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
                yaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.markdown("**💀 Evolução Mensal de Tentativas de Homicídio (2023-2024)**", unsafe_allow_html=True)
            df_crimes_violentos = df_filtered[['mes', 'ano', 'hom_tent']].copy()
            df_crimes_violentos = df_crimes_violentos.sort_values(['ano', 'mes'])

            fig4 = go.Figure()
            anos_crimes = df_crimes_violentos['ano'].unique()
            for ano in anos_crimes:
                if ano in ['2023', '2024']:  # Só mostrar anos com dados de crimes violentos
                    dados_ano = df_crimes_violentos[df_crimes_violentos['ano'] == ano]
                    colors = {'2023': '#ffbb3c', '2024': '#ec152f'}
                    fig4.add_trace(go.Scatter(
                        x=dados_ano['mes'], y=dados_ano['hom_tent'],
                        name=f'{ano}',
                        mode='lines+markers',
                        line=dict(color=colors.get(ano, '#000000'))
                    ))

            fig4.update_layout(
                yaxis_title='Número de Casos',
                hovermode='x unified',
                margin=dict(t=20, b=50, l=50, r=0),
                legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
                yaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(fig4, use_container_width=True)

        col1, col2 = st.columns(2, border=True, gap="small")
        with col1:
            st.markdown("**💀 Evolução Mensal de Feminicídios (2023-2024)**", unsafe_allow_html=True)
            df_crimes_violentos = df_filtered[['mes', 'ano', 'fem']].copy()
            df_crimes_violentos = df_crimes_violentos.sort_values(['ano', 'mes'])
            
            fig4 = go.Figure()
            anos_crimes = df_crimes_violentos['ano'].unique()
            for ano in anos_crimes:
                if ano in ['2023', '2024']:  # Só mostrar anos com dados de crimes violentos
                    dados_ano = df_crimes_violentos[df_crimes_violentos['ano'] == ano]
                    colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
                    fig4.add_trace(go.Scatter(
                        x=dados_ano['mes'], y=dados_ano['fem'],
                        name=f'{ano}',
                        mode='lines+markers',
                        line=dict(color=colors.get(ano, '#000000'))
                    ))

            fig4.update_layout(
                yaxis_title='Número de Casos',
                hovermode='x unified',
                margin=dict(t=20, b=50, l=50, r=0),
                legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
                yaxis=dict(range=[0, 20])
            )
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            st.markdown("**💀 Evolução Mensal de Tentativas de Feminicídio (2023-2024)**", unsafe_allow_html=True)
            df_crimes_violentos = df_filtered[['mes', 'ano', 'fem_tent']].copy()
            df_crimes_violentos = df_crimes_violentos.sort_values(['ano', 'mes'])

            fig5 = go.Figure()
            colors = {'2023': 'pink', '2024': 'red'}
            anos_crimes = df_crimes_violentos['ano'].unique()
            for ano in anos_crimes:
                if ano in ['2023', '2024']:  # Só mostrar anos com dados de crimes violentos
                    dados_ano = df_crimes_violentos[df_crimes_violentos['ano'] == ano]
                    colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
                    fig5.add_trace(go.Scatter(
                        x=dados_ano['mes'], y=dados_ano['fem_tent'],
                        name=f'{ano}',
                        mode='lines+markers',
                        line=dict(color=colors.get(ano, '#000000'))
                    ))

            fig5.update_layout(
                yaxis_title='Número de Casos',
                hovermode='x unified',
                margin=dict(t=20, b=50, l=50, r=0),
                legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
                yaxis=dict(range=[0, 20])
            )
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("#### Crimes Violentos contra a Vida", unsafe_allow_html=True)
        st.warning("Dados de Crimes Violentos contra a Vida não estão disponíveis para o ano de 2022.", icon=":material/warning:")



    # =======================================================
    # SEÇÃO 3 - ACIDENTES DE TRÂNSITO 
    # =======================================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Acidentes de Trânsito", unsafe_allow_html=True)
    st.info("Apenas acidentes de trânsito atendidos pela PMDF no triênio.", icon=":material/info:")

    # MÉTRICAS
    col1, col2, col3, col4 = st.columns(4, border=True, gap="small")
    with col1:
        # Total 
        total_acidentes = df_filtered['acid_tran_cvit'].sum() + df_filtered['acid_tran_svit'].sum() + df_filtered['acid_tran_vit_fat'].sum() if not df_filtered.empty else 0
        st.metric("💥 Acidentes de Trânsito", formatar_numero_br(total_acidentes))
    with col2:
        # Total com vítima
        total_acidentes_cvit = df_filtered['acid_tran_cvit'].sum() if not df_filtered.empty else 0
        st.metric("🤕 Acidentes com Vítima", formatar_numero_br(total_acidentes_cvit))
    with col3:
        # Total sem vítima
        total_acidentes_svit = df_filtered['acid_tran_svit'].sum() if not df_filtered.empty else 0
        st.metric("🥴 Acidentes sem Vítima", formatar_numero_br(total_acidentes_svit))
    with col4:
        # Total de vítimas fatais
        total_acidentes_vit_fat = df_filtered['acid_tran_vit_fat'].sum() if not df_filtered.empty else 0
        st.metric("💀 Vítimas Fatais", formatar_numero_br(total_acidentes_vit_fat))

    # GRÁFICOS
    with st.container(border=True):
        st.markdown("**💥 Total de Acidentes de Trânsito atendidos pela PMDF**", unsafe_allow_html=True)
        # Gráfico com a soma de acid_tran_cvit, acid_tran_svit e acid_tran_vit_fat por ano 
        df_acidentes_total = df_filtered[['ano', 'acid_tran_cvit', 'acid_tran_svit', 'acid_tran_vit_fat']].copy()
        df_acidentes_total = df_acidentes_total.groupby('ano').sum().reset_index()
        # Renomear colunas para "Com Vítima", "Sem Vítima" e "Com Vítima Fatal"
        df_acidentes_total = df_acidentes_total.rename(columns={
            'acid_tran_cvit': 'Com Vítima',
            'acid_tran_svit': 'Sem Vítima',
            'acid_tran_vit_fat': 'Vítimas Fatais'
        })
        # Transformar o dataframe para formato longo
        df_acidentes_total_long = df_acidentes_total.melt(
            id_vars='ano',
            value_vars=['Com Vítima', 'Sem Vítima', 'Vítimas Fatais'],
            var_name='Tipo de Acidente',
            value_name='Número de Acidentes'
        )
        fig_acidentes_total = px.bar(
            df_acidentes_total_long,
            x='ano',
            y='Número de Acidentes',
            color='Tipo de Acidente',
            barmode='group',
            labels={
                'ano': 'Ano',
                'Número de Acidentes': 'Número de Acidentes',
                'Tipo de Acidente': 'Tipo de Acidente'
            },
            color_discrete_sequence=['#002156', '#ffbb3c', '#ec152f']

        )
        fig_acidentes_total.update_layout(margin=dict(t=20, b=00, l=50, r=50))
        # Mostra valores acima das barras
        fig_acidentes_total.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
        fig_acidentes_total.update_layout(
            margin=dict(t=20, b=0, l=50, r=50),
            bargap=0.25,         # Espaço entre grupos de barras (0 = coladas, 1 = muito separadas)
            bargroupgap=0.10,    # Espaço entre barras do mesmo grupo
            xaxis_title=''
        )
        st.plotly_chart(fig_acidentes_total, use_container_width=True)
        

    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.markdown("**🥴 Evolução Mensal de Acidentes de Trânsito Sem Vítima**", unsafe_allow_html=True)
        df_acidentes = df_filtered[['mes', 'ano', 'acid_tran_svit']].copy()
        df_acidentes = df_acidentes.sort_values(['ano', 'mes'])

        fig6 = go.Figure()
        anos_acidentes = df_acidentes['ano'].unique()
        for ano in anos_acidentes:
            dados_ano = df_acidentes[df_acidentes['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig6.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano['acid_tran_svit'],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig6.update_layout(
            yaxis_title='Número de Acidentes',
            hovermode='x unified',
            margin=dict(t=20, b=50, l=50, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            yaxis=dict(range=[0, 300]),
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        st.markdown("**🤕 Evolução Mensal de Acidentes de Trânsito Com Vítima**", unsafe_allow_html=True)
        df_acidentes = df_filtered[['mes', 'ano', 'acid_tran_cvit']].copy()
        df_acidentes = df_acidentes.sort_values(['ano', 'mes'])

        fig7 = go.Figure()
        anos_acidentes = df_acidentes['ano'].unique()
        for ano in anos_acidentes:
            dados_ano = df_acidentes[df_acidentes['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig7.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano['acid_tran_cvit'],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig7.update_layout(
            yaxis_title='Número de Acidentes',
            hovermode='x unified',
            margin=dict(t=20, b=50, l=50, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            yaxis=dict(range=[0, 300]),
        )
        st.plotly_chart(fig7, use_container_width=True)


    # =======================================================
    # SEÇÃO 4 - CRIMES MARIA DA PENHA E VIAS DE FATO
    # =======================================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Crimes de Violência Doméstica", unsafe_allow_html=True)
    st.info("Análise dos crimes de Violência Doméstica registrados pela PMDF no triênio.", icon=":material/info:")

    # MÉTRICAS
    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        # Total mar_penha
        total_mar_penha = df_filtered['mar_penha'].sum() if not df_filtered.empty else 0
        st.metric("🤦‍♀️ Ocorrências por Violência Doméstica - Maria da Penha", formatar_numero_br(total_mar_penha))
    with col2:
        # Total vias_fato
        total_vias_fato = df_filtered['vias_fato'].sum() if not df_filtered.empty else 0
        st.metric("👊 Ocorrências por Vias de Fato", formatar_numero_br(total_vias_fato))

    # GRÁFICOS
    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.markdown("**👩‍⚖️ Evolução Mensal - Maria da Penha**", unsafe_allow_html=True)
        df_maria_da_penha = df_filtered[['mes', 'ano', 'mar_penha']].copy()
        df_maria_da_penha = df_maria_da_penha.sort_values(['ano', 'mes'])

        fig8 = go.Figure()
        anos_maria = df_maria_da_penha['ano'].unique()
        for ano in anos_maria:
            dados_ano = df_maria_da_penha[df_maria_da_penha['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig8.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano['mar_penha'],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig8.update_layout(
            yaxis_title='Número de Casos',
            hovermode='x unified',
            margin=dict(t=20, b=50, l=50, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            # INSERE LINHA PONTILHADA NA MÉDIA
            shapes=[
                dict(
                    type='line',
                    x0=dados_ano['mes'].min(), y0=dados_ano['mar_penha'].mean(),
                    x1=dados_ano['mes'].max(), y1=dados_ano['mar_penha'].mean(),
                    line=dict(color='gray', width=1, dash='dash')
                ) for ano in anos_maria
            ]
        )
        st.plotly_chart(fig8, use_container_width=True)

    with col2:
        st.markdown("**👊 Evolução Mensal - Vias de Fato**", unsafe_allow_html=True)
        df_vias_de_fato = df_filtered[['mes', 'ano', 'vias_fato']].copy()
        df_vias_de_fato = df_vias_de_fato.sort_values(['ano', 'mes'])

        fig9 = go.Figure()
        anos_vias = df_vias_de_fato['ano'].unique()
        for ano in anos_vias:
            dados_ano = df_vias_de_fato[df_vias_de_fato['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig9.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano['vias_fato'],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig9.update_layout(
            yaxis_title='Número de Casos',
            hovermode='x unified',
            margin=dict(t=20, b=50, l=50, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            # INSERE LINHA PONTILHADA NA MÉDIA
            shapes=[
                dict(
                    type='line',
                    x0=dados_ano['mes'].min(), y0=dados_ano['vias_fato'].mean(),
                    x1=dados_ano['mes'].max(), y1=dados_ano['vias_fato'].mean(),
                    line=dict(color='gray', width=1, dash='dash')
                ) for ano in anos_vias
            ]
        )
        st.plotly_chart(fig9, use_container_width=True)



    # =======================================================
    # SEÇÃO 5 - CRIMES PATRIMONIAIS
    # =======================================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Crimes Patrimoniais", unsafe_allow_html=True)
    st.info("Análise dos crimes patrimoniais registrados pela PMDF no triênio.", icon=":material/info:")

    # MÉTRICAS
    col1, col2, col3, col4 = st.columns(4, border=True, gap="small")
    with col1:
        # Total de furtos
        total_furtos = df_filtered[['furt_trans', 'furt_cel', 'furt_veic', 'furt_com', 'furt_res']].sum().sum() if not df_filtered.empty else 0
        st.metric("🏃 Total de Furtos", formatar_numero_br(total_furtos))
    with col2:
        # Média mensal de furtos
        media_mensal_furtos = total_furtos / (len(df_filtered['ano'].unique()) * 12) if not df_filtered.empty else 0
        st.metric("📅 Média Mensal de Furtos", formatar_numero_br(media_mensal_furtos))
    with col3:
        # Total de Roubos
        total_roubos = df_filtered[['roub_trans', 'roub_veic', 'roub_col', 'roub_res']].sum().sum() if not df_filtered.empty else 0
        st.metric("🔫 Total de Roubos", formatar_numero_br(total_roubos))
    with col4:
        # Média mensal de roubos
        media_mensal_roubos = total_roubos / (len(df_filtered['ano'].unique()) * 12) if not df_filtered.empty else 0
        st.metric("📅 Média Mensal de Roubos", formatar_numero_br(media_mensal_roubos))

    # GRÁFICOS
    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.markdown("**💸 Evolução Mensal de Furtos**", unsafe_allow_html=True)
        # SOMA FURTOS (furt_trans + furt_cel + furt_veic + furt_com + furt_res)
        df_furtos = df_filtered[['mes', 'ano']].copy()
        df_furtos['total_furtos'] = df_filtered[['furt_trans', 'furt_cel', 'furt_veic', 'furt_com', 'furt_res']].sum(axis=1)
        df_furtos = df_furtos.sort_values(['ano', 'mes'])
        fig10 = go.Figure()
        anos_furtos = df_furtos['ano'].unique()
        for ano in anos_furtos:
            dados_ano = df_furtos[df_furtos['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig10.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano['total_furtos'],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig10.update_layout(
            yaxis_title='Número de Casos',
            hovermode='x unified',
            margin=dict(t=20, b=50, l=50, r=00),
            yaxis=dict(range=[0, 600]),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            shapes=[
                dict(
                    type='line',
                    x0=dados_ano['mes'].min(), y0=dados_ano['total_furtos'].mean(),
                    x1=dados_ano['mes'].max(), y1=dados_ano['total_furtos'].mean(),
                    line=dict(color='gray', width=1, dash='dash')
                ) for ano in anos_furtos
            ]
        )
        st.plotly_chart(fig10, use_container_width=True)

    with col2:
        st.markdown("**🔫 Evolução Mensal de Roubos**", unsafe_allow_html=True)
        # total_roubos
        df_roubos = df_filtered[['mes', 'ano']].copy()
        df_roubos['total_roubos'] = df_filtered[['roub_trans', 'roub_veic', 'roub_col', 'roub_res']].sum(axis=1)
        df_roubos = df_roubos.sort_values(['ano', 'mes'])
        fig11 = go.Figure()
        anos_roubos = df_roubos['ano'].unique()
        for ano in anos_roubos:
            dados_ano = df_roubos[df_roubos['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig11.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano['total_roubos'],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig11.update_layout(
            yaxis_title='Número de Casos',
            hovermode='x unified',
            margin=dict(t=20, b=50, l=50, r=00),
            yaxis=dict(range=[0, 600]),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            shapes=[
                dict(
                    type='line',
                    x0=dados_ano['mes'].min(), y0=dados_ano['total_roubos'].mean(),
                    x1=dados_ano['mes'].max(), y1=dados_ano['total_roubos'].mean(),
                    line=dict(color='gray', width=1, dash='dash')
                ) for ano in anos_roubos
            ],
            
        )
        st.plotly_chart(fig11, use_container_width=True)

    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:

        # Evolução anual de Furtos
        st.markdown("**🏃 Total Anual de Furtos**", unsafe_allow_html=True)
        df_furtos_anual = df_filtered[['ano', 'furt_trans', 'furt_cel', 'furt_veic', 'furt_com', 'furt_res']].copy()
        df_furtos_anual = df_furtos_anual.groupby('ano').sum().reset_index()
        df_furtos_anual['total_furtos'] = df_furtos_anual[['furt_trans', 'furt_cel', 'furt_veic', 'furt_com', 'furt_res']].sum(axis=1)

        # 🔧 Converter 'ano' para string
        df_furtos_anual['ano'] = df_furtos_anual['ano'].astype(str)

        fig12 = px.bar(
            df_furtos_anual,
            x='ano',
            y='total_furtos',
            labels={'total_furtos': 'Número de Furtos', 'ano': 'Ano'},
            color='ano',
            text='total_furtos',
            color_discrete_sequence=['#002156', '#ffbb3c', '#ec152f']
        )

        fig12.update_traces(texttemplate='%{text:,.0f}', textposition='outside')

        # 🔧 Forçar o eixo X a ser categórico
        fig12.update_xaxes(type='category')

        fig12.update_layout(
            margin=dict(t=20, b=50, l=50, r=50),
            showlegend=False,
            xaxis_title=''
        )

        st.plotly_chart(fig12, use_container_width=True)

    with col2:
        # Evolução anual de Roubos
        st.markdown("**🔫 Total Anual de Roubos**", unsafe_allow_html=True)
        df_roubos_anual = df_filtered[['ano', 'roub_trans', 'roub_veic', 'roub_col', 'roub_res']].copy()
        df_roubos_anual = df_roubos_anual.groupby('ano').sum().reset_index()
        df_roubos_anual['total_roubos'] = df_roubos_anual[['roub_trans', 'roub_veic', 'roub_col', 'roub_res']].sum(axis=1)
        # 🔧 Converter 'ano' para string
        df_roubos_anual['ano'] = df_roubos_anual['ano'].astype(str)

        fig13 = px.bar(
            df_roubos_anual,
            x='ano',
            y='total_roubos',
            labels={'total_roubos': 'Número de Roubos', 'ano': 'Ano'},
            color='ano',
            text='total_roubos',
            color_discrete_sequence=['#002156', '#ffbb3c', '#ec152f']
        )
        fig13.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        # 🔧 Forçar o eixo X a ser categórico
        fig13.update_xaxes(type='category')

        fig13.update_layout(
            margin=dict(t=20, b=50, l=50, r=50),
            showlegend=False,
            xaxis_title=''
        )
        st.plotly_chart(fig13, use_container_width=True)


    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("##### Detalhamento dos Crimes Patrimoniais - Furtos", unsafe_allow_html=True)

    # MÉTRICAS
    col1, col2, col3, col4, col5 = st.columns(5, border=True, gap="small")
    with col1:
        # Total de furtos a transeuntes
        total_furtos = df_filtered['furt_trans'].sum() if not df_filtered.empty else 0
        st.metric("🏃 Furtos a Transeuntes", formatar_numero_br(total_furtos))
    with col2:
        # Total de furtos de celulares
        total_furtos = df_filtered['furt_cel'].sum() if not df_filtered.empty else 0
        st.metric("🏃📱 Furtos de Celulares", formatar_numero_br(total_furtos))
    with col3:
        # Total de furtos de veículos
        total_furtos = df_filtered['furt_veic'].sum() if not df_filtered.empty else 0
        st.metric("🏃🚗 Furtos de Veículos", formatar_numero_br(total_furtos))
    with col4:
        # Total de furtos a comércios
        total_furtos = df_filtered['furt_com'].sum() if not df_filtered.empty else 0
        st.metric("🏃🏪 Furtos a Comércios", formatar_numero_br(total_furtos))
    with col5:
        # Total de furtos a residências
        total_furtos = df_filtered['furt_res'].sum() if not df_filtered.empty else 0
        st.metric("🏃🏠 Furtos a Residências", formatar_numero_br(total_furtos))

    # GRÁFICOS
    variaveis = {
        "Furtos a Transeuntes": "furt_trans",
        "Furtos de Celulares": "furt_cel",
        "Furtos de Veículos": "furt_veic",
        "Furtos a Comércios": "furt_com",
        "Furtos a Residências": "furt_res"
    }

    # Dropdown para escolher variável

    with st.container(border=True):
        tipo_var = st.selectbox("Selecione o tipo de variável:", list(variaveis.keys()))
        var_col = variaveis[tipo_var]

    col_esq, col_dir = st.columns(2, border=True, gap="small")

    # --- Evolução mensal ---
    with col_esq:
        st.markdown(f"**📅 Evolução Mensal de {tipo_var}**", unsafe_allow_html=True)
        df_mensal = df_filtered.groupby(['ano', 'mes'])[var_col].sum().reset_index()
        df_mensal = df_mensal.sort_values(['ano', 'mes'])
        df_mensal['mes'] = df_mensal['mes'].astype(str)

        fig_mensal = go.Figure()
        anos_mensal = df_mensal['ano'].unique()
        for ano in anos_mensal: 
            dados_ano = df_mensal[df_mensal['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig_mensal.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano[var_col],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig_mensal.update_layout(
            yaxis_title='Número de Casos',
            hovermode='x unified',
            margin=dict(t=30, b=50, l=50, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            )
        )
        st.plotly_chart(fig_mensal, use_container_width=True)

    # --- Evolução anual ---
    with col_dir:
        st.markdown(f"**📊 Total Anual de {tipo_var}**", unsafe_allow_html=True)
        df_anual = df_filtered.groupby('ano')[var_col].sum().reset_index()
        df_anual['ano'] = df_anual['ano'].astype(int).astype(str)

        fig_anual = px.bar(
            df_anual,
            x='ano',
            y=var_col,
            labels={var_col: 'Número de Casos', 'ano': 'Ano'},
            color='ano',
            text=var_col,
            color_discrete_sequence=['#002156', '#ffbb3c', '#ec152f']
        )
        fig_anual.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_anual.update_xaxes(type='category')
        fig_anual.update_layout(
            margin=dict(t=30, b=50, l=50, r=50),
            showlegend=False,
            xaxis_title=''
        )
        st.plotly_chart(fig_anual, use_container_width=True)


    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("##### Detalhamento dos Crimes Patrimoniais - Roubos", unsafe_allow_html=True)

    # MÉTRICAS
    col1, col2, col3, col4 = st.columns(4, border=True, gap="small")
    with col1:
        # Total de roubos a transeuntes
        total_roubos = df_filtered['roub_trans'].sum() if not df_filtered.empty else 0
        st.metric("🔫 Roubos a Transeuntes", formatar_numero_br(total_roubos))
    with col2:
        # Total de roubos de veículos
        total_roubos = df_filtered['roub_veic'].sum() if not df_filtered.empty else 0
        st.metric("🔫🚗 Roubos de Veículos", formatar_numero_br(total_roubos))
    with col3:
        # Total de roubos a coletivos
        total_roubos = df_filtered['roub_col'].sum() if not df_filtered.empty else 0
        st.metric("🔫🚍 Roubos a Coletivos", formatar_numero_br(total_roubos))
    with col4:
        # Total de roubos a residências
        total_roubos = df_filtered['roub_res'].sum() if not df_filtered.empty else 0
        st.metric("🔫🏠 Roubos a Residências", formatar_numero_br(total_roubos))

    # GRÁFICOS
    variaveis = {
        "Roubos a Transeuntes": "roub_trans",
        "Roubos de Veículos": "roub_veic",
        "Roubos a Coletivos": "roub_col",
        "Roubos a Residências": "roub_res"
    }

    # Dropdown para escolher variável
    with st.container(border=True):
        tipo_var = st.selectbox("Selecione o tipo de variável:", list(variaveis.keys()))
        var_col = variaveis[tipo_var]

    col_esq, col_dir = st.columns(2, border=True, gap="small")

    # --- Evolução mensal ---
    with col_esq:
        st.markdown(f"**📅 Evolução Mensal de {tipo_var}**", unsafe_allow_html=True)
        df_mensal = df_filtered.groupby(['ano', 'mes'])[var_col].sum().reset_index()
        df_mensal = df_mensal.sort_values(['ano', 'mes'])
        df_mensal['mes'] = df_mensal['mes'].astype(str)

        fig_mensal = go.Figure()
        anos_mensal = df_mensal['ano'].unique()
        for ano in anos_mensal: 
            dados_ano = df_mensal[df_mensal['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig_mensal.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano[var_col],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig_mensal.update_layout(
            yaxis_title='Número de Casos',
            hovermode='x unified',
            margin=dict(t=30, b=50, l=50, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            )
        )
        st.plotly_chart(fig_mensal, use_container_width=True)

    # --- Evolução anual ---
    with col_dir:
        st.markdown(f"**📊 Total Anual de {tipo_var}**", unsafe_allow_html=True)
        df_anual = df_filtered.groupby('ano')[var_col].sum().reset_index()
        df_anual['ano'] = df_anual['ano'].astype(int).astype(str)

        fig_anual = px.bar(
            df_anual,
            x='ano',
            y=var_col,
            labels={var_col: 'Número de Casos', 'ano': 'Ano'},
            color='ano',
            text=var_col,
            color_discrete_sequence=['#002156', '#ffbb3c', '#ec152f']
        )
        fig_anual.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_anual.update_xaxes(type='category')
        fig_anual.update_layout(
            margin=dict(t=30, b=50, l=50, r=50),
            showlegend=False,
            xaxis_title=''
        )
        st.plotly_chart(fig_anual, use_container_width=True)



    # =======================================================
    # SEÇÃO 6 - APREENSÕES DE ARMAS E DROGAS
    # =======================================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Apreensões de Armas e Drogas", unsafe_allow_html=True)
    st.info("Análise das apreensões de armas e drogas realizadas pela PMDF no triênio.", icon=":material/info:")

    # MÉTRICAS
    col1, col2, col3, col4 = st.columns(4, border=True, gap="small")
    with col1:
        # Total de apreensões de armas
        total_apreensoes_armas = df_filtered[['arm_fogo_apre']].sum().sum() if not df_filtered.empty else 0
        st.metric("🔫 Apreensões de Armas de Fogo", formatar_numero_br(total_apreensoes_armas))
    with col2:
        # Total apreensões de armas brancas
        total_apreensoes_armas_brancas = df_filtered[['arm_branc_apr']].sum().sum() if not df_filtered.empty else 0
        st.metric("🗡 Apreensões de Armas Brancas", formatar_numero_br(total_apreensoes_armas_brancas))
    with col3:
        # Total de apreensões de drogas kg
        total_apreensoes_drogas_kg = df_filtered[['drog_kg_apr']].sum().sum() if not df_filtered.empty else 0
        st.metric("💊 Apreensões de Drogas (kg)", formatar_numero_br(total_apreensoes_drogas_kg))
    with col4:
        # Total de apreensões de drogas unidades
        total_apreensoes_drogas_unidades = df_filtered[['drog_un_apr']].sum().sum() if not df_filtered.empty else 0
        st.metric("💊 Apreensões de Drogas (unidades)", formatar_numero_br(total_apreensoes_drogas_unidades))

    # GRÁFICOS
    variaveis = {
        "Armas de Fogo": "arm_fogo_apre",
        "Armas Brancas": "arm_branc_apr",
        "Drogas (kg)": "drog_kg_apr",
        "Drogas (unidades)": "drog_un_apr"
    }

    # Dropdown para escolher variável
    with st.container(border=True):
        tipo_var = st.selectbox("Selecione o tipo de variável:", list(variaveis.keys()))
        var_col = variaveis[tipo_var]

    col_esq, col_dir = st.columns(2, border=True, gap="small")

    # --- Evolução mensal ---
    with col_esq:
        st.markdown(f"**📅 Evolução Mensal de Apreensões de {tipo_var}**", unsafe_allow_html=True)
        df_mensal = df_filtered.groupby(['ano', 'mes'])[var_col].sum().reset_index()
        df_mensal = df_mensal.sort_values(['ano', 'mes'])
        df_mensal['mes'] = df_mensal['mes'].astype(str)

        fig_mensal = go.Figure()
        anos_mensal = df_mensal['ano'].unique()
        for ano in anos_mensal: 
            dados_ano = df_mensal[df_mensal['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig_mensal.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano[var_col],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig_mensal.update_layout(
            yaxis_title='Número de Apreensões',
            hovermode='x unified',
            margin=dict(t=30, b=50, l=50, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            )
        )
        st.plotly_chart(fig_mensal, use_container_width=True)

    # --- Evolução anual ---
    with col_dir:
        st.markdown(f"**📊 Total Anual de Apreensões de {tipo_var}**", unsafe_allow_html=True)
        df_anual = df_filtered.groupby('ano')[var_col].sum().reset_index()
        df_anual['ano'] = df_anual['ano'].astype(int).astype(str)

        fig_anual = px.bar(
            df_anual,
            x='ano',
            y=var_col,
            labels={var_col: 'Número de Apreensões', 'ano': 'Ano'},
            color='ano',
            text=var_col,
            color_discrete_sequence=['#002156', '#ffbb3c', '#ec152f']
        )
        fig_anual.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_anual.update_xaxes(type='category')
        fig_anual.update_layout(
            margin=dict(t=30, b=50, l=50, r=50),
            showlegend=False,
            xaxis_title=''
        )
        st.plotly_chart(fig_anual, use_container_width=True)




    # =======================================================
    # SEÇÃO 7 - FLAGRANTES, PRISÕES E TCOs
    # =======================================================

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Flagrantes, Prisões e TCOs", unsafe_allow_html=True)
    st.info("Análise dos flagrantes, prisões e TCOs (Termos Circunstanciados de Ocorrência) realizados pela PMDF no triênio.", icon=":material/info:")

    # MÉTRICAS
    col1, col2, col3, col4, col5 = st.columns(5, border=True, gap="small")
    with col1:
        # Total de Flagrantes
        total_flagrantes = df_filtered[['flagrantes']].sum().sum() if not df_filtered.empty else 0
        st.metric("🚨 Total de Flagrantes", formatar_numero_br(total_flagrantes))
    with col2:
        # Total de Prisões em Flagrante
        total_prisoes = df_filtered[['mai_presos_flag']].sum().sum() if not df_filtered.empty else 0
        st.metric("👮 Total de Prisões em Flagrante", formatar_numero_br(total_prisoes))
    with col3:
        # Total de Detenções de Maiores de Idade
        total_apreensoes_maiores = df_filtered[['mai_detidos']].sum().sum() if not df_filtered.empty else 0
        st.metric("⛓️ Detenções de Maiores de Idade", formatar_numero_br(total_apreensoes_maiores))
    with col4:
        # Total de Menores Apreendidos
        total_apreensoes_menores = df_filtered[['men_apre']].sum().sum() if not df_filtered.empty else 0
        st.metric("👦 Total de Menores Apreendidos", formatar_numero_br(total_apreensoes_menores))
    with col5:
        # Total de TCOs Lavrados
        tcos = df_filtered[['tco_pmdf', 'tco_outros']].sum().sum()
        total_tcos = tcos if not df_filtered.empty else 0
        st.metric("📝 Total de TCOs Lavrados", formatar_numero_br(total_tcos))


    # GRÁFICOS
    variaveis = {
        "Flagrantes": "flagrantes",
        "Prisões em Flagrante": "mai_presos_flag",
        "Detenções de Maiores de Idade": "mai_detidos",
        "Menores Apreendidos": "men_apre",
        "TCOs Lavrados": "tcos"
    }

    # Dropdown para escolher variável
    with st.container(border=True):
        tipo_var = st.selectbox("Selecione o tipo de variável:", list(variaveis.keys()))
        var_col = variaveis[tipo_var]

    col_esq, col_dir = st.columns(2, border=True, gap="small")
    # --- Evolução mensal ---
    with col_esq:
        st.markdown(f"**📅 Evolução Mensal de {tipo_var}**", unsafe_allow_html=True)
        if var_col == 'tcos':
            df_mensal = df_filtered.groupby(['ano', 'mes'])[['tco_pmdf', 'tco_outros']].sum().reset_index()
            df_mensal['tcos'] = df_mensal['tco_pmdf'] + df_mensal['tco_outros']
        else:
            df_mensal = df_filtered.groupby(['ano', 'mes'])[var_col].sum().reset_index()
        df_mensal = df_mensal.sort_values(['ano', 'mes'])
        df_mensal['mes'] = df_mensal['mes'].astype(str)

        fig7 = go.Figure()
        anos_mensal = df_mensal['ano'].unique()
        for ano in anos_mensal: 
            dados_ano = df_mensal[df_mensal['ano'] == ano]
            colors = {'2022': '#002156', '2023': '#ffbb3c', '2024': '#ec152f'}
            fig7.add_trace(go.Scatter(
                x=dados_ano['mes'], y=dados_ano[var_col] if var_col != 'tcos' else dados_ano['tcos'],
                name=f'{ano}',
                mode='lines+markers',
                line=dict(color=colors.get(ano, '#000000'))
            ))
        fig7.update_layout(
            yaxis_title='Número de Casos',
            hovermode='x unified',
            margin=dict(t=30, b=50, l=50, r=0),
            legend=dict(
                x=1,  # Posição horizontal da legenda (0 = esquerda, 1 = direita)
                y=1,  # Posição vertical da legenda (0 = inferior, 1 = superior)
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='lightgray',
                borderwidth=1
            )
        )
        st.plotly_chart(fig7, use_container_width=True)

    # --- Evolução anual ---
    with col_dir:   
        st.markdown(f"**📊 Total Anual de {tipo_var}**", unsafe_allow_html=True)
        if var_col == 'tcos':
            df_anual = df_filtered.groupby('ano')[['tco_pmdf', 'tco_outros']].sum().reset_index()
            df_anual['tcos'] = df_anual['tco_pmdf'] + df_anual['tco_outros']
        else:
            df_anual = df_filtered.groupby('ano')[var_col].sum().reset_index()
        df_anual['ano'] = df_anual['ano'].astype(int).astype(str)

        fig_anual = px.bar(
            df_anual,
            x='ano',
            y=var_col if var_col != 'tcos' else 'tcos',
            labels={var_col: 'Número de Casos', 'ano': 'Ano'} if var_col != 'tcos' else {'tcos': 'Número de Casos', 'ano': 'Ano'},
            color='ano',
            text=var_col if var_col != 'tcos' else 'tcos',
            color_discrete_sequence=['#002156', '#ffbb3c', '#ec152f']
        )
        fig_anual.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_anual.update_xaxes(type='category')
        fig_anual.update_layout(
            margin=dict(t=30, b=50, l=50, r=50),
            showlegend=False,
            xaxis_title=''
        )
        st.plotly_chart(fig_anual, use_container_width=True)









elif tipo_analise == "Análise de Correlações":
    st.markdown("#### <br>Análise de Correlações entre Variáveis", unsafe_allow_html=True)

    # Criar variável temporal no df_filtered
    df['ano'] = df['ano'].astype(str)
    
    # Criar variáveis agregadas
    df['total_furtos'] = df[['furt_trans', 'furt_cel', 'furt_veic', 
                              'furt_com', 'furt_res']].sum(axis=1)
    df['total_roubos'] = df[['roub_trans', 'roub_veic', 
                              'roub_col', 'roub_res']].sum(axis=1)
    
    # Ordenar meses
    meses_ordem = ['JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO', 
                   'JULHO', 'AGOSTO', 'SETEMBRO', 'OUTUBRO', 'NOVEMBRO', 'DEZEMBRO']
    df['mes'] = pd.Categorical(df['mes'], categories=meses_ordem, ordered=True)
    
    # Criar variável temporal
    df_sorted = df.sort_values(['ano', 'mes']).reset_index(drop=True)
    df_sorted['tempo'] = range(len(df_sorted))

    # Criar abas
    tab1, tab2, tab3 = st.tabs([
        "🔗 Correlações",
        "🛡️ Apreensões",
        "📅 Sazonalidade"
    ])


    with tab1:
        # ========== 1. MATRIZ DE CORRELAÇÃO - CRIMES VIOLENTOS ==========

        # Mapeamento das variáveis de crimes violentos - Nomes amigáveis
        crimes_violentos = {
            'hom': 'Homicídios',
            'hom_tent': 'Tent. Homicídio',
            'fem': 'Feminicídios',
            'fem_tent': 'Tent. Feminicídio',
            'hom_culp': 'Hom. Culposos',
            'infant': 'Infanticídios',
            'vias_fato': 'Vias de Fato',
            'mar_penha': 'Viol. Doméstica'
        }

        st.markdown("##### <br>Matriz de Correlação - Crimes Violentos", unsafe_allow_html=True)
        df_crimes = df_filtered[list(crimes_violentos.keys())]

        with st.container(border=True):
            corr_matrix = df_crimes.corr()

            # Mudar os nomes das colunas e índices para nomes amigáveis
            corr_matrix.rename(columns=crimes_violentos, index=crimes_violentos, inplace=True)

            fig10 = px.imshow(corr_matrix, 
                    text_auto='.2f',
                    labels=dict(color="Correlação"),
                    color_continuous_scale='RdBu_r',
                    aspect='auto')
            fig10.update_layout(margin=dict(t=30, b=50, l=50, r=10))
            fig10.update_coloraxes(showscale=False)
            st.plotly_chart(fig10, use_container_width=True)

        with st.container(border=True):
            # Top correlações
            st.markdown("<b>Top 5 Correlações entre Crimes Violentos</b>", unsafe_allow_html=True)
            corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            corr_sorted = corr_flat.unstack().sort_values(ascending=False).dropna().head(5)
            
            # Transformar em DataFrame
            corr_df = pd.DataFrame(corr_sorted, columns=["Correlação"]).reset_index()
            corr_df.columns = ["Variável 1", "Variável 2", "Correlação"]
            
            # Mostrar o DataFrame
            st.dataframe(corr_df, hide_index=True, use_container_width=True)

        with st.container(border=True):
            # Criar DataFrame com explicações sobre os níveis de correlação
            explicacoes = pd.DataFrame({
                "Nível de Correlação": [
                    "Correlação Forte Positiva",
                    "Correlação Moderada Positiva",
                    "Correlação Fraca Positiva",
                    "Correlação Fraca Negativa",
                    "Correlação Moderada Negativa",
                    "Correlação Forte Negativa",
                    "Sem Correlação"
                ],
                "Descrição": [
                    "As variáveis aumentam juntas (0.7 a 1.0).",
                    "As variáveis tendem a aumentar juntas (0.3 a 0.7).",
                    "Pequena tendência de aumento conjunto (0.0 a 0.3).",
                    "Pequena tendência de uma variável aumentar enquanto a outra diminui (-0.3 a 0.0).",
                    "As variáveis tendem a se mover em direções opostas (-0.7 a -0.3).",
                    "Uma variável aumenta enquanto a outra diminui significativamente (-1.0 a -0.7).",
                    "Não há relação linear aparente entre as variáveis (próximo de 0)."
                ]
            })

            # Exibir o DataFrame no Streamlit
            st.markdown("<b>Explicação dos Níveis de Correlação</b>", unsafe_allow_html=True)
            st.dataframe(explicacoes, use_container_width=True, hide_index=True)


    with tab2:
        st.markdown("##### <br>Impacto das Apreensões na Criminalidade", unsafe_allow_html=True)
        
        with st.container(border=True):
            tipo_apre = st.selectbox(
                "Selecione o Tipo de Apreensão:",
                ['arm_fogo_apre', 'arm_branc_apr', 'drog_kg_apr', 'drog_un_apr'],
                format_func=lambda x: {
                    'arm_fogo_apre': 'Armas de Fogo',
                    'arm_branc_apr': 'Armas Brancas',
                    'drog_kg_apr': 'Drogas (kg)',
                    'drog_un_apr': 'Drogas (unidades)'
                }[x]
            )

            # Nome amigável para o tipo de apreensão
            nome_tipo_apre = {
                'arm_fogo_apre': 'Apreensões de Armas de Fogo',
                'arm_branc_apr': 'Apreensões de Armas Brancas',
                'drog_kg_apr': 'Apreensões de Drogas (kg)',
                'drog_un_apr': 'Apreensões de Drogas (unidades)'
            }[tipo_apre]
        
        crimes_principais = {
            'hom': 'Homicídios',
            'fem': 'Feminicídios',
            'vias_fato': 'Vias de Fato',
            'total_furtos': 'Total de Furtos',
            'total_roubos': 'Total de Roubos'
        }

        with st.container(border=True):
            st.markdown(f"**Correlação entre {nome_tipo_apre} e Crimes**", unsafe_allow_html=True)
            resultados = []
            for crime, nome in crimes_principais.items():
                if crime in ['hom', 'fem']:
                    dados = df_filtered[[tipo_apre, crime]].dropna()
                    if len(dados) > 3:
                        corr, pval = pearsonr(dados[tipo_apre], dados[crime])
                        resultados.append({'Crime': nome, 'Correlação': corr, 'P-valor': pval})
                else:
                    corr, pval = pearsonr(df_filtered[tipo_apre], df_filtered[crime])
                    resultados.append({'Crime': nome, 'Correlação': corr, 'P-valor': pval})
            
            df_resultados = pd.DataFrame(resultados)
            df_resultados['Significância'] = df_resultados['P-valor'].apply(
            lambda x: '✅ Sim' if x < 0.05 else '❌ Não'
            )
            st.dataframe(df_resultados, use_container_width=True)
        
        with st.container(border=True):
            st.markdown(f"**Gráfico de Dispersão: {nome_tipo_apre} vs Total de Roubos**", unsafe_allow_html=True)
            fig11 = px.scatter(df_filtered, x=tipo_apre, y='total_roubos',
                            color='ano', size='total_roubos',
                            hover_data=['mes'],
                            trendline='ols')
            st.plotly_chart(fig11, use_container_width=True, margin=dict(t=10, b=10, l=10, r=10))


    with tab3:
        st.markdown("##### <br>Análise de Sazonalidade das Ocorrências Atendidas", unsafe_allow_html=True)
    
        with st.container(border=True):
            st.markdown("**Média Mensal de Ocorrências Atendidas no Triênio**", unsafe_allow_html=True)
            ocor_por_mes = df_filtered.groupby('mes', observed=True)['ocor_atend'].mean().reset_index()
        
            fig50 = px.bar(ocor_por_mes, x='mes', y='ocor_atend', color_discrete_sequence=['#002156'])
            fig50.update_traces(texttemplate='%{y:,.2f}', textposition='outside')
            fig50.update_layout(margin=dict(t=20, b=0, l=50, r=50))
            st.plotly_chart(fig50, use_container_width=True)
        
        

        with st.container(border=True):
            st.markdown("**Distribuição de Ocorrências por Mês e Ano**", unsafe_allow_html=True)
            fig60 = px.box(df_filtered, x='mes', y='ocor_atend', color='ano', color_discrete_map={
                            '2022': '#002156',
                            '2023': '#ffbb3c',
                            '2024': '#ec152f'
                        })
            fig60.update_layout(margin=dict(t=20, b=50, l=50, r=50), boxmode='group')
            st.plotly_chart(fig60, use_container_width=True)
        
        with st.container(border=True):
            st.markdown("**Estatísticas Descritivas por Mês**", unsafe_allow_html=True)
            stats_mes = df_filtered.groupby('mes', observed=True)['ocor_atend'].agg([
                ('Média', 'mean'),
                ('Mediana', 'median'),
                ('Desvio Padrão', 'std'),
                ('Mínimo', 'min'),
                ('Máximo', 'max')
            ]).round(2)
            st.dataframe(stats_mes, use_container_width=True)

