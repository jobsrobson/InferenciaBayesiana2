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

def formatar_numero_br(valor):
    return f"{valor:,.0f}".replace(",", ".")

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
    with st.expander("Explicação dos Níveis de Correlação", icon=":material/info:", expanded=True):
        st.markdown(""" 
                    Os dados revelam padrões nítidos de associação entre diferentes tipos de crimes violentos. A correlação mais intensa ocorre entre **Violência Doméstica** e **Vias de Fato** (0,96), indicando que esses eventos tendem a ocorrer conjuntamente e provavelmente refletem dinâmicas semelhantes de conflito interpessoal. Em seguida, destaca-se a forte relação entre **Tentativa de Feminicídio** e **Tentativa de Homicídio** (0,66), o que sugere que ambos emergem de contextos de violência letal mal consumada, compartilhando um mesmo perfil de escalada agressiva.
                    Outro ponto relevante é que **Vias de Fato** apresenta correlação consistente com várias categorias — especialmente **Tentativa de Homicídio** e **Tentativa de Feminicídio** — reforçando sua natureza como etapa preliminar ou manifestação menos grave dentro de um continuum de violência. Já crimes como **Homicídios**, **Feminicídios** e **Homicídios Culposos** mostram correlações mais discretas com as demais variáveis, indicando dinâmicas menos previsíveis ou mais independentes no conjunto analisado.
                    No geral, os padrões sugerem que violência doméstica e conflitos físicos de menor gravidade são altamente interligados e constituem importantes indicadores de risco para ocorrências mais severas. Essa interdependência pode orientar políticas de prevenção e atuação mais integrada entre órgãos de segurança e proteção social.
                    """)
        st.markdown("<br>", unsafe_allow_html=True)
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
    with st.expander("Explicação dos Achados", icon=":material/info:", expanded=True):
        if tipo_apre == 'arm_fogo_apre':
            st.markdown(""" 
                        Os dados mostram que as **apreensões de armas de fogo** apresentam impactos distintos sobre diferentes categorias criminais. Entre os indicadores avaliados, apenas **Vias de Fato** e **Total de Furtos** exibem correlações estatisticamente significativas.
                        - **Vias de Fato**: correlação de **0,4483** (p = 0,0061) - Indica que regiões com mais conflitos físicos também tendem a registrar maior número de apreensões, possivelmente devido a operações policiais motivadas por denúncias ou confrontos.
                        - **Total de Furtos**: correlação de **0,6301** (p < 0,0001) - Sugere que o aumento das apreensões ocorre em contextos onde há maior atividade criminosa patrimonial, refletindo áreas mais problemáticas ou com policiamento mais intenso.
                        
                        Por outro lado, crimes mais graves — como **Homicídios**, **Feminicídios** e **Roubos** — **não apresentam correlações estatisticamente significativas**. Isso sugere que, no período analisado, o volume de armas removidas de circulação **não se traduz diretamente** em mudanças nesses delitos.
                        
                        O gráfico de dispersão entre apreensões e total de roubos reforça essa leitura:
                        
                        - **2022**: há uma tendência levemente negativa, indicando que mais apreensões podem estar associadas a redução de roubos nesses anos;
                        - **2023 e 2024**: observa-se o oposto, com forte tendência positiva, sugerindo aumento simultâneo de apreensões e roubos.
                        
                        Esse comportamento **inconsistente ao longo dos anos** aponta para a influência de fatores externos, mudanças operacionais ou variações estruturais na dinâmica criminosa.
                        
                        As apreensões de armas parecem estar mais ligadas a **crimes de menor gravidade ou situacionais** do que a delitos letais ou altamente organizados. Para medir com precisão o impacto sobre crimes violentos, seria necessário recorrer a séries temporais mais longas e modelos causais que controlem variáveis contextuais.
                        """)
            
        elif tipo_apre == 'arm_branc_apr':
            st.markdown(""" 
                        As apreensões de **armas brancas** mostram correlações significativamente positivas com quatro tipos de crimes: **homicídios**, **vias de fato**, **furtos** e **roubos**. A associação mais forte aparece em furtos e homicídios, sugerindo que regiões com maior circulação desse tipo de arma também registram níveis mais altos de violência letal e crimes patrimoniais. O fato de vias de fato também apresentar significância reforça a ideia de que conflitos interpessoais estão diretamente ligados ao porte ou uso de armas brancas.
                        
                        O gráfico de dispersão mostra uma tendência crescente em todos os anos analisados, indicando que, conforme aumentam as apreensões, também aumentam os registros de roubos — especialmente em 2024, onde a inclinação da linha é mais acentuada. Isso sugere que as apreensões podem estar ocorrendo em contextos de maior atividade criminosa geral, e não necessariamente reduzindo incidentes relacionados. As correlações positivas e estatisticamente significativas apontam para uma dinâmica em que apreensões refletem a intensidade da criminalidade local, funcionando mais como um indicador de cenário do que como fator redutor imediato.
                        """)
            
        elif tipo_apre == 'drog_kg_apr':
            st.markdown(""" 
                        As apreensões de drogas em quantidade (kg) não apresentam correlação significativa com nenhum dos crimes analisados. Todas as associações possuem p-valores elevados, indicando ausência de relação estatisticamente detectável entre o volume apreendido e variações em homicídios, furtos, roubos ou conflitos físicos. Isso sugere que operações de grande porte contra o tráfico — que costumam resultar em apreensões de centenas ou milhares de quilos — não impactam diretamente os indicadores criminais de curto prazo.
                        
                        O gráfico reforça essa leitura: embora haja variação expressiva na quantidade apreendida, os pontos permanecem dispersos sem formar uma tendência clara. Em 2022 e 2023 há leve inclinação negativa, mas em 2024 a direção muda completamente, revelando que o comportamento anual é inconsistente. Dessa forma, apreensões volumosas parecem refletir ações pontuais, sem efeito direto na dinâmica cotidiana dos crimes analisados.
                        """)
            
        elif tipo_apre == 'drog_un_apr':
            st.markdown(""" 
                        As apreensões de drogas medidas em unidades também não mostram correlações estatisticamente significativas com nenhum tipo de crime. Os coeficientes são baixos e os p-valores elevados, indicando que o número total de porções apreendidas não se relaciona de forma consistente com homicídios, furtos, roubos ou ocorrências de vias de fato.
                        
                        O gráfico de dispersão reforça a ausência de padrão: mesmo com apreensões variando de poucas unidades a mais de dez mil, os crimes analisados não acompanham essa oscilação. Em 2024 há uma leve tendência positiva entre apreensões e roubos, mas em 2022 e 2023 a tendência é negativa, evidenciando novamente um comportamento irregular. Isso sugere que apreensões de pequenas porções — geralmente associadas ao varejo de drogas — não exercem impacto direto sobre os indicadores criminais agregados no período estudado.
                        """)
        
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
    with st.expander("Explicação dos Achados", icon=":material/info:", expanded=True):
        st.markdown(""" 
                    A análise sazonal mostra que o volume de ocorrências atendidas no DF segue um padrão relativamente estável ao longo do ano, mas com picos evidentes em alguns meses específicos. Fevereiro apresenta o maior nível médio de ocorrências, sugerindo um período de maior demanda operacional para as forças de segurança. Em contrapartida, setembro e novembro registram as menores médias, indicando meses estruturalmente menos intensos.
                    
                    A comparação entre média e mediana confirma que a distribuição mensal é consistente, com poucas distorções causadas por valores muito altos ou muito baixos. Já o desvio padrão revela maior variabilidade em meses como fevereiro, março e outubro, o que indica ocorrência de eventos atípicos ou operações pontuais que elevam o número de registros. No geral, os dados apontam para uma sazonalidade moderada, com meses de maior pressão operacional bem delimitados e outros de comportamento mais homogêneo.
                    """)