import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard Bayesiano PMDF",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega e processa os dados da PMDF"""
    try:
        # Tentar carregar o arquivo
        df = pd.read_csv('data/PMDF_ocorrencias_2022-2024.csv')
        
        # CORREÇÃO: Criar mapeamento de mês para número
        mes_num = {
            'JANEIRO': 1, 'FEVEREIRO': 2, 'MARÇO': 3,
            'ABRIL': 4, 'MAIO': 5, 'JUNHO': 6,
            'JULHO': 7, 'AGOSTO': 8, 'SETEMBRO': 9,
            'OUTUBRO': 10, 'NOVEMBRO': 11, 'DEZEMBRO': 12
        }
        
        # mapeia o nome do mês para número e cria coluna datetime com dia 1
        df['data'] = pd.to_datetime(
            df['ano'].astype(str) + '-' + df['mes'].map(mes_num).astype(int).astype(str).str.zfill(2) + '-01'
        )
        
        # Ordenar por data
        df = df.sort_values('data').reset_index(drop=True)
        
        # Traduzir meses para português
        meses_pt = {
            'JANEIRO': 'Janeiro', 'FEVEREIRO': 'Fevereiro', 'MARÇO': 'Março',
            'ABRIL': 'Abril', 'MAIO': 'Maio', 'JUNHO': 'Junho',
            'JULHO': 'Julho', 'AGOSTO': 'Agosto', 'SETEMBRO': 'Setembro',
            'OUTUBRO': 'Outubro', 'NOVEMBRO': 'Novembro', 'DEZEMBRO': 'Dezembro'
        }
        df['mes_nome'] = df['mes'].map(meses_pt)
        
        return df
        
    except FileNotFoundError:
        st.error("Arquivo 'PMDF_ocorrencias_2022-2024.csv' não encontrado!")
        st.info("Por favor, certifique-se de que o arquivo está no diretório correto.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.info("Verifique se o arquivo CSV está formatado corretamente.")
        return None

# Navegação lateral
def main():
    """Função principal do dashboard"""
    
    # Sidebar para navegação
    st.sidebar.markdown("# 🚔 Dashboard PMDF")
    st.sidebar.markdown("### Análise Bayesiana de Ocorrências Criminais")
    st.sidebar.markdown("---")
    
    # Menu de navegação
    page = st.sidebar.selectbox(
        "Navegação",
        ["📊 Análise Exploratória", "🧮 Modelos Bayesianos", "📋 Análise de Resultados"],
        index=0
    )
    
    # Carregar dados
    df = load_data()
    
    if df is not None:
        # Roteamento de páginas
        if page == "📊 Análise Exploratória":
            pagina_eda(df)
        elif page == "🧮 Modelos Bayesianos":
            pagina_modelos_bayesianos()
        elif page == "📋 Análise de Resultados":
            pagina_analise_resultados()
    else:
        st.stop()

def pagina_eda(df):
    """Página de Análise Exploratória Completa"""
    
    # Cabeçalho
    st.markdown('<h1 class="main-header">📊 Análise Exploratória dos Dados</h1>', unsafe_allow_html=True)
    
    # Informações do dataset
    st.markdown('<h2 class="section-header">📋 Informações Gerais do Dataset</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total de Observações", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Período Analisado", f"{df['ano'].min()}-{df['ano'].max()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total de Variáveis", f"{len(df.columns)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        total_ocorrencias = df['ocor_atend'].sum()
        st.metric("Total de Ocorrências", f"{total_ocorrencias:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Insights principais
    st.markdown("""
    <div class="insight-box">
    <h4>📌 Insights Principais do Dataset</h4>
    <ul>
        <li><strong>Estrutura temporal:</strong> 36 meses de dados (2022-2024), permitindo análise de tendências e sazonalidade</li>
        <li><strong>Variável principal:</strong> 'ocor_atend' (ocorrências atendidas) sem dados faltantes em todo período</li>
        <li><strong>Dados faltantes estruturais:</strong> Variáveis de homicídio ausentes em 2022 (decisão institucional da PMDF)</li>
        <li><strong>Região de estudo:</strong> Distrito Federal (RIDE-DF), área metropolitana de Brasília</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise da variável principal
    st.markdown('<h2 class="section-header">🎯 Análise da Variável Principal: Ocorrências Atendidas</h2>', unsafe_allow_html=True)
    
    # Gráfico temporal principal
    fig_temporal = px.line(
        df, x='data', y='ocor_atend',
        title='Evolução Temporal das Ocorrências Atendidas pela PMDF (2022-2024)',
        labels={'ocor_atend': 'Ocorrências Atendidas', 'data': 'Período'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Adicionar tendência
    from scipy import stats
    x_numeric = np.arange(len(df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df['ocor_atend'])
    trend_line = slope * x_numeric + intercept
    
    fig_temporal.add_trace(
        go.Scatter(
            x=df['data'], 
            y=trend_line, 
            mode='lines', 
            name='Tendência Linear',
            line=dict(dash='dash', color='red')
        )
    )
    
    # Marcar anomalia de fevereiro 2024
    fev_2024 = df[df['mes_ano'] == 'FEV_2024']
    if not fev_2024.empty:
        fig_temporal.add_trace(
            go.Scatter(
                x=fev_2024['data'], 
                y=fev_2024['ocor_atend'],
                mode='markers', 
                name='Anomalia (Fev/2024)',
                marker=dict(size=15, color='orange', symbol='star')
            )
        )
    
    fig_temporal.update_layout(
        height=500,
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig_temporal, use_container_width=True)
    
    # Interpretação da tendência
    tendencia_anual = slope * 12  # slope mensal * 12 meses
    r2 = r_value ** 2
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>📈 Interpretação da Tendência Temporal</h4>
    <ul>
        <li><strong>Tendência:</strong> {'Decrescente' if slope < 0 else 'Crescente'} de {abs(tendencia_anual):.0f} ocorrências por ano</li>
        <li><strong>Coeficiente de determinação (R²):</strong> {r2:.3f} - {'Alta' if r2 > 0.7 else 'Moderada' if r2 > 0.4 else 'Baixa'} correlação linear</li>
        <li><strong>Significância estatística:</strong> {'Significativa' if p_value < 0.05 else 'Não significativa'} (p-valor: {p_value:.4f})</li>
        <li><strong>Anomalia identificada:</strong> Fevereiro/2024 com {fev_2024['ocor_atend'].iloc[0] if not fev_2024.empty else 'N/A'} ocorrências (outlier)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Estatísticas por ano
    st.markdown('<h2 class="section-header">📊 Estatísticas Descritivas por Ano</h2>', unsafe_allow_html=True)
    
    # Calcular estatísticas
    stats_por_ano = df.groupby('ano')['ocor_atend'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(0)
    
    # Criar gráfico de boxplot
    fig_box = px.box(
        df, x='ano', y='ocor_atend',
        title='Distribuição das Ocorrências por Ano',
        labels={'ocor_atend': 'Ocorrências Atendidas', 'ano': 'Ano'}
    )
    fig_box.update_layout(height=400)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.markdown("**Estatísticas Descritivas:**")
        st.dataframe(stats_por_ano, use_container_width=True)
    
    # Análise de sazonalidade
    st.markdown('<h2 class="section-header">📅 Análise de Sazonalidade</h2>', unsafe_allow_html=True)
    
    # Calcular médias mensais
    sazonalidade = df.groupby('mes_nome')['ocor_atend'].mean().reindex([
        'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
        'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
    ])
    
    # Gráfico de sazonalidade
    fig_sazon = px.bar(
        x=sazonalidade.index, 
        y=sazonalidade.values,
        title='Padrão Sazonal Médio das Ocorrências (2022-2024)',
        labels={'x': 'Mês', 'y': 'Média de Ocorrências'}
    )
    fig_sazon.update_layout(height=400)
    
    st.plotly_chart(fig_sazon, use_container_width=True)
    
    # Identificar picos e vales sazonais
    mes_maior = sazonalidade.idxmax()
    valor_maior = sazonalidade.max()
    mes_menor = sazonalidade.idxmin()
    valor_menor = sazonalidade.min()
    variacao_sazonal = ((valor_maior - valor_menor) / valor_menor) * 100
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>📅 Padrão Sazonal Identificado</h4>
    <ul>
        <li><strong>Pico sazonal:</strong> {mes_maior} ({valor_maior:.0f} ocorrências em média)</li>
        <li><strong>Vale sazonal:</strong> {mes_menor} ({valor_menor:.0f} ocorrências em média)</li>
        <li><strong>Variação sazonal:</strong> {variacao_sazonal:.1f}% entre pico e vale</li>
        <li><strong>Interpretação:</strong> {'Alta sazonalidade' if variacao_sazonal > 20 else 'Moderada sazonalidade' if variacao_sazonal > 10 else 'Baixa sazonalidade'}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise de outras variáveis importantes
    st.markdown('<h2 class="section-header">🔍 Análise de Outras Variáveis Relevantes</h2>', unsafe_allow_html=True)
    
    # Seleção de variáveis para análise
    variaveis_interesse = [
        'flagrantes', 'mai_detidos', 'mai_presos_flag', 'roub_trans', 
        'roub_veic', 'furt_veic', 'arm_fogo_apre', 'drog_kg_apr'
    ]
    
    # Criar gráficos de correlação
    df_corr = df[['ocor_atend'] + variaveis_interesse].corr()['ocor_atend'].drop('ocor_atend').sort_values(ascending=True)
    
    fig_corr = px.bar(
        x=df_corr.values, 
        y=df_corr.index,
        orientation='h',
        title='Correlação com Ocorrências Atendidas',
        labels={'x': 'Coeficiente de Correlação', 'y': 'Variáveis'}
    )
    fig_corr.update_layout(height=400)
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Análise de dados faltantes
    st.markdown('<h2 class="section-header">⚠️ Análise de Dados Faltantes</h2>', unsafe_allow_html=True)
    
    # Calcular dados faltantes
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_missing = px.bar(
                x=missing_data.values, 
                y=missing_data.index,
                orientation='h',
                title='Contagem de Dados Faltantes por Variável'
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        
        with col2:
            st.markdown("**Detalhamento dos Dados Faltantes:**")
            for var, count in missing_data.items():
                porcentagem = (count / len(df)) * 100
                st.write(f"**{var}:** {count} casos ({porcentagem:.1f}%)")
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Importante: Dados Faltantes Estruturais</h4>
    <p>As variáveis relacionadas a homicídios (hom, hom_tent, fem, fem_tent, hom_culp, infant) 
    não foram disponibilizadas pela PMDF para o ano de 2022. Esta ausência é classificada como 
    <strong>MNAR (Missing Not At Random)</strong> - estrutural e sistemática.</p>
    <p><strong>Implicação para análise bayesiana:</strong> A análise dessas variáveis será restrita 
    ao período 2023-2024 (24 meses), enquanto a variável principal 'ocor_atend' utiliza todo 
    o período 2022-2024 (36 meses).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Resumo executivo
    st.markdown('<h2 class="section-header">📋 Resumo Executivo da Análise Exploratória</h2>', unsafe_allow_html=True)
    
    total_ocor = df['ocor_atend'].sum()
    media_anual = df.groupby('ano')['ocor_atend'].sum()
    reducao_2022_2024 = ((media_anual[2022] - media_anual[2024]) / media_anual[2022]) * 100
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>🎯 Principais Conclusões para Modelagem Bayesiana</h4>
    <ol>
        <li><strong>Tendência decrescente clara:</strong> Redução de {reducao_2022_2024:.1f}% nas ocorrências entre 2022 e 2024</li>
        <li><strong>Variabilidade temporal:</strong> Coeficiente de variação indica necessidade de modelo que capture incerteza</li>
        <li><strong>Sazonalidade moderada:</strong> Variação sazonal de {variacao_sazonal:.1f}% sugere componente sazonal no modelo</li>
        <li><strong>Outlier identificado:</strong> Fevereiro/2024 requer atenção especial na modelagem (posterior predictive check)</li>
        <li><strong>Dados robustos:</strong> Variável principal completa permite construção de Power Prior confiável</li>
        <li><strong>Estratégia de modelagem:</strong> Modelo Poisson-Gamma adequado para dados de contagem com over-dispersion</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

def pagina_modelos_bayesianos():
    """Página para Modelos Bayesianos (placeholder)"""
    
    st.markdown('<h1 class="main-header">🧮 Modelos Bayesianos</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h3>🚧 Em Desenvolvimento</h3>
    <p>Esta página será preenchida com:</p>
    <ul>
        <li><strong>Modelo A:</strong> Poisson-Gamma para ocorrências totais (36 meses, 2022-2024)</li>
        <li><strong>Modelo B:</strong> Poisson-Gamma para crimes violentos (24 meses, 2023-2024)</li>
        <li><strong>Power Prior:</strong> Construção de prioris informativas baseadas em dados históricos</li>
        <li><strong>Análise de sensibilidade:</strong> Comparação entre prioris informativas e não-informativas</li>
        <li><strong>Posterior Predictive Checks:</strong> Validação dos modelos</li>
        <li><strong>Predições 2025:</strong> Intervalos de credibilidade para taxa futura</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Próximos passos:** Implementação dos modelos PyMC com distribuições Poisson-Gamma conjugadas")

def pagina_analise_resultados():
    """Página para Análise de Resultados (placeholder)"""
    
    st.markdown('<h1 class="main-header">📋 Análise de Resultados</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h3>🚧 Em Desenvolvimento</h3>
    <p>Esta página será preenchida com:</p>
    <ul>
        <li><strong>Interpretação dos resultados bayesianos:</strong> Distribuições posteriori e intervalos de credibilidade</li>
        <li><strong>Comparação de modelos:</strong> Análise de sensibilidade às escolhas de prioris</li>
        <li><strong>Resposta ao problema de pesquisa:</strong> Predições para 2025 e probabilidades de exceder limiares críticos</li>
        <li><strong>Validação dos modelos:</strong> Posterior predictive checks e diagnósticos de convergência</li>
        <li><strong>Limitações e recomendações:</strong> Discussão crítica dos resultados</li>
        <li><strong>Conclusões finais:</strong> Síntese dos achados para gestão de segurança pública</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Próximos passos:** Interpretação estatística e prática dos resultados dos modelos bayesianos")

# Rodapé
def add_footer():
    """Adiciona rodapé ao dashboard"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem;'>
        📊 Dashboard de Análise Bayesiana - Dados PMDF/DF (2022-2024)<br>
        🎓 Trabalho de Inferência Bayesiana - Ciência de Dados e IA<br>
        📍 Fonte: DataIESB - Portal de Dados da RIDE-DF
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    add_footer()