
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

# Função para carregar dados
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/PMDF_ocorrencias_2022-2024.csv')
        
        # Criar mapeamento de mês para número
        mes_num = {
            'JANEIRO': 1, 'FEVEREIRO': 2, 'MARÇO': 3,
            'ABRIL': 4, 'MAIO': 5, 'JUNHO': 6,
            'JULHO': 7, 'AGOSTO': 8, 'SETEMBRO': 9,
            'OUTUBRO': 10, 'NOVEMBRO': 11, 'DEZEMBRO': 12
        }
        # Criar coluna de data
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

        # Preparar os dados
        df['ano'] = df['ano'].astype(str)

        # Criar ordem correta dos meses
        meses_ordem = ['JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO', 
                    'JULHO', 'AGOSTO', 'SETEMBRO', 'OUTUBRO', 'NOVEMBRO', 'DEZEMBRO']
        df['mes'] = pd.Categorical(df['mes'], categories=meses_ordem, ordered=True)
                
        return df
        
    except FileNotFoundError:
        st.error("Arquivo CSV não encontrado!")
        st.info("Por favor, certifique-se de que o arquivo está no diretório correto.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.info("Verifique se o arquivo CSV está formatado corretamente.")
        return None

df = load_data()


