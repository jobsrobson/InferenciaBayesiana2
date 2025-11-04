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


with open('resultados_bayesianos_completos.pkl', 'rb') as f:
    resultados = pickle.load(f)

# Show pickled data (optional)
st.write("Resultados Bayesianos carregados com sucesso!")
st.write(resultados)