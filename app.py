import streamlit as st
import pandas as pd
import importlib.util
from pathlib import Path

BASE = Path(__file__).parent
ENGINE = BASE / 'football_prediction_engine_v1(1).py'
DATA = BASE / 'matches_demo(1).csv'

spec = importlib.util.spec_from_file_location('prediction_engine', ENGINE)
engine = importlib.util.module_from_spec(spec)
import sys
sys.modules['prediction_engine'] = engine
spec.loader.exec_module(engine)

st.set_page_config(page_title='Football Predictor', page_icon='⚽', layout='centered')
st.title('⚽ Football Predictor')
st.caption('Motor V1.0 · Poisson + Dixon-Coles + Elo + Monte Carlo')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()
teams = sorted(set(df['home_team']).union(df['away_team']))

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox('🏠 Equipo local', teams, index=teams.index('Alpha') if 'Alpha' in teams else 0)
with col2:
    away_options = [t for t in teams if t != home] or teams
    away = st.selectbox('✈️ Equipo visitante', away_options, index=away_options.index('Beta') if 'Beta' in away_options else 0)

simulations = st.slider('Simulaciones Monte Carlo', 10000, 200000, 100000, 10000)

if st.button('🔮 ANALIZAR PARTIDO', type='primary', use_container_width=True):
    with st.spinner('Calculando...'):
        r = engine.predict(df, home, away, simulations)

    st.subheader(f'{home}  vs  {away}')
    c1, c2, c3 = st.columns(3)
    c1.metric('1 · Local', f"{float(r['home_win']):.2f}%")
    c2.metric('X · Empate', f"{float(r['draw']):.2f}%")
    c3.metric('2 · Visitante', f"{float(r['away_win']):.2f}%")

    st.markdown('### ⚽ Goles esperados')
    g1, g2 = st.columns(2)
    g1.metric(home, f"{r['expected_goals_home']:.2f}")
    g2.metric(away, f"{r['expected_goals_away']:.2f}")

    st.markdown('### 📊 Mercados estadísticos')
    markets = pd.DataFrame([
        ['Over 1.5', r['over_1_5']],
        ['Over 2.5', r['over_2_5']],
        ['Over 3.5', r['over_3_5']],
        ['Under 2.5', r['under_2_5']],
        ['Ambos marcan · Sí', r['btts_yes']],
        ['Ambos marcan · No', r['btts_no']],
    ], columns=['Mercado', 'Probabilidad %'])
    markets['Probabilidad %'] = markets['Probabilidad %'].astype(float).round(2)
    st.dataframe(markets, hide_index=True, use_container_width=True)

    st.markdown('### 🎯 Marcadores más probables')
    scores = pd.DataFrame(r['top_scores'])
    scores['probability'] = scores['probability'].astype(float)
    scores = scores.rename(columns={'score':'Marcador', 'probability':'Probabilidad %'})
    st.dataframe(scores, hide_index=True, use_container_width=True)

    st.info(f"Elo: {home} {r['elo_home']:.1f} · {away} {r['elo_away']:.1f}")

st.divider()
st.caption('Herramienta estadística experimental. Las probabilidades son estimaciones del modelo, no garantías de resultado.')
