import streamlit as st
import math
from historical_data import load_history
st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Football Predictor")
st.caption("Predicción de partidos reales")

st.subheader("🏟️ Partido")
liga = st.selectbox(
    "Competición",
    ["Primera División", "Segunda División"]
)

temporadas = [
    "2526", "2425", "2324", "2223", "2122",
    "2021", "1920", "1819", "1718", "1617"
]

@st.cache_data(ttl=3600)
def cargar_datos(liga):
    return load_history(liga, temporadas)

datos = cargar_datos(liga)
if not datos.empty:
    equipos = sorted(
        set(datos["HomeTeam"].dropna()) |
        set(datos["AwayTeam"].dropna())
    )

    local = st.selectbox(
        "Equipo local",
        equipos,
        index=0
    )

    equipos_visitantes = [e for e in equipos if e != local]

    visitante = st.selectbox(
        "Equipo visitante",
        equipos_visitantes,
        index=0
    )
else:
    st.error("No se pudieron cargar los datos históricos.")
    st.stop()

st.subheader("📊 Datos recientes")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### {local}")
    gf_local = st.number_input(
        "Goles a favor por partido (local)",
        min_value=0.0,
        value=1.30,
        step=0.1
    )
    gc_local = st.number_input(
        "Goles recibidos por partido (local)",
        min_value=0.0,
        value=1.00,
        step=0.1
    )

with col2:
    st.markdown(f"### {visitante}")
    gf_visit = st.number_input(
        "Goles a favor por partido (visitante)",
        min_value=0.0,
        value=1.10,
        step=0.1
    )
    gc_visit = st.number_input(
        "Goles recibidos por partido (visitante)",
        min_value=0.0,
        value=1.30,
        step=0.1
    )

def poisson(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

if st.button("🔮 ANALIZAR PARTIDO", type="primary"):

    xg_local = (gf_local + gc_visit) / 2
    xg_visit = (gf_visit + gc_local) / 2

    max_goals = 8

    home = 0
    draw = 0
    away = 0
    over15 = 0
    over25 = 0
    over35 = 0
    btts = 0

    scores = []

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):

            p = poisson(i, xg_local) * poisson(j, xg_visit)

            scores.append((p, i, j))

            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p

            if i + j > 1:
                over15 += p

            if i + j > 2:
                over25 += p

            if i + j > 3:
                over35 += p

            if i > 0 and j > 0:
                btts += p

    st.subheader("🏆 Probabilidades 1X2")

    c1, c2, c3 = st.columns(3)

    c1.metric(local, f"{home*100:.1f}%")
    c2.metric("Empate", f"{draw*100:.1f}%")
    c3.metric(visitante, f"{away*100:.1f}%")

    st.subheader("⚽ Goles esperados")

    c1, c2 = st.columns(2)

    c1.metric(local, f"{xg_local:.2f}")
    c2.metric(visitante, f"{xg_visit:.2f}")

    st.subheader("📊 Mercados estadísticos")

    st.write(f"Over 1.5: **{over15*100:.1f}%**")
    st.write(f"Over 2.5: **{over25*100:.1f}%**")
    st.write(f"Under 2.5: **{(1-over25)*100:.1f}%**")
    st.write(f"Over 3.5: **{over35*100:.1f}%**")
    st.write(f"Ambos marcan · Sí: **{btts*100:.1f}%**")
    st.write(f"Ambos marcan · No: **{(1-btts)*100:.1f}%**")

    st.subheader("🎯 Marcadores más probables")

    scores.sort(reverse=True)

    for p, i, j in scores[:5]:
        st.write(
            f"**{local} {i} - {j} {visitante}** "
            f"— {p*100:.1f}%"
        )