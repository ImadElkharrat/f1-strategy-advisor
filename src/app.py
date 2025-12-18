import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import numpy as np
from models import TireDegradationModel
from simulation import RaceSimulator, DuelSimulator

st.set_page_config(page_title="F1 Strategy Duel", layout="wide")


@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/2023_Bahrain_laps.parquet")


@st.cache_resource
def train_model(driver_code, _df):
    """Trains a model for a specific driver."""
    model = TireDegradationModel(driver_code)
    driver_data, compounds, map_ = model.prepare_data(_df)

    with model.build_model(
            driver_data['TyreLife'].values,
            driver_data['CompoundIdx'].values,
            driver_data['LapTime_Adjusted'].values,
            len(compounds)
    ):
        trace = pm.sample(draws=200, tune=200, chains=1, cores=1, progressbar=True)

    return trace, compounds, map_


# --- SIDEBAR CONFIG ---
st.sidebar.header("⚔️ Strategy Duel Config")

# Select Hero
hero_driver = st.sidebar.selectbox("Select Hero (Us)", ["VER", "HAM"], index=0)
# Select Rival (excluding Hero)
available_rivals = [d for d in ["VER", "HAM", "LEC", "ALO"] if d != hero_driver]
rival_driver = st.sidebar.selectbox("Select Rival (Them)", available_rivals, index=0)

current_lap = st.sidebar.number_input("Current Lap", 10, 50, 15)
gap_to_rival = st.sidebar.number_input("Gap to Rival (sec)", -10.0, 10.0, 2.5,
                                       help="Positive = We are behind. Negative = We are ahead.")
pit_loss = st.sidebar.slider("Pit Loss Time", 18.0, 25.0, 22.0)

# --- MAIN UI ---
st.title(f"Strategy Duel: {hero_driver} vs {rival_driver}")

data = load_data()

# Train Both Models
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### 🔵 {hero_driver} Intelligence")
    with st.spinner(f"Analyzing {hero_driver}..."):
        trace_hero, _, _ = train_model(hero_driver, data)
    st.success("Ready")

with col2:
    st.markdown(f"### 🔴 {rival_driver} Intelligence")
    with st.spinner(f"Analyzing {rival_driver}..."):
        trace_rival, _, _ = train_model(rival_driver, data)
    st.success("Ready")

# --- SIMULATION ---
st.divider()
st.subheader("🔮 The Undercut Simulator")

# Input: When do we pit? When do they pit?
c1, c2 = st.columns(2)
with c1:
    hero_pit = st.slider(f"Pit Lap: {hero_driver}", current_lap + 1, 57, current_lap + 1)
with c2:
    rival_pit = st.slider(f"Pit Lap: {rival_driver}", current_lap + 1, 57, current_lap + 5)

if st.button("Simulate Duel"):
    # 1. Init Simulators
    sim_hero = RaceSimulator(trace_hero, hero_driver, current_lap)
    sim_rival = RaceSimulator(trace_rival, rival_driver, current_lap)
    duel = DuelSimulator(sim_hero, sim_rival)

    # 2. Run Gap Simulation
    # Result is distribution of (HeroTime - RivalTime)
    gap_dist = duel.simulate_gap(hero_pit, rival_pit, gap_to_rival, pit_loss)

    # 3. Analysis
    # If gap < 0, Hero is AHEAD (Time is smaller)
    prob_overtake = (gap_dist < 0).mean() * 100
    expected_gap = -gap_dist.mean()  # Invert so Positive = Lead

    # 4. Display
    st.markdown(f"### Probability of Being Ahead: **{prob_overtake:.1f}%**")

    if expected_gap > 0:
        st.success(f"Prediction: {hero_driver} leads by {expected_gap:.2f} sec")
    else:
        st.error(f"Prediction: {hero_driver} trails by {abs(expected_gap):.2f} sec")

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot the distribution of the final gap
    # We invert it so Right = Good (Leading)
    sns.kdeplot(-gap_dist, fill=True, color='purple', alpha=0.6)
    plt.axvline(0, color='red', linestyle='--', label="Overtake Threshold")

    plt.title(f"Predicted Gap at Finish Line ({hero_driver} vs {rival_driver})")
    plt.xlabel(f"Gap (seconds) - Positive means {hero_driver} is Ahead")
    plt.legend()
    st.pyplot(fig)