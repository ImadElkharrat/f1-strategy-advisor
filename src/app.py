import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import numpy as np
from models import TireDegradationModel
from simulation import RaceSimulator, DuelSimulator

# --- CONFIGURATION ---
st.set_page_config(page_title="F1 Strategy AI", layout="wide")


# --- CACHED FUNCTIONS ---
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/2023_Bahrain_laps.parquet")


@st.cache_resource
def train_model(driver_code, _df):
    """
    Trains the Bayesian model for a specific driver.
    Cached so it doesn't re-run when you switch tabs.
    """
    model = TireDegradationModel(driver_code)
    driver_data, compounds, map_ = model.prepare_data(_df)

    # Run MCMC (Single Core for Windows Stability)
    with model.build_model(
            driver_data['TyreLife'].values,
            driver_data['CompoundIdx'].values,
            driver_data['LapTime_Adjusted'].values,
            len(compounds)
    ):
        trace = pm.sample(draws=200, tune=200, chains=1, cores=1, progressbar=True)

    return trace, compounds, map_


# --- MAIN APP UI ---
st.title("🏎️ Probabilistic F1 Strategy")

# Global Settings (Sidebar)
st.sidebar.header("Global Settings")
pit_loss = st.sidebar.slider("Est. Pit Loss (sec)", 18.0, 26.0, 22.0)

# Load Data Once
data = load_data()

# Create Tabs
tab_analysis, tab_duel = st.tabs(["📉 Tire Degradation Analysis", "⚔️ Strategy Duel"])

# =========================================
# TAB 1: TIRE DEGRADATION (Physics View)
# =========================================
with tab_analysis:
    st.header("Tire Physics & Single-Car Optimization")

    # 1. Driver Selection
    driver = st.selectbox("Select Driver to Analyze", ["VER", "LEC", "HAM", "ALO", "RUS"], index=0)

    # 2. Train Model
    with st.spinner(f"Calibrating physics model for {driver}..."):
        trace, compounds, map_ = train_model(driver, data)
    st.success(f"Model calibrated for {driver}")

    # 3. Visualization
    st.subheader("Learned Degradation Curves")
    st.caption("This chart shows how much pace (seconds) the driver loses per lap on each compound.")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Extract mean degradation rates from Posterior
    deg_means = trace.posterior['deg_rate'].mean(dim=["chain", "draw"]).values

    # Plot curves for visualization (0 to 30 laps)
    x_laps = range(31)
    colors = ['red', 'yellow', 'white']  # Soft, Medium, Hard (Standard F1 colors)

    for i, deg_rate in enumerate(deg_means):
        if i < len(compounds):
            # Linear approx: Loss = DegRate * Lap
            y_loss = [deg_rate * l for l in x_laps]
            name = compounds[i]

            # Smart color selection
            c = 'gray'
            if "SOFT" in name.upper():
                c = 'red'
            elif "MEDIUM" in name.upper():
                c = 'gold'
            elif "HARD" in name.upper():
                c = 'white'

            ax.plot(x_laps, y_loss, label=f"{name} ({deg_rate:.3f} s/lap)", color=c, linewidth=2.5)

    ax.set_facecolor('#1e1e1e')  # Dark background for chart
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.set_ylabel("Pace Loss (seconds)")
    ax.set_xlabel("Tire Age (Laps)")
    ax.grid(color='#444444', linestyle='--')
    ax.legend(facecolor='#1e1e1e', labelcolor='white')
    st.pyplot(fig)

    # 4. Single Car Simulation (The "Old" Feature)
    st.divider()
    st.subheader("Single-Car Pit Window Optimizer")
    c1, c2 = st.columns(2)
    with c1:
        curr_lap_single = st.number_input("Current Lap", 1, 50, 15, key="single_lap")
    with c2:
        target_pit = st.slider("Alternative Pit Lap", curr_lap_single + 1, 57, curr_lap_single + 5, key="single_target")

    if st.button("Analyze Strategy Risk"):
        sim = RaceSimulator(trace, driver, curr_lap_single)

        # Sim Box Now
        res_now = sim.simulate_strategy(curr_lap_single + 1, pit_loss)
        # Sim Box Later
        res_later = sim.simulate_strategy(target_pit, pit_loss)

        delta = res_later.mean() - res_now.mean()

        if delta > 0:
            st.success(f"Recommendation: BOX NOW (Gain: {delta:.2f}s)")
        else:
            st.warning(f"Recommendation: WAIT (Loss: {delta:.2f}s)")

# =========================================
# TAB 2: DUEL MODE (Strategy View)
# =========================================
with tab_duel:
    st.header("Head-to-Head Undercut Simulator")

    # 1. Duel Setup
    col_h, col_r = st.columns(2)

    with col_h:
        hero_driver = st.selectbox("Hero (Us)", ["VER", "HAM", "LEC", "ALO"], index=0, key="hero")
        hero_pit = st.slider(f"Pit Lap: {hero_driver}", 10, 57, 20, key="pit_hero")

    with col_r:
        # Filter rival list to exclude hero
        rivals = [d for d in ["VER", "HAM", "LEC", "ALO"] if d != hero_driver]
        rival_driver = st.selectbox("Rival (Them)", rivals, index=0, key="rival")
        rival_pit = st.slider(f"Pit Lap: {rival_driver}", 10, 57, 24, key="pit_rival")

    # 2. Race Context
    st.caption("Race Situation")
    c_gap, c_lap = st.columns(2)
    with c_gap:
        gap_to_rival = st.number_input("Gap to Rival (sec)", -10.0, 10.0, 2.0, help="Positive = We are BEHIND")
    with c_lap:
        current_lap_duel = st.number_input("Current Lap", 10, 50, 15, key="duel_lap")

    # 3. Run Simulation
    if st.button("⚔️ Simulate Duel"):
        # Train both models on the fly (cached)
        with st.spinner("Calibrating Hero & Rival models..."):
            trace_h, _, _ = train_model(hero_driver, data)
            trace_r, _, _ = train_model(rival_driver, data)

        # Init Simulators
        sim_h = RaceSimulator(trace_h, hero_driver, current_lap_duel)
        sim_r = RaceSimulator(trace_r, rival_driver, current_lap_duel)
        duel = DuelSimulator(sim_h, sim_r)

        # Run Math
        gap_dist = duel.simulate_gap(hero_pit, rival_pit, gap_to_rival, pit_loss)

        # Metrics
        prob_ahead = (gap_dist < 0).mean() * 100
        avg_gap = -gap_dist.mean()  # Invert so positive = leading

        # Display
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Probability of Overtake", f"{prob_ahead:.1f}%")
        m2.metric("Predicted Lead", f"{avg_gap:.2f} sec", delta_color="normal")

        # Visual
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.kdeplot(-gap_dist, fill=True, color='purple', alpha=0.6)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label="Overtake Line")

        ax2.set_title(f"Forecast: {hero_driver} vs {rival_driver}")
        ax2.set_xlabel(f"Gap (sec) - Right of Red Line = {hero_driver} Ahead")
        ax2.set_yticks([])  # Hide y-axis for clean look
        ax2.legend()

        st.pyplot(fig2)