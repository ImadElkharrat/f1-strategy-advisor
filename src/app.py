import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm  # <--- This was missing
import arviz as az
from models import TireDegradationModel
from simulation import RaceSimulator

# Page Config
st.set_page_config(page_title="F1 Strategy AI", layout="wide")


@st.cache_data
def load_data():
    # Load the processed data
    return pd.read_parquet("data/processed/2023_Bahrain_laps.parquet")


@st.cache_resource
def train_model(driver_code, _df):
    model = TireDegradationModel(driver_code)

    # --- CRITICAL FIX 1: Prepare Data ---
    # We must generate 'CompoundIdx' before building the model
    driver_data, compounds, compound_map = model.prepare_data(_df)

    # --- CRITICAL FIX 2: Single Core for Windows ---
    # We force cores=1 and chains=1 to prevent the app from crashing on Windows
    with model.build_model(
            driver_data['TyreLife'].values,
            driver_data['CompoundIdx'].values,
            driver_data['LapTime_Adjusted'].values,
            len(compounds)
    ):
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=1,  # FORCE 1 CHAIN
            cores=1,  # FORCE 1 CORE
            progressbar=True
        )
    return model, trace, compounds, compound_map


# --- SIDEBAR ---
st.sidebar.title("🏎️ Pit Wall AI")
driver = st.sidebar.selectbox("Select Driver", ["VER", "LEC", "HAM", "ALO"])
pit_loss = st.sidebar.slider("Est. Pit Loss (sec)", 18.0, 26.0, 22.0)

# --- MAIN PAGE ---
st.title(f"Probabilistic Strategy: {driver}")

# 1. Load & Train
data = load_data()

with st.spinner(f"Calibrating tire model for {driver}..."):
    # We unpack all 4 values returned by the fixed function
    model, trace, compounds, compound_map = train_model(driver, data)

st.success("Model Calibrated.")

# 2. Visualization of Deg Curves
st.subheader("1. Learned Tire Degradation (Physics)")
fig, ax = plt.subplots(figsize=(10, 4))

# Get the mean degradation rate from the posterior
az_summary = trace.posterior['deg_rate'].mean(dim=["chain", "draw"]).values

# Colors for Soft, Medium, Hard (assuming order, can be adjusted)
colors = ['red', 'yellow', 'white']

for i, rate in enumerate(az_summary):
    if i < len(compounds):
        # Create a dummy line for visualization: 0 to 30 laps
        x = range(30)
        y = [r * i for r in x]
        # Map index back to name (e.g., 0 -> Soft)
        compound_name = compounds[i]
        ax.plot(x, y, label=f"{compound_name} (Deg: {rate:.3f} s/lap)", color=colors[i % 3])

ax.set_ylabel("Pace Loss (s)")
ax.set_xlabel("Laps driven")
ax.legend()
st.pyplot(fig)

# 3. Strategy Simulator
st.subheader("2. Real-time Strategy Simulation")
col1, col2 = st.columns(2)

with col1:
    current_lap = st.number_input("Current Lap", 10, 50, 15)

with col2:
    target_pit_lap = st.slider("Compare Pitting Lap", current_lap + 1, 57, current_lap + 5)

if st.button("Run Monte Carlo Simulation"):
    sim = RaceSimulator(trace, driver, current_lap=current_lap)

    # Simulate Strategy A: Pit NOW (next lap)
    res_now = sim.simulate_strategy(pit_stop_lap=current_lap + 1, pit_loss=pit_loss)

    # Simulate Strategy B: Pit LATER (target lap)
    res_later = sim.simulate_strategy(pit_stop_lap=target_pit_lap, pit_loss=pit_loss)

    # Calculate Delta
    delta = res_later.mean() - res_now.mean()
    prob_winning = (res_now < res_later).mean() * 100

    # Display Results
    st.metric("Expected Gain if Pitting EARLY", f"{delta:.2f} sec", delta_color="normal")
    st.caption(f"Confidence: {prob_winning:.1f}% of simulations favor pitting early.")

    # Plot Distribution
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.kdeplot(res_now, fill=True, label=f"Pit Lap {current_lap + 1} (Undercut)", color='green')
    sns.kdeplot(res_later, fill=True, label=f"Pit Lap {target_pit_lap} (Extend)", color='orange')
    ax2.set_title("Predicted Race Time Distribution")
    ax2.set_xlabel("Total Race Time (s)")
    ax2.legend()
    st.pyplot(fig2)