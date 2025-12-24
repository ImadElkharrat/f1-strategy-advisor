# Probabilistic F1 Strategy Advisor 🏎️

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyMC](https://img.shields.io/badge/PyMC-Bayesian%20Inference-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

A probabilistic decision-support system for Formula 1 race strategy. This project uses **Hierarchical Bayesian Modeling** to learn tire degradation parameters from public telemetry data and a **Monte Carlo Simulation** engine to optimize pit-stop windows under uncertainty.

## 🎯 Project Goal
To replace deterministic strategy predictions (e.g., "Pit on lap 20") with probabilistic confidence intervals (e.g., "Pitting on lap 20 gives an 85% probability of beating the undercut").

## 🏗️ Architecture
1.  **Data Pipeline:** Ingests raw lap data via `FastF1`, corrects for fuel-mass physics (~0.035s/kg), and filters for clean air laps.
2.  **Core Model (PyMC):** A probabilistic generative model:
    $$\text{LapTime} \sim \mathcal{N}(\mu, \sigma)$$
    $$\mu = \text{BasePace} + (\text{DegRate} \times \text{TireAge})$$
3.  **Simulation Engine:** Projects race outcomes by sampling 1,000+ futures from the Bayesian posterior trace.
4.  **Dashboard:** Interactive Streamlit UI for real-time "What-If" analysis.

## Features
### ⚔️ Duel Mode: The Undercut Estimator
Beyond single-car optimization, the system simulates head-to-head duels.
* **Scenario:** User selects a "Hero" and a "Rival" and defines the current time gap.
* **Simulation:** The engine runs 2,000 paired simulations, calculating the probability of an "Undercut" (overtaking via pit strategy) vs. the risk of being caught later on older tires.
* **Output:** A probability distribution of the final gap, helping engineers quantify the risk/reward of pitting early.

## 🚀 Quick Start

### 1. Installation
Recommended to use Conda due to PyMC's C++ dependencies.
```bash
conda create -c conda-forge -n f1-strategy "pymc>=5" pytensor numpy pandas seaborn streamlit
conda activate f1-strategy
pip install fastf1