import numpy as np
import pandas as pd
import arviz as az


class RaceSimulator:
    def __init__(self, trace, driver_code, current_lap, total_laps=57):
        self.trace = trace
        self.driver_code = driver_code
        self.current_lap = current_lap
        self.total_laps = total_laps

        # Extract posteriors from the trace (the model's "brain")
        # We flatten the chains to get a simple list of possibilities
        self.deg_rates = trace.posterior['deg_rate'].values.flatten()
        self.base_paces = trace.posterior['base_pace'].values.flatten()

    def predict_stint(self, start_lap, end_lap, compound_idx, current_tire_age=0):
        """
        Simulates a single stint for ALL posterior samples at once.
        Returns: Matrix of shape (n_samples, n_laps_in_stint)
        """
        n_samples = len(self.base_paces)
        n_laps = end_lap - start_lap

        # 1. Select the specific degradation rate for this compound
        # Note: If our model has shape (samples, 3), we need to handle that.
        # For simplicity in this portfolio v1, we assume the trace is flattened correctly
        # or we just pick the mean if dimensions get complex.
        # Let's do a robust sampling method:

        # Get random indices to sample from the posterior
        idx = np.random.choice(len(self.base_paces), size=n_samples)

        base_pace_sampled = self.base_paces[idx]

        # Handle multidimensional deg_rates (Chains x Draws x Compounds)
        # We assume the user passed a trace where deg_rate has shape (..., n_compounds)
        # We grab the specific column for the chosen compound
        deg_rate_all = self.trace.posterior['deg_rate'].stack(sample=("chain", "draw")).values
        deg_rate_sampled = deg_rate_all[compound_idx, idx]  # Shape: (n_samples,)

        # 2. Vectorized Simulation
        # Create a matrix of laps: rows=samples, cols=lap_number
        laps = np.arange(n_laps) + current_tire_age

        # Formula: Time = Base + (Deg * Age)
        # Result shape: (n_samples, n_laps) using broadcasting
        predicted_times = base_pace_sampled[:, None] + (deg_rate_sampled[:, None] * laps)

        return predicted_times

    def simulate_strategy(self, pit_stop_lap, pit_loss=22.0):
        """
        Simulates a specific strategy: "Stay out until X, then Pit."
        """
        n_samples = len(self.base_paces)

        # --- Stint 1: Current Lap to Pit Stop ---
        # We assume we are on 'Old Tires' (e.g., Softs from start)
        # compound_idx 0 = Soft (example)
        stint1_times = self.predict_stint(
            start_lap=self.current_lap,
            end_lap=pit_stop_lap,
            compound_idx=0,  # Currently on Softs
            current_tire_age=10  # Example: Tires are already 10 laps old
        )

        # --- Pit Stop ---
        # We add a scalar time loss to every simulation
        pit_time = np.full((n_samples, 1), pit_loss)

        # --- Stint 2: Pit Stop to Finish ---
        # We assume we switch to Mediums (compound_idx 1) and age resets to 0
        stint2_times = self.predict_stint(
            start_lap=pit_stop_lap,
            end_lap=self.total_laps,
            compound_idx=1,  # Switch to Mediums
            current_tire_age=0  # Fresh tires
        )

        # --- Aggregate ---
        # Sum times for each sample to get "Total Race Time" distribution
        total_time_stint1 = stint1_times.sum(axis=1)
        total_time_stint2 = stint2_times.sum(axis=1)

        total_race_time = total_time_stint1 + pit_loss + total_time_stint2

        return total_race_time


if __name__ == "__main__":
    # Integration Test
    from models import TireDegradationModel

    # 1. Load Data
    df = pd.read_parquet("../data/processed/2023_Bahrain_laps.parquet")

    # 2. Train Model (Quickly)
    print("Training model for simulation test...")
    model = TireDegradationModel('VER')
    trace, compounds = model.fit(df)

    # 3. Run Simulation
    print("\n🏎️ Simulating Strategy: Pit on Lap 20...")
    sim = RaceSimulator(trace, 'VER', current_lap=15)

    # Compare two strategies
    results_pit_20 = sim.simulate_strategy(pit_stop_lap=20)
    results_pit_25 = sim.simulate_strategy(pit_stop_lap=25)

    # 4. Analysis
    mean_20 = results_pit_20.mean()
    mean_25 = results_pit_25.mean()

    print(f"Strategy A (Pit Lap 20) Mean Time: {mean_20:.2f}s")
    print(f"Strategy B (Pit Lap 25) Mean Time: {mean_25:.2f}s")

    if mean_20 < mean_25:
        print("✅ Recommendation: BOX EARLY (Lap 20)")
    else:
        print("✅ Recommendation: EXTEND STINT (Lap 25)")