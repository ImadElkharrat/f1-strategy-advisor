import numpy as np
import pandas as pd
import arviz as az


class RaceSimulator:
    def __init__(self, trace, driver_code, current_lap, total_laps=57):
        self.trace = trace
        self.driver_code = driver_code
        self.current_lap = current_lap
        self.total_laps = total_laps

        # Extract posteriors
        self.deg_rates = trace.posterior['deg_rate'].stack(sample=("chain", "draw")).values
        self.base_paces = trace.posterior['base_pace'].stack(sample=("chain", "draw")).values

    def predict_stint(self, start_lap, end_lap, compound_idx, current_tire_age=0):
        n_samples = self.base_paces.shape[0]
        n_laps = end_lap - start_lap
        if n_laps <= 0: return np.zeros(n_samples)  # Handle 0 length stints

        # Sample indices
        idx = np.random.choice(n_samples, size=n_samples)

        base_sampled = self.base_paces[idx]  # (n_samples,)
        deg_sampled = self.deg_rates[compound_idx, idx]  # (n_samples,)

        # Vectorized Lap Generation
        laps = np.arange(n_laps) + current_tire_age
        # Shape: (n_samples, n_laps)
        lap_times = base_sampled[:, None] + (deg_sampled[:, None] * laps)

        return lap_times.sum(axis=1)  # Return total time for the stint

    def simulate_strategy(self, pit_stop_lap, pit_loss=22.0, start_compound=0, end_compound=1):
        """
        Returns Total Remaining Race Time distribution
        """
        # Stint 1: Now -> Pit
        t1 = self.predict_stint(self.current_lap, pit_stop_lap, start_compound, current_tire_age=10)

        # Stint 2: Pit -> Finish
        t2 = self.predict_stint(pit_stop_lap, self.total_laps, end_compound, current_tire_age=0)

        return t1 + pit_loss + t2


class DuelSimulator:
    def __init__(self, sim_hero, sim_rival):
        self.hero = sim_hero
        self.rival = sim_rival

    def simulate_gap(self, hero_pit, rival_pit, current_gap, pit_loss=22.0):
        """
        Simulates the race time difference.
        Positive Result = Hero is slower (Behind)
        Negative Result = Hero is faster (Ahead)
        """
        hero_time = self.hero.simulate_strategy(hero_pit, pit_loss)
        rival_time = self.rival.simulate_strategy(rival_pit, pit_loss)

        # Add the current physical gap on track
        # If Hero is 5s behind, their total time is effectively 5s longer relative to track position
        final_delta = (hero_time + current_gap) - rival_time

        return final_delta