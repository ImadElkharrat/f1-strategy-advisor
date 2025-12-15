import fastf1
import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
CACHE_DIR = Path("data/cache")
PROCESSED_DIR = Path("data/processed")

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Enable FastF1 Cache (Critical for speed)
fastf1.Cache.enable_cache(CACHE_DIR)


class RaceDataLoader:
    def __init__(self, year, gp_identifier):
        """
        :param year: e.g., 2023
        :param gp_identifier: e.g., 'Bahrain', 'Monaco', or Round Number
        """
        self.year = year
        self.gp_identifier = gp_identifier
        self.session = None
        self.laps = None

    def load_session(self):
        """Downloads and parses the race session."""
        print(f"⬇️ Loading {self.year} {self.gp_identifier}...")
        self.session = fastf1.get_session(self.year, self.gp_identifier, 'R')
        self.session.load(telemetry=False, weather=True)  # Telemetry off for now to save time
        self.laps = self.session.laps
        print("✅ Session Loaded.")

    def process_data(self):
        """
        Cleans data and adds physics-based features.
        """
        if self.laps is None:
            raise ValueError("Session not loaded. Call load_session() first.")

        print("⚙️ Processing Laps...")

        # 1. Select relevant columns
        cols = [
            'Driver', 'Team', 'LapNumber', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
            'Compound', 'TyreLife', 'Stint', 'PitInTime', 'PitOutTime', 'TrackStatus',
            'IsAccurate'
        ]
        df = self.laps[cols].copy()

        # 2. Convert Timedeltas to Seconds
        time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        for col in time_cols:
            df[col] = df[col].dt.total_seconds()

        # 3. Fuel Mass Proxy
        # Start ~110kg, burn ~1.7kg/lap
        df['FuelLoad'] = 110 - (df['LapNumber'] * 1.7)
        df['FuelLoad'] = df['FuelLoad'].clip(lower=0)

        # --- NEW: PHYSICS ENGINE ---
        # Calculate Fuel-Corrected Lap Time
        # We subtract the time lost due to fuel weight to isolate tire performance.
        # Penalty: ~0.035 seconds per kg of fuel
        FUEL_PENALTY_PER_KG = 0.035
        df['FuelCorrection'] = df['FuelLoad'] * FUEL_PENALTY_PER_KG
        df['LapTime_Adjusted'] = df['LapTime'] - df['FuelCorrection']
        # ---------------------------

        # 4. Filter for Valid Racing Laps
        df_clean = df[
            (df['TrackStatus'] == '1') &
            (df['IsAccurate'] == True) &
            (df['LapTime'].notna())
            ].copy()

        # 5. Enrich with Metadata
        df_clean['Year'] = self.year
        df_clean['Race'] = self.gp_identifier

        df_clean = df_clean.drop_duplicates()

        print(f"📊 Extracted {len(df_clean)} clean racing laps.")
        return df_clean

    def save_data(self, df):
        filename = PROCESSED_DIR / f"{self.year}_{self.gp_identifier}_laps.parquet"
        df.to_parquet(filename, index=False)
        print(f"💾 Data saved to {filename}")


if __name__ == "__main__":
    # Test Run: Bahrain 2023 (A high degradation track, perfect for testing)
    loader = RaceDataLoader(2023, 'Bahrain')
    loader.load_session()
    data = loader.process_data()
    loader.save_data(data)

    # Validation
    print("\n--- Data Preview ---")
    print(data[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'FuelLoad']].head())