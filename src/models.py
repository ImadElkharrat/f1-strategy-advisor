import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt


class TireDegradationModel:
    def __init__(self, driver_code):
        self.driver_code = driver_code
        self.model = None
        self.trace = None

    def prepare_data(self, df):
        """
        Filters data for the specific driver and encodes categorical variables.
        """
        # Filter for driver and valid racing laps
        driver_data = df[(df['Driver'] == self.driver_code) &
                         (df['LapTime_Adjusted'].notna())].copy()

        # Create integer encoding for compounds: 0=Soft, 1=Medium, 2=Hard
        compounds = driver_data['Compound'].unique()
        compound_map = {c: i for i, c in enumerate(compounds)}

        driver_data['CompoundIdx'] = driver_data['Compound'].map(compound_map)

        return driver_data, compounds, compound_map

    def build_model(self, X_age, X_compound_idx, y_laptime, n_compounds):
        """
        Defines the PyMC probabilistic graph.
        Formula: LapTime = BasePace + (DegRate[Compound] * Age) + Noise
        """
        print("    Building PyMC Graph...")
        with pm.Model() as self.model:
            # 1. Priors
            # Base Pace: The theoretical lap time on fresh tires
            base_pace = pm.Normal("base_pace", mu=np.mean(y_laptime), sigma=1.0)

            # Degradation Rate: specific to each compound (Soft > Hard)
            # We use HalfNormal because degradation is always positive (tires get slower)
            deg_rate = pm.HalfNormal("deg_rate", sigma=0.1, shape=n_compounds)

            # Model Error (Noise)
            sigma = pm.HalfNormal("sigma", sigma=0.5)

            # 2. Expected Value Calculation
            # We select the specific deg_rate for the compound used on that lap
            deg_effect = deg_rate[X_compound_idx] * X_age

            mu = base_pace + deg_effect

            # 3. Likelihood
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_laptime)

        return self.model

    def fit(self, df):
        """Train the model using MCMC sampling."""
        data, compounds, self.compound_map = self.prepare_data(df)

        X_age = data['TyreLife'].values
        X_compound = data['CompoundIdx'].values

        # IMPORTANT: We train on Fuel-Corrected times
        if 'LapTime_Adjusted' not in data.columns:
            raise ValueError("Data must have 'LapTime_Adjusted'. Run physics pipeline first.")

        y = data['LapTime_Adjusted'].values

        print(f"🧠 Training Bayesian Model for {self.driver_code} on {len(y)} laps...")

        self.build_model(X_age, X_compound, y, len(compounds))

        with self.model:
            # Draw 1000 posterior samples (this explores the probability space)
            # tune=1000 helps the sampler 'warm up'
            self.trace = pm.sample(1000, tune=1000, chains=2, progressbar=True)

        return self.trace, compounds

    def plot_posterior(self, compounds):
        """Visualizes what the model learned about tire wear."""
        if self.trace is None:
            print("Model not trained yet.")
            return

        # Plot the degradation rates for each compound
        ax = az.plot_forest(self.trace, var_names=["deg_rate"], combined=True)
        plt.title(f"Estimated Degradation Rates (s/lap) - {self.driver_code}")

        # Label the Y-axis with compound names (0, 1, 2 -> Soft, Medium, Hard)
        # Note: Depending on Arviz version, y-labels might need manual adjustment.
        print(f"Compound Map: {self.compound_map}")
        plt.show()


if __name__ == "__main__":
    # Test Run
    try:
        processed_path = "../data/processed/2023_Bahrain_laps.parquet"
        df = pd.read_parquet(processed_path)

        # Initialize and Fit
        model = TireDegradationModel(driver_code='VER')  # Max Verstappen
        trace, compounds = model.fit(df)
        model.plot_posterior(compounds)
        print("✅ Model Training Complete.")

    except FileNotFoundError:
        print("❌ Error: Processed data not found. Run 'main.py' first.")
    except Exception as e:
        print(f"❌ Error: {e}")