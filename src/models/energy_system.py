import pandas as pd

class EnergySystem:
    def __init__(self, **kwargs):
        self.params = kwargs

    def get_power_output(self):
        # Placeholder for power output calculation
        print("Calculating power output from the energy system.")
        # Example: return a pandas Series with hourly power output for a year
        return pd.Series(1, index=range(8760))
