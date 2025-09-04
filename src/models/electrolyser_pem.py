class ElectrolyserPEM:
    def __init__(self, **kwargs):
        self.params = kwargs

    def calculate_efficiency(self):
        # Placeholder for efficiency calculation
        print("Calculating PEM electrolyser efficiency.")
        return 0.8

    def calculate_hydrogen_production(self, power_kw):
        # Placeholder for hydrogen production calculation
        print(f"Calculating hydrogen production for {power_kw} kW.")
        return power_kw / self.params.get("sec", 50)
