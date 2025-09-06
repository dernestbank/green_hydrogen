import pandas as pd
import numpy as np
from scipy.optimize import minimize

class EnergySystem:
    def __init__(self, capacity_mw=10, battery_power_mw=0, battery_hours=0,
                 solar_capacity_mw=10, wind_capacity_mw=0, scale_factor=0.75, **kwargs):
        self.capacity = capacity_mw  # electrolyser capacity MW
        self.battery_power = battery_power_mw  # MW
        self.battery_hours = battery_hours  # hours
        self.battery_energy = battery_power_mw * battery_hours  # MWh
        self.solar_capacity = solar_capacity_mw  # MW
        self.wind_capacity = wind_capacity_mw  # MW
        self.total_capacity = solar_capacity_mw + wind_capacity_mw  # MW

        # Base costs (example values, should be from config)
        self.solar_capex_base = 1120  # $/KW
        self.wind_capex_base = 1942   # $/KW
        self.battery_capex_base = 400  # $/KWh

        # Scaling parameters
        self.scale_factor = scale_factor
        self.reference_capacity = 1  # MW reference
        self.params = kwargs

        # Electrochemical parameters (can be loaded from config)
        self.electrochemical_params = {
            'battery': {
                'nominal_voltage': 3.6,  # V
                'internal_resistance': 0.01,  # ohm
                'nominal_capacity': 50,  # Ah per module
                'max_dod': 0.8,
                'capacity_fade_rate': 0.0005,  # per cycle
                'power_fade_rate': 0.0002,  # per cycle
            },
            'electrolyser': {
                'overpotential': 0.4,  # V
                'resistance': 0.1,  # ohm
                'exchange_current_density': 0.01,  # A/cm2
                'activation_energy': 50,  # kJ/mol
            }
        }

        # Degradation tracking
        self.battery_cycles = 0
        self.battery_operating_hours = 0
        self.solar_degradation_rate = 0.005  # 0.5% per year
        self.wind_degradation_rate = 0.005

    def get_power_output(self):
        """Get scaled power output based on capacity"""
        print(f"Calculating power output from energy system with {self.total_capacity} MW capacity.")
        base_output = pd.Series(1, index=range(8760))  # MW
        return base_output * self.total_capacity

    def get_solar_capacity_factor_data(self):
        """Placeholder for solar capacity factor data"""
        return pd.Series(np.random.uniform(0, 1, 8760), index=range(8760))

    def get_wind_capacity_factor_data(self):
        """Placeholder for wind capacity factor data"""
        return pd.Series(np.random.uniform(0, 1, 8760), index=range(8760))

    def get_scaled_capex(self, base_cost, reference_capacity, actual_capacity):
        """Scale CAPEX based on capacity using power law scaling"""
        if actual_capacity <= 0:
            return base_cost
        scaled_cost = (base_cost * reference_capacity *
                      (actual_capacity / reference_capacity) ** self.scale_factor /
                      actual_capacity)
        return scaled_cost

    def get_scaled_solar_capex(self):
        """Get scaled solar CAPEX"""
        base_equip_cost = self.solar_capex_base * self.solar_capacity * 1000  # $ equip
        scaled_equip_cost = self.get_scaled_capex(base_equip_cost, self.reference_capacity, self.solar_capacity)
        return scaled_equip_cost

    def get_scaled_wind_capex(self):
        """Get scaled wind CAPEX"""
        base_equip_cost = self.wind_capex_base * self.wind_capacity * 1000  # $ equip
        scaled_equip_cost = self.get_scaled_capex(base_equip_cost, self.reference_capacity, self.wind_capacity)
        return scaled_equip_cost

    def get_scaled_battery_capex(self):
        """Get scaled battery CAPEX"""
        return self.get_scaled_capex(self.battery_capex_base * self.battery_energy * 1000,
                                   self.reference_capacity, self.battery_energy)

    def get_total_system_capex(self):
        """Get total scaled CAPEX for the energy system"""
        solar_capex = self.get_scaled_solar_capex()
        wind_capex = self.get_scaled_wind_capex()
        battery_capex = self.get_scaled_battery_capex()
        return solar_capex + wind_capex + battery_capex

    def optimize_battery_capacity(self, daily_load_profile):
        """
        Optimize battery power and duration to minimize costs while meeting load.

        Args:
            daily_load_profile: pandas Series of hourly load in MW

        Returns:
            dict with optimal battery_power and battery_hours, and cost
        """
        def cost_function(battery_params):
            power, hours = battery_params
            battery_cost = self.get_scaled_capex(self.battery_capex_base * power * hours * 1000, 1, power * hours)
            return battery_cost

        # Constraints: power between 1-50 MW, hours 1-8
        bounds = [(1, 50), (1, 8)]
        initial_guess = [10, 4]

        # Note: This is a simplified optimization ignoring detailed energy balance
        result = minimize(cost_function, initial_guess, bounds=bounds, method='L-BFGS-B')

        return {
            'battery_power': result.x[0],
            'battery_hours': result.x[1],
            'optimal_cost': result.fun,
            'success': result.success
        }

    def optimize_battery_operation(self, generation_profile, load_profile):
       """
       Heuristic optimization for battery charge/discharge operations with electrochemical parameters.
   
       Args:
           generation_profile: pandas Series of hourly generation MW
           load_profile: pandas Series of hourly load MW
   
       Returns:
           dict with charge/discharge schedule
       """
       excess_gen = generation_profile - load_profile
       battery_state = 0.5  # start at half charge
       min_soc = 0.1
       max_soc = 1.0
       nominal_voltage = self.electrochemical_params['battery']['nominal_voltage']
   
       charge_schedule = []
       discharge_schedule = []
   
       for hour, excess in excess_gen.items():
           if excess > 0 and battery_state < max_soc:
               # Charge
               adjusted_energy = self.get_adjusted_battery_energy()
               potential_charge = min(excess, self.battery_power, (max_soc - battery_state) * adjusted_energy)
               # Calculate current (A) from power (MW) and voltage
               current_charge = potential_charge * 1e6 / nominal_voltage  # A
               # Get electrochemical efficiency
               charge_eff = self.get_battery_electrochemical_efficiency(current_charge, battery_state)
               # Apply efficiency to actual charge
               actual_charge = potential_charge * charge_eff
               battery_state += actual_charge / adjusted_energy
               charge_schedule.append((hour, actual_charge))
           elif excess < 0 and battery_state > min_soc:
               # Discharge
               adjusted_energy = self.get_adjusted_battery_energy()
               potential_discharge = min(-excess, self.battery_power, (battery_state - min_soc) * adjusted_energy)
               # Calculate current (A)
               current_discharge = potential_discharge * 1e6 / nominal_voltage  # A (negative)
               # Get efficiency (discharge efficiency)
               discharge_eff = self.get_battery_electrochemical_efficiency(-current_discharge, battery_state)
               # Apply efficiency
               actual_energy = potential_discharge * discharge_eff
               battery_state -= actual_energy / adjusted_energy
               discharge_schedule.append((hour, actual_energy))
   
       return {
           'charge_schedule': charge_schedule,
           'discharge_schedule': discharge_schedule,
           'final_soc': battery_state
       }
   
    def get_battery_electrochemical_efficiency(self, current_A, soc, temperature=25):
       """
       Calculate battery efficiency based on electrochemical parameters.
   
       Args:
           current_A: Current in A (positive for charge, negative for discharge)
           soc: State of charge (0-1)
           temperature: Temperature in C
   
       Returns:
           Efficiency factor
       """
       params = self.electrochemical_params['battery']
       v_nom = params['nominal_voltage']
       r_int = params['internal_resistance']
   
       # Ohmic voltage drop
       v_drop = abs(current_A) * r_int
   
       # Simple activation overpotential (simplified)
       overpot = 0.02 * abs(current_A) + 0.1 * (1 - soc)  # higher overpot at low SOC
   
       # Total voltage drop
       total_drop = v_drop + overpot
   
       # Efficiency (round-trip for discharge)
       if current_A < 0:  # discharge
           efficiency = (v_nom - total_drop) / v_nom
       else:  # charge
           efficiency = v_nom / (v_nom + total_drop)
   
       return max(0.4, min(0.95, efficiency))  # clamp between 40% and 95%
   
    def get_electrolyser_electrochemical_voltage(self, current_density, temperature=25):
       """
       Calculate electrolyser voltage based on electrochemical parameters.
   
       Args:
           current_density: Current density in A/cm2
           temperature: Temperature in C
   
       Returns:
           Voltage in V
       """
       params = self.electrochemical_params['electrolyser']
   
       # Thermodynamics voltage (simplified)
       v_therm = 1.23  # at STP
   
       # Activation overpotential, Tafel equation approx
       over_act = params['overpotential'] * (current_density / params['exchange_current_density']) ** 0.5
   
       # Ohmic overpotential
       over_ohm = current_density * params['resistance']
   
       # Concentration overpotential (simplified)
       over_conc = 0.01 * current_density
   
       voltage = v_therm + over_act + over_ohm + over_conc
   
       return voltage
   
    def update_electrochemical_parameters(self, source_config=None):
       """
       Update electrochemical parameters from config or external source.
   
       Args:
           source_config: Dict with electrochemical parameters
       """
       if source_config:
           self.electrochemical_params.update(source_config)

    def update_capacity(self, new_capacity_mw, new_solar_mw=None, new_wind_mw=None,
                       new_battery_power=None, new_battery_hours=None):
        """Update system capacities and recalculate"""
        self.capacity = new_capacity_mw
        if new_solar_mw is not None:
            self.solar_capacity = new_solar_mw
        if new_wind_mw is not None:
            self.wind_capacity = new_wind_mw
        if new_battery_power is not None:
            self.battery_power = new_battery_power
        if new_battery_hours is not None:
            self.battery_hours = new_battery_hours
            self.battery_energy = new_battery_power * new_battery_hours
        self.total_capacity = self.solar_capacity + self.wind_capacity

    def get_battery_degradation_factor(self):
        """
        Calculate degradation factor for battery capacity and power.

        Returns:
            float: Degradation factor (0-1)
        """
        params = self.electrochemical_params['battery']
        capacity_fade = 1 - (params['capacity_fade_rate'] * self.battery_cycles)
        power_fade = 1 - (params['power_fade_rate'] * self.battery_cycles)
        return min(max(capacity_fade, 0.5), 1.0)  # clamp to reasonable bounds

    def get_adjusted_battery_energy(self):
        """
        Get battery energy capacity adjusted for degradation.

        Returns:
            float: Adjusted battery energy (MWh)
        """
        return self.battery_energy * self.get_battery_degradation_factor()

    def get_solar_degradation_factor(self, years):
        """
        Calculate solar panel degradation factor over time.

        Args:
            years: Number of years in operation

        Returns:
            float: Degradation factor (0-1)
        """
        return (1 - self.solar_degradation_rate) ** years

    def get_wind_degradation_factor(self, years):
        """
        Calculate wind turbine degradation factor over time.

        Args:
            years: Number of years in operation

        Returns:
            float: Degradation factor (0-1)
        """
        return (1 - self.wind_degradation_rate) ** years

    def update_battery_degradation(self, delta_cycles=1, delta_hours=1):
        """
        Update battery degradation state.

        Args:
            delta_cycles: Number of charge/discharge cycles to add
            delta_hours: Number of operating hours to add
        """
        self.battery_cycles += delta_cycles
        self.battery_operating_hours += delta_hours
