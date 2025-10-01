"""Hydrogen Model
Defines a class HydrogenModel which is the technoecomic analysis model/engine for green hydrogen.
ref: the HySupply Cost Tool v1.3 Excel model.
"""

import os
import pandas as pd
import numpy as np
import yaml
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class HydrogenModel:
    """
    A class that defines a techno-economic model for green hydrogen production.

    Attributes
    ----------
    solar_df : pandas dataframe
        a dataframe containing hourly solar traces for 1 year
    wind_df : pandas dataframe
        a dataframe containing hourly wind traces for 1 year
    location : str
        the ID for the location to be modelled. Needs to correspond to a column in solar_df and wind_df
        (default "US.CA")
    elecType : str
        the electrolyser type - either "AE" for Alkaline Electrolyte or "PEM" for Polymer Electrolyte Membrane
        (default "PEM")
    elecCapacity : int
        the rated capacity of the electrolyser in MW (default 10)
    solarCapacity : float
        the rated capacity of the solar farm in MW (default 10.0)
    windCapacity : float
        the rated capacity of the wind farm in MW (default 0.0)
    batteryPower : float
        the rated power capacity of the battery. Set to 0 to remove battery from the model (default 0)
    batteryHours : int
        the time period that the battery can discharge at full power, must be 0, 1, 2, 4 or 8 (default 0)
    spotPrice : float
        Price that excess generation can be sold to the grid for, in $/MWh (default 0.0)
    ppaPrice : float
        Price that electricity can be purchased for, in $/MWh. Setting this value greater than zero results in all
        electricity being bought from the grid and hence CAPEX and OPEX for the generation are ignored (default 0.0)

    Methods
    -------
    calculate_electrolyser_output()
        returns a dictionary with summarised output values for the model including capacity factors, energy input and
        hydrogen production
    calculate_costs(specific_consumtion_type='fixed')
        returns the levelised cost of hydrogen for the model for either "fixed" or "variable" values for the
        specific energy consumption vs electrolyser load
    make_duration_curve(generator=True, electrolyser=False)
        creates annual duration curves for the generator and/or the electrolyser
    """

    def __init__(self, config_manager=None, solardata=None, winddata=None, config_path='config/config.yaml',
                 location='US.CA', elec_type='PEM', elec_capacity=10, solar_capacity=10.0, wind_capacity=0.0,
                 battery_power=0, battery_hours=0, spot_price=0.0, ppa_price=0.0):
        """
        Initialize the hydrogen model with the specified parameters.
        
        Args:
            config_manager: Optional ConfigManager instance to use
            solardata: Optional DataFrame with solar data
            winddata: Optional DataFrame with wind data
            config_path: Path to configuration file
            location: Location ID corresponding to columns in solar/wind data
            elec_type: Electrolyser type ('AE' or 'PEM')
            elec_capacity: Electrolyser capacity in MW
            solar_capacity: Solar farm capacity in MW
            wind_capacity: Wind farm capacity in MW
            battery_power: Battery power capacity in MW
            battery_hours: Battery discharge duration (0, 1, 2, 4, or 8)
            spot_price: Price for selling excess generation in $/MWh
            ppa_price: Price for purchasing electricity in $/MWh
        """
        if solardata is not None:
            self.solar_df = solardata
        elif solar_capacity > 0:
            solarfile = str(Path('Data/solar-traces.csv'))
            if not os.path.exists(solarfile):
                raise FileNotFoundError(f"Solar data file not found at {solarfile}")
            self.solar_df = pd.read_csv(solarfile, header=[0], skiprows=[1], index_col=0)
        else:
            self.solar_df = pd.DataFrame()
            
        if winddata is not None:
            self.wind_df = winddata
        elif wind_capacity > 0:
            windfile = str(Path('Data/wind-traces.csv'))
            if not os.path.exists(windfile):
                raise FileNotFoundError(f"Wind data file not found at {windfile}")
            self.wind_df = pd.read_csv(windfile, header=[0], skiprows=[1], index_col=0)
        else:
            self.wind_df = pd.DataFrame()
            
        # Load configuration from file or ConfigManager
        if config_manager:
            self.config_dict = config_manager.load_config()
            logger.info("Loaded configuration from ConfigManager")
        else:
            try:
                config_path = Path(config_path)
                with open(config_path, 'r') as config_file:
                    self.config_dict = yaml.safe_load(config_file)
                    logger.info(f"Loaded configuration from {config_path}")
            except (FileNotFoundError, yaml.YAMLError) as e:
                logger.error(f"Error loading configuration file {config_path}: {e}")
                raise
        
        if location in self.solar_df.columns or location in self.wind_df.columns:
            self.location = location
        else:
            raise KeyError(f"Location '{location}' not found in solar or wind file header.")
            
        if elec_type not in ['AE', 'PEM']:
            raise ValueError("elec_type must be 'AE' or 'PEM'")
        else:
            self.elecType = elec_type
            
        self.elecCapacity = elec_capacity
        self.solarCapacity = solar_capacity
        self.windCapacity = wind_capacity
        self.batteryPower = battery_power
        self.batteryHours = battery_hours
        self.spotPrice = spot_price
        self.ppaPrice = ppa_price

        # Initialize parameters from configuration
        try:
            self._initialize_parameters()
        except KeyError as e:
            logger.error(f"Missing key in configuration file: {e}")
            raise KeyError(f"Error: Entry {e} not found in config file.")

        # Empty variables for storing calculation results
        self.operating_outputs = {}
        self.LCH2 = {}

    def _initialize_parameters(self):
        """Initialize model parameters from configuration."""
        # Electrolyser parameters - convert type to lowercase for config lookup
        elec_type_key = self.elecType.lower()
        
        self.elecMaxLoad = self.config_dict.get('elec_max_load', 100) / 100
        self.elecOverload = self.config_dict[elec_type_key]['elec_overload'] / 100
        self.elecOverloadRecharge = self.config_dict[elec_type_key]['elec_overload_recharge']
        self.elecReferenceCap = self.config_dict.get('elec_reference_capacity', 10)
        self.elecCostReduction = self.config_dict.get('elec_cost_reduction', 1.0)
        self.elecMinLoad = self.config_dict[elec_type_key]['elec_min_load'] / 100
        self.elecEff = self.config_dict.get('elec_efficiency', 83) / 100
        self.specCons = self.config_dict[elec_type_key]['spec_consumption']  # kWh/Nm3
        self.H2VoltoMass = self.config_dict.get('h2_vol_to_mass', 0.089)  # kg/m3
        self.MWtokW = 1000  # kW/MW
        self.hydOutput = self.H2VoltoMass * self.MWtokW * self.elecEff  # kg.kWh/m3.MWh
        self.hoursPerYear = 8760
        self.kgtoTonne = 1/1000
        self.stackLifetime = self.config_dict[elec_type_key]['stack_lifetime']  # hours before replacement
        self.waterNeeds = self.config_dict[elec_type_key]['water_needs']  # kL/ton

        # Generation parameters
        self.genCapacity = self.solarCapacity + self.windCapacity
        if self.genCapacity > 0:
            self.solarRatio = self.solarCapacity / self.genCapacity
            self.windRatio = self.windCapacity / self.genCapacity
        else:
            self.solarRatio = 0
            self.windRatio = 0

        # Battery parameters
        self.batteryEnergy = self.batteryPower * self.batteryHours
        self.batteryEfficiency = self.config_dict.get('battery_efficiency', 85) / 100
        self.battMin = self.config_dict.get('battery_min', 0) / 100
        self.battLife = self.config_dict.get('battery_lifetime', 10)

        # Cost parameters - solar
        self.solarCapex = self.config_dict.get('solar_capex', 1120) * self.MWtokW  # $/MW
        self.solarCapex = self._scale_capex(
            self.solarCapex,
            self.solarCapacity,
            self.config_dict.get('powerplant_reference_capacity', 1),
            self.config_dict.get('powerplant_cost_reduction', 1.0)
        )
        self.solarCapex = self._get_capex(
            self.solarCapex,
            self.config_dict.get('powerplant_equip', 1.0),
            self.config_dict.get('powerplant_install', 0.0),
            self.config_dict.get('powerplant_land', 0.0)
        )
        self.solarOpex = self.config_dict.get('solar_opex', 16990)  # $/MW

        # Cost parameters - wind
        self.windCapex = self.config_dict.get('wind_capex', 1942) * self.MWtokW  # $/MW
        self.windCapex = self._scale_capex(
            self.windCapex,
            self.windCapacity,
            self.config_dict.get('powerplant_reference_capacity', 1),
            self.config_dict.get('powerplant_cost_reduction', 1.0)
        )
        self.windCapex = self._get_capex(
            self.windCapex,
            self.config_dict.get('powerplant_equip', 1.0),
            self.config_dict.get('powerplant_install', 0.0),
            self.config_dict.get('powerplant_land', 0.0)
        )
        self.windOpex = self.config_dict.get('wind_opex', 25000)  # $/MW

        # Battery costs
        self.batteryCapex = self.config_dict.get('battery_capex', {0: 0, 1: 827, 2: 542, 4: 446, 8: 421})  # $/kWh
        # Convert to $/MWh
        self.batteryCapex.update({n: self.batteryCapex[n] * self.MWtokW for n in self.batteryCapex.keys()})
        self.batteryOpex = self.config_dict.get('battery_opex', {0: 0, 1: 4833, 2: 9717, 4: 19239, 8: 39314})  # $/MW
        self.battReplacement = self.config_dict.get('battery_replacement', 100) / 100 * self.batteryCapex[self.batteryHours]

        # Electrolyser costs
        electrolyserCapexUnscaled = self.config_dict[elec_type_key]['electrolyser_capex'] * self.MWtokW  # $/MW
        self.electrolyserCapex = self._scale_capex(
            electrolyserCapexUnscaled,
            self.elecCapacity,
            self.elecReferenceCap,
            self.elecCostReduction
        )
        self.electrolyserOandM = self.config_dict[elec_type_key]['electrolyser_om'] / 100 * self.electrolyserCapex  # $/MW
        self.electrolyserStackCost = self.config_dict.get('electrolyser_stack_cost', 40) / 100 * self.electrolyserCapex  # $/MW
        self.electrolyserCapex = self._get_capex(
            self.electrolyserCapex,
            self.config_dict.get('elec_equip', 1.0),
            self.config_dict.get('elec_install', 0.0),
            self.config_dict.get('elec_land', 0.0)
        )

        # Other costs
        self.waterCost = self.config_dict.get('water_cost', 5)  # $/kL
        self.discountRate = self.config_dict.get('discount_rate', 4) / 100  # percentage as decimal
        self.projectLife = self.config_dict.get('project_life', 20)

    def __str__(self):
        """Return string representation of the model."""
        return (f"This model has inputs:\n"
                f"Location = {self.location}\n"
                f"Electrolyser Capacity = {self.elecCapacity}\n"
                f"Solar Capacity = {self.solarCapacity}\n"
                f"Wind Capacity = {self.windCapacity}\n"
                f"Battery Power = {self.batteryPower}\n"
                f"Battery duration = {self.batteryHours}")

    def calculate_electrolyser_output(self):
        """
        Calculate the hourly operation of the electrolyser and return a summary of the results.

        Returns
        -------
        operating_outputs : dict
            Dictionary with keys 'Generator Capacity Factor', 'Time Electrolyser is at its Rated Capacity',
            'Total Time Electrolyser is Operating', 'Achieved Electrolyser Capacity Factor',
            'Energy in to Electrolyser [MWh/yr]', 'Surplus Energy [MWh/yr]',
            'Hydrogen Output for Fixed Operation [t/yr]', 'Hydrogen Output for Variable Operation [t/yr]'
        """
        working_df = self._calculate_hourly_operation()
        # Generate results table to mirror the one in the excel tool
        operating_outputs = self._get_tabulated_outputs(working_df)
        return operating_outputs

    def calculate_costs(self, specific_consumption_type='fixed'):
        """
        Calculate the levelised cost of hydrogen production for the model.

        Parameters
        ----------
        specific_consumption_type : str, optional
            Method for calculating electrolyser specific consumption: "fixed" or "variable"

        Returns
        -------
        lcoh : float
            The levelised cost of hydrogen in $/kg rounded to two decimal places
        """
        if not self.operating_outputs:
            self.calculate_electrolyser_output()

        gen_capex = self.solarCapex * self.solarCapacity + self.windCapex * self.windCapacity
        gen_opex = self.solarOpex * self.solarCapacity + self.windOpex * self.windCapacity

        if specific_consumption_type == "variable":
            annual_hydrogen = self.operating_outputs["Hydrogen Output for Variable Operation [t/yr]"]
        elif specific_consumption_type == "fixed":
            annual_hydrogen = self.operating_outputs["Hydrogen Output for Fixed Operation [t/yr]"]
        else:
            raise ValueError("Specific consumption type not valid, please select either 'variable' or 'fixed'")

        # Calculate the annual cash flows
        cash_flow_df = pd.DataFrame(
            index=range(self.projectLife + 1),
            columns=['Year', 'Gen_CAPEX', 'Elec_CAPEX', 'Gen_OPEX', 'Elec_OandM',
                    'Power_cost', 'Stack_replacement', 'Water_cost', 'Battery_cost', 'Total']
        )
        cash_flow_df['Year'] = range(self.projectLife + 1)

        if self.ppaPrice > 0:
            cash_flow_df.loc[1:, 'Power_cost'] = self.operating_outputs["Energy in to Electrolyser [MWh/yr]"] * self.ppaPrice
        else:
            cash_flow_df.at[0, 'Gen_CAPEX'] = gen_capex
            cash_flow_df.loc[1:, 'Gen_OPEX'] = gen_opex
            cash_flow_df.loc[1:, 'Power_cost'] = -1 * self.operating_outputs["Surplus Energy [MWh/yr]"] * self.spotPrice

        cash_flow_df.at[0, 'Elec_CAPEX'] = self.electrolyserCapex * self.elecCapacity
        cash_flow_df.loc[1:, 'Elec_OandM'] = self.electrolyserOandM * self.elecCapacity
        stack_years = self._find_stack_replacement_years()
        cash_flow_df.loc[stack_years, 'Stack_replacement'] = self.electrolyserStackCost * self.elecCapacity
        cash_flow_df.loc[1:, 'Water_cost'] = annual_hydrogen * self.waterNeeds * self.waterCost
        cash_flow_df.at[0, 'Battery_cost'] = self.batteryCapex[self.batteryHours] * self.batteryEnergy
        cash_flow_df.loc[1:, 'Battery_cost'] = self.batteryOpex[self.batteryHours] * self.batteryPower
        cash_flow_df.at[10, 'Battery_cost'] += self.battReplacement * self.batteryEnergy
        cash_flow_df['Total'] = cash_flow_df.sum(axis=1)

        # Calculate the annual discounted cash flows for hydrogen and total costs
        discounted_flow = pd.DataFrame(
            index=range(self.projectLife + 1),
            columns=['Year', 'Hydrogen_kg', 'Hydrogen_kg_Discounted', 'Total']
        )
        discounted_flow['Year'] = range(self.projectLife + 1)
        discounted_flow.loc[1:, 'Hydrogen_kg'] = annual_hydrogen / self.kgtoTonne
        discounted_flow['Hydrogen_kg_Discounted'] = discounted_flow['Hydrogen_kg'] * \
            (1 / (1 + self.discountRate)) ** discounted_flow['Year']
        discounted_flow['Total'] = cash_flow_df['Total'] * (1 / (1 + self.discountRate)) ** discounted_flow['Year']

        # Calculate the LCOH
        lcoh = discounted_flow['Total'].sum() / discounted_flow['Hydrogen_kg_Discounted'].sum()
        self.LCH2 = round(lcoh, 2)
        return round(lcoh, 2)

    def _calculate_hourly_operation(self):
        """
        Calculate hourly operation profile for the system.
        
        Returns:
            DataFrame with hourly operation data
        """
        oversize = self.genCapacity / self.elecCapacity if self.elecCapacity > 0 else 0
        working_df = pd.DataFrame()
        
        if self.solarRatio == 1:
            working_df['Generator_CF'] = self.solar_df[self.location]
        elif self.windRatio == 1:
            working_df['Generator_CF'] = self.wind_df[self.location]
        else:
            working_df['Generator_CF'] = self.solar_df[self.location] * self.solarRatio + \
                                        self.wind_df[self.location] * self.windRatio

        has_excess_gen = working_df['Generator_CF'] * oversize > self.elecMaxLoad
        has_insufficient_gen = working_df['Generator_CF'] * oversize < self.elecMinLoad
        working_df['Electrolyser_CF'] = np.where(
            has_excess_gen,
            self.elecMaxLoad,
            np.where(
                has_insufficient_gen,
                0,
                working_df['Generator_CF'] * oversize
            )
        )
        
        if self.elecOverload > self.elecMaxLoad and self.elecOverloadRecharge > 0:
            working_df['Electrolyser_CF'] = self._overloading_model(working_df, oversize)
            
        if self.batteryEnergy > 0:
            if self.batteryHours not in [1, 2, 4, 8]:
                raise ValueError("Battery storage length not valid. Please enter one of 1, 2, 4 or 8")
            working_df['Electrolyser_CF'] = self._battery_model(oversize, working_df)

        working_df['Hydrogen_prod_fixed'] = working_df['Electrolyser_CF'] * self.hydOutput / self.specCons
        working_df['Hydrogen_prod_variable'] = working_df['Electrolyser_CF'].apply(
            lambda x: x * self.hydOutput / self._electrolyser_output_polynomial(x)
        )

        return working_df

    def _electrolyser_output_polynomial(self, x):
        """
        Calculate the specific energy consumption based on electrolyser load.
        
        Args:
            x: Electrolyser load factor
            
        Returns:
            Specific energy consumption
        """
        return 1.25 * x**2 - 0.4286 * x + self.specCons - 0.85

    def _find_stack_replacement_years(self):
        """
        Find years when electrolyser stack needs replacement.
        
        Returns:
            List of years requiring stack replacement
        """
        if len(self.operating_outputs.keys()) == 0:
            self.calculate_electrolyser_output()
            
        op_hours_per_year = self.operating_outputs["Total Time Electrolyser is Operating"] * self.hoursPerYear
        stack_years = []
        
        for year in range(1, self.projectLife):
            if (math.floor(op_hours_per_year * year / self.stackLifetime) -
                math.floor(op_hours_per_year * (year - 1) / self.stackLifetime)) == 1.0:
                stack_years.append(year)
                
        return stack_years

    def _battery_model(self, oversize, cf_profile_df):
        """
        Calculate electrolyser operation with battery storage.
        
        Args:
            oversize: Generation capacity ratio
            cf_profile_df: DataFrame with capacity factors
            
        Returns:
            Series with updated electrolyser capacity factors
        """
        cf_profile_df = cf_profile_df.reset_index()
        index_name = cf_profile_df.columns[0]
        cf_profile_df['Excess_Generation'] = (cf_profile_df['Generator_CF'] * oversize -
                                            cf_profile_df['Electrolyser_CF']) * self.elecCapacity
        cf_profile_df['Battery_Net_Charge'] = 0.0
        cf_profile_df['Battery_SOC'] = 0.0
        cf_profile_df['Electrolyser_CF_batt'] = 0.0
        batt_losses = (1-(1-self.batteryEfficiency)/2)
        elec_min = self.elecMinLoad * self.elecCapacity
        elec_max = self.elecMaxLoad * self.elecCapacity

        cf_profile_df.at[0, 'Battery_Net_Charge'] = min(
            self.batteryPower,
            cf_profile_df.at[0, 'Excess_Generation'] * batt_losses
        )
        cf_profile_df.at[0, 'Battery_SOC'] = cf_profile_df.at[0, 'Battery_Net_Charge'] / self.batteryEnergy

        for hour in range(1, len(cf_profile_df)):
            batt_soc = cf_profile_df.at[hour - 1, 'Battery_SOC']
            spill = cf_profile_df.at[hour, 'Excess_Generation']
            elec_cons = cf_profile_df.at[hour, 'Electrolyser_CF'] * self.elecCapacity
            batt_discharge_potential = min(
                self.batteryPower,
                (batt_soc - self.battMin) * self.batteryEnergy
            ) * batt_losses
            elec_just_operating = (
                elec_cons > 0 or
                cf_profile_df.at[hour - 1, 'Battery_Net_Charge'] < 0 or
                cf_profile_df.at[hour - 1, 'Electrolyser_CF'] > 0
            )

            # Determine battery behavior based on current conditions
            if elec_cons == 0 and spill + batt_discharge_potential > elec_min and elec_just_operating:
                # Generation insufficient alone but with battery can power electrolyser
                if spill + batt_discharge_potential > elec_max:
                    cf_profile_df.at[hour, 'Battery_Net_Charge'] = -1 * min(
                        self.batteryPower,
                        (elec_max-spill) * 1/batt_losses
                    )
                else:
                    cf_profile_df.at[hour, 'Battery_Net_Charge'] = -1 * batt_discharge_potential * 1/batt_losses
            elif spill > 0 and batt_soc + spill/self.batteryEnergy * batt_losses > 1:
                # Spilled generation enough to completely charge battery
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = min(
                    self.batteryPower,
                    max(self.batteryEnergy * (1.0 - batt_soc), 0.0)
                )
            elif spill > 0:
                # Any other cases with spilled generation
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = min(self.batteryPower, spill * batt_losses)
            elif (elec_cons + batt_discharge_potential < elec_min or
                  (spill == 0 and batt_soc <= self.battMin)):
                # Insufficient combined power or empty battery
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = 0
            elif (spill == 0 and
                  elec_max - elec_cons > (batt_soc - self.battMin) * batt_losses * self.batteryEnergy and
                  elec_just_operating):
                # Electrolyser operating but energy to get to max exceeds stored power
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = -1 * batt_discharge_potential * 1/batt_losses
            elif spill == 0 and elec_just_operating:
                # Stored power enough to power electrolyser at max capacity
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = -1 * min(
                    self.batteryPower,
                    (elec_max - elec_cons) * 1/batt_losses
                )
            elif spill == 0:
                cf_profile_df.at[hour, 'Battery_Net_Charge'] = 0
            else:
                logger.warning("Error: battery configuration not accounted for")

            # Update battery state of charge
            cf_profile_df.at[hour, 'Battery_SOC'] = cf_profile_df.at[hour - 1, 'Battery_SOC'] + \
                cf_profile_df.at[hour, 'Battery_Net_Charge'] / self.batteryEnergy

        # Update electrolyser capacity factor with battery contribution
        cf_profile_df['Electrolyser_CF_batt'] = np.where(
            cf_profile_df['Battery_Net_Charge'] < 0,
            cf_profile_df['Electrolyser_CF'] +
                (-1*cf_profile_df['Battery_Net_Charge'] * batt_losses +
                cf_profile_df['Excess_Generation']) / self.elecCapacity,
            cf_profile_df['Electrolyser_CF']
        )
        
        cf_profile_df.set_index(index_name, inplace=True)
        return cf_profile_df['Electrolyser_CF_batt']

    def _overloading_model(self, cf_profile_df, oversize):
        """
        Calculate electrolyser operation with overloading capability.
        
        Args:
            cf_profile_df: DataFrame with capacity factors
            oversize: Generation capacity ratio
            
        Returns:
            Series with updated electrolyser capacity factors
        """
        can_overload = cf_profile_df['Generator_CF'] * oversize > self.elecMaxLoad

        # Check overloading constraints for each hour
        for hour in range(1, len(cf_profile_df)):
            for hour_i in range(1, min(hour, self.elecOverloadRecharge)+1):
                if can_overload[hour] and can_overload[hour-hour_i]:
                    can_overload[hour] = False
                    
        cf_profile_df['Max_Overload'] = self.elecOverload
        cf_profile_df['Energy_generated'] = cf_profile_df['Generator_CF'] * oversize
        cf_profile_df['Energy_for_overloading'] = cf_profile_df[['Max_Overload', 'Energy_generated']].min(axis=1)
        cf_profile_df['Electrolyser_CF_overload'] = np.where(
            can_overload,
            cf_profile_df['Energy_for_overloading'],
            cf_profile_df['Electrolyser_CF']
        )

        return cf_profile_df['Electrolyser_CF_overload']

    def _scale_capex(self, unscaled_capex, capacity, reference_capacity, scale_factor):
        """
        Scale capital costs based on capacity.
        
        Args:
            unscaled_capex: Base capital cost
            capacity: Capacity being costed
            reference_capacity: Reference capacity for scaling
            scale_factor: Scaling factor for cost reduction
            
        Returns:
            Scaled capital cost
        """
        if capacity > 0:
            scaled_capex = unscaled_capex * reference_capacity * (capacity / reference_capacity) ** scale_factor / capacity
        else:
            scaled_capex = unscaled_capex
        return scaled_capex

    def _get_capex(self, equip_cost, equip_pc, install_pc, land_pc):
        """
        Calculate total capital cost including indirect costs.
        
        Args:
            equip_cost: Equipment cost
            equip_pc: Equipment percentage
            install_pc: Installation percentage
            land_pc: Land percentage
            
        Returns:
            Total capital cost
        """
        capex = equip_cost * (1 + install_pc / equip_pc) * (1 + land_pc)
        return capex

    def _get_tabulated_outputs(self, working_df):
        """
        Generate summary results from hourly operation data.
        
        Args:
            working_df: DataFrame with hourly operation data
            
        Returns:
            Dictionary with summarized outputs
        """
        operating_outputs = {}
        
        operating_outputs["Generator Capacity Factor"] = working_df['Generator_CF'].mean()
        operating_outputs["Time Electrolyser is at its Rated Capacity"] = \
            working_df.loc[working_df['Electrolyser_CF'] == self.elecMaxLoad,
                          'Electrolyser_CF'].count() / self.hoursPerYear
        operating_outputs["Total Time Electrolyser is Operating"] = working_df.loc[working_df['Electrolyser_CF'] > 0,
                                                            'Electrolyser_CF'].count() / self.hoursPerYear
        operating_outputs["Achieved Electrolyser Capacity Factor"] = working_df['Electrolyser_CF'].mean()
        operating_outputs["Energy in to Electrolyser [MWh/yr]"] = working_df['Electrolyser_CF'].sum() * \
                                                                 self.elecCapacity
        operating_outputs["Surplus Energy [MWh/yr]"] = working_df['Generator_CF'].sum() * self.genCapacity - \
            working_df['Electrolyser_CF'].sum() * self.elecCapacity
        operating_outputs["Hydrogen Output for Fixed Operation [t/yr]"] = working_df['Hydrogen_prod_fixed'].sum() * \
            self.elecCapacity * self.kgtoTonne
        operating_outputs["Hydrogen Output for Variable Operation [t/yr]"] = \
            working_df['Hydrogen_prod_variable'].sum() * self.elecCapacity * self.kgtoTonne
            
        self.operating_outputs = operating_outputs
        return operating_outputs

    def make_duration_curve(self, save_path=None, generator=True, electrolyser=False):
        """
        Create annual duration curves for the chosen configuration.
        For Streamlit compatibility, this returns a figure instead of showing it.

        Parameters
        ----------
        save_path : str, optional
            Path to save the figure. If None, figure is not saved
        generator : bool, optional
            Include generator duration curve
        electrolyser : bool, optional
            Include electrolyser duration curve

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure object
        """
        import matplotlib.pyplot as plt
        
        plots = []
        if generator:
            plots.append("Generator")
        if electrolyser:
            plots.append("Electrolyser")
        elif not generator:
            raise ValueError("generator or electrolyser must be True")

        if self.windCapacity == 0:
            tech = "solar"
        elif self.solarCapacity == 0:
            tech = "wind"
        else:
            tech = "hybrid"

        colours = {"solar": "goldenrod", "wind": "royalblue", "hybrid": "limegreen"}
        hourly_df = self._calculate_hourly_operation()
        
        fig = plt.figure(figsize=(10, 6))
        for i in range(len(plots)):
            gen_elec = plots[i]
            generation = hourly_df[gen_elec + '_CF'].sort_values(ascending=False).reset_index(drop=True) * 100
            generation.index = generation.index / 8760 * 100
            ax = fig.add_subplot(1, len(plots), i+1)
            generation.plot(ax=ax, color=colours[tech])
            ax.set(title=f"{tech.capitalize()} {gen_elec} Capacity Factor - {self.location}",
                  xlabel="Proportion of year (%)", ylabel=f"{gen_elec} Capacity Factor (%)")
            ax.set_ylim(0, 100)
            ax.grid(axis='y', which='both')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def get_duration_curve_data(self, generator=True, electrolyser=False):
        """
        Get duration curve data in a Streamlit/Plotly compatible format.
        
        Parameters
        ----------
        generator : bool, optional
            Include generator duration curve data
        electrolyser : bool, optional
            Include electrolyser duration curve data
            
        Returns
        -------
        dict
            Dictionary containing plot data with keys 'generator' and/or 'electrolyser'
            Each containing 'x' (time %), 'y' (capacity factor %), 'title', 'color'
        """
        plots = []
        if generator:
            plots.append("Generator")
        if electrolyser:
            plots.append("Electrolyser")
        elif not generator:
            raise ValueError("generator or electrolyser must be True")
        
        if self.windCapacity == 0:
            tech = "solar"
        elif self.solarCapacity == 0:
            tech = "wind"
        else:
            tech = "hybrid"
            
        colours = {"solar": "#DAA520", "wind": "#4169E1", "hybrid": "#32CD32"}
        hourly_df = self._calculate_hourly_operation()
        
        plot_data = {}
        for plot_type in plots:
            gen_elec = plot_type.lower()
            generation = hourly_df[plot_type + '_CF'].sort_values(ascending=False).reset_index(drop=True) * 100
            time_percent = generation.index / 8760 * 100
            
            plot_data[gen_elec] = {
                'x': time_percent.tolist(),
                'y': generation.tolist(),
                'title': f"{tech.capitalize()} {plot_type} Capacity Factor - {self.location}",
                'xlabel': "Proportion of year (%)",
                'ylabel': f"{plot_type} Capacity Factor (%)",
                'color': colours[tech],
                'technology': tech
            }
            
        return plot_data
    
    def get_results_summary(self):
        """
        Get a comprehensive results summary suitable for Streamlit display.
        
        Returns
        -------
        dict
            Dictionary with organized results for dashboard display
        """
        if not self.operating_outputs:
            self.calculate_electrolyser_output()
            
        # Calculate costs if not already done
        fixed_lcoh = self.calculate_costs('fixed')
        variable_lcoh = self.calculate_costs('variable')
        
        return {
            'system_configuration': {
                'Location': self.location,
                'Electrolyser Type': self.elecType,
                'Electrolyser Capacity (MW)': self.elecCapacity,
                'Solar Capacity (MW)': self.solarCapacity,
                'Wind Capacity (MW)': self.windCapacity,
                'Battery Power (MW)': self.batteryPower,
                'Battery Duration (hours)': self.batteryHours,
                'Total Generator Capacity (MW)': self.genCapacity
            },
            'operational_results': {
                'Generator Capacity Factor': f"{self.operating_outputs['Generator Capacity Factor']:.1%}",
                'Electrolyser Capacity Factor': f"{self.operating_outputs['Achieved Electrolyser Capacity Factor']:.1%}",
                'Time at Rated Capacity': f"{self.operating_outputs['Time Electrolyser is at its Rated Capacity']:.1%}",
                'Total Operating Time': f"{self.operating_outputs['Total Time Electrolyser is Operating']:.1%}",
                'Annual Energy to Electrolyser (MWh)': f"{self.operating_outputs['Energy in to Electrolyser [MWh/yr]']:,.0f}",
                'Annual Surplus Energy (MWh)': f"{self.operating_outputs['Surplus Energy [MWh/yr]']:,.0f}",
            },
            'hydrogen_production': {
                'Fixed Operation (t/year)': f"{self.operating_outputs['Hydrogen Output for Fixed Operation [t/yr]']:,.1f}",
                'Variable Operation (t/year)': f"{self.operating_outputs['Hydrogen Output for Variable Operation [t/yr]']:,.1f}",
                'Fixed Operation (kg/day)': f"{self.operating_outputs['Hydrogen Output for Fixed Operation [t/yr]'] * 1000 / 365:,.1f}",
                'Variable Operation (kg/day)': f"{self.operating_outputs['Hydrogen Output for Variable Operation [t/yr]'] * 1000 / 365:,.1f}",
            },
            'financial_results': {
                'LCOH - Fixed Consumption ($/kg)': f"{fixed_lcoh:.2f}",
                'LCOH - Variable Consumption ($/kg)': f"{variable_lcoh:.2f}",
                'Project Life (years)': self.projectLife,
                'Discount Rate': f"{self.discountRate:.1%}",
            },
            'technical_parameters': {
                'Electrolyser Efficiency': f"{self.elecEff:.1%}",
                'Specific Consumption (kWh/Nm3)': f"{self.specCons:.1f}",
                'Stack Lifetime (hours)': f"{self.stackLifetime:,}",
                'Water Needs (L/kg H2)': f"{self.waterNeeds:.0f}",
                'H2 Volume to Mass (kg/m3)': f"{self.H2VoltoMass:.3f}",
            }
        }
