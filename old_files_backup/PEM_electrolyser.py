
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

# -----------------------------------
# 0. GLOBAL STYLE SETTINGS
# -----------------------------------
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelweight': 'bold',
    #'axes.titleweight': 'bold',
    #'axes.titlesize': 14,
    #'axes.labelsize': 12,
    #'legend.fontsize': 10,
    #'xtick.labelsize': 10,
    #'ytick.labelsize': 10,
    #'figure.figsize': (),
    #'grid.color': '0.8',
    #'grid.linestyle': '--',
    #'grid.linewidth': 0.5,
    #'lines.linewidth': 2,
    #'lines.markersize': 6,
})


# Color-blind–friendly palette (Wong/Courant “CB” set)
cb = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']

# Color-blind-friendly palette (blue, vermillion, green, purple)

cb_colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']


# ------------------------
# 1. MODEL EQUATIONS
# ------------------------

# Constants
F = 96485.0           # C/mol
R = 8.314             # J/mol/K

# Default parameters
base_params = {
    'T': 333.15,           # K    Temperature 353.15,
    'Pa': 1.0,             # atm   Anode Pressure In
    'Pc': 13.6,            # atm   Cathode Pressure out 13.6,
    'j0_a': 1.65e-8,        # A/cm²  Exchange current density Anode, Harion et
    'j0_c': 9e-2,          # A/cm²  Exchange current density Cath,Harison et al
    'alpha_a': 2.0,        #        Charge transfer coeffs. anode
    'alpha_c': 0.5,        #        Charge transfer coeffs cathode
    'd_m': 178e-4,         # cm (178 µm)
    'sigma_m_coeff': (0.005139*22 - 0.00326),  # pre-exponential
    'j_lim': 3.0,           # A/cm² (limiting current density)
    'd_e': 200e-4,         # cm (electrode thickness)
    'rho_e_a': 5e-3,       # Ω·cm (anode electrode resistivity) Titanium
    'rho_e_c': 8e-2,       # Ω·cm (cathode electrode resistivity) Carbon paper
}

def reversible_voltage(T, a_H2=1.0, a_O2=1.0, a_H2O=1.0):
    V0 = 1.229 - 0.0009*(T - 298.15)
    return V0 + (R * T) / (2 * F) * np.log(a_H2 / (a_O2**0.5 * a_H2O))

def activation_overpotential(j, j0, alpha, T):
    return (R * T) / (alpha * F) * np.arcsinh(j / (2 * j0))

def activation_total(j, params):
    V_act_a = activation_overpotential(j, params['j0_a'], params['alpha_a'], params['T'])
    V_act_c = activation_overpotential(j, params['j0_c'], params['alpha_c'], params['T'])
    return V_act_a + V_act_c

def diffusion_overpotential(j, params):
    return - (R * params['T']) / (2 * F) * np.log(1 - j / params['j_lim'])

def sigma_m(T, coeff):
    return coeff * np.exp(1268 * (1/303 - 1/T))

def ohmic_overpotential(j, params):
    return j * params['d_m'] / sigma_m(params['T'], params['sigma_m_coeff'])

def ohmic_overpotential(j, params):
    # Electrode resistances (Ω·cm²)
    R_elec_a = params['rho_e_a'] * params['d_e']   # anode electrode
    R_elec_c = params['rho_e_c'] * params['d_e']   # cathode electrode
    # Membrane resistance (Ω·cm²)
    R_mem = params['d_m'] / sigma_m(params['T'], params['sigma_m_coeff'])
    # Total ohmic overpotential
    return j * (R_elec_a + R_elec_c + R_mem)

def cell_voltage(j, params):
    V_rev = reversible_voltage(params['T'])
    V_act = activation_total(j, params)
    V_diff = diffusion_overpotential(j, params)
    V_ohm = ohmic_overpotential(j, params)
    return V_rev + V_act + V_diff + V_ohm

# ------------------------
# 2. VALIDATION FRAMEWORK
# ------------------------

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot

def validate_model(j_exp, V_exp, model_func, params):
    V_pred = model_func(j_exp, params)
    metrics = {
        'RMSE': rmse(V_exp, V_pred),
        'MAE': mae(V_exp, V_pred),
        'R2': r2_score(V_exp, V_pred)
    }
    return V_pred, metrics

def plot_validation(j_test, V_pred, j_exp, V_exp, params, metrics):
	
	
    # Polarization curve comparison
    plt.figure()
    plt.scatter(j_exp, V_exp, label='Experimental')
    V_pred = cell_voltage(j_test, params)
    plt.plot(j_test, V_pred, label='Model')
    plt.xlabel('Current Density (A/cm²)')
    plt.ylabel('Cell Voltage (V)')
    plt.title('Polarization Curve Comparison')
    plt.legend()
    plt.show()

    # Voltage components breakdown
    V_rev_val = reversible_voltage(params['T'])
    V_rev = np.full_like(j_exp, V_rev_val)
    V_act = activation_total(j_exp, params)
    V_diff = diffusion_overpotential(j_exp, params)
    V_ohm = ohmic_overpotential(j_exp, params)

    plt.figure()
    plt.stackplot(j_exp, V_rev, V_act, V_diff, V_ohm,
                  labels=['Reversible','Activation','Diffusion','Ohmic'])
    plt.xlabel('Current Density (A/cm²)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Components Breakdown')
    plt.legend(loc='upper left')
    plt.show()

    # Error metrics bar chart
    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Validation Metrics')
    plt.show()
# ---------------------------
#  Test + experimental data
# -------------------------------
j_test = np.linspace(0.01, 3.0, 50)
V_true = cell_voltage(j_test, base_params)
#V_exp = V_true + np.random.normal(scale=0.02, size=j_test.shape)

#experimental data from Debe et al on 3m membranes.
j_exp =np.array([0.14, 0.24, 0.33, 0.47, 0.54, 0.58, 0.80, 0.96, 1.10, 1.24, 1.34, 1.46, 1.59, 1.72, 1.84])
V_exp = np.array([1.543, 1.584, 1.607, 1.619, 1.663, 1.650, 1.692, 1.723, 1.748, 1.773, 1.791, 1.811, 1.832, 1.853, 1.874])

V_pred, metrics = validate_model(j_exp, V_exp, cell_voltage, base_params)
print(metrics)
plot_validation(j_test, V_pred, j_exp, V_exp, base_params, metrics)

# ------------------------
# 3. SENSITIVITY ANALYSIS
# ------------------------

def plot_sensitivity(param_name, values, base_params, j_range):
    plt.figure()
    for val in values:
        params = base_params.copy()
        params[param_name] = val
        V = cell_voltage(j_range, params)
        plt.plot(j_range, V, label=f"{param_name}={val}")
    plt.xlabel('Current Density (A/cm²)')
    plt.ylabel('Cell Voltage (V)')
    plt.title(f'Sensitivity: {param_name}')
    plt.legend()
    plt.show()

j_range = np.linspace(0.01, 3.0, 100)

# Sensitivities
plot_sensitivity('j0_a', [1e-6, 1e-5, 1e-4], base_params, j_range)
plot_sensitivity('T', [323.15, 353.15, 383.15], base_params, j_range)
plot_sensitivity('d_m', [100e-4, 178e-4, 250e-4], base_params, j_range)
plot_sensitivity()
# Extend with pressure, membrane thickness, etc. as needed.
