# EE320 IV Analysis Pipeline — parser + plotting
'''
[ ] plot I-V curves for Photodiode
[ ] scale factor between LED power output and the PD saturation current Isat
[ ] plot L-I-V curves for LED
    * plot L-I, V-I (where I is on x-axis), and overlay onto the same graph
    // where L is light output (power)
[ ] threshold current of Laser diode
[ ] scale factor between LD power output and the PD saturation current Isat
[ ] plot L-I curves of LD
    // where L is light output (power), and I is foward current (I_F)
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# PRE-PROCESS

# 1. Parse a single Keithley CSV file
# EDIT: swap_axes
def parse_keithley_csv(filepath, swap_axes=False):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Value" in line and "Reading" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find header line in {filepath}")

    df = pd.read_csv(filepath, skiprows=header_idx)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df = df[["Value", "Reading"]].copy()
    if swap_axes:
        # For cases like LD L–I–V where Value=I and Reading=V
        df.columns = ["Current (A)", "Voltage (V)"]
    else:
        df.columns = ["Voltage (V)", "Current (A)"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df

# 2. Load all CSVs in a folder into an experiment dict
def load_experiment(folder_path):
    experiment = {}
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]

    for fname in csv_files:
        fpath = os.path.join(folder_path, fname)
        try:
            df = parse_keithley_csv(fpath)
            key = os.path.splitext(fname)[0]
            experiment[key] = df
            print(f"Loaded {fname}: {len(df)} points")
        except Exception as e:
            print(f"Skipped {fname}: {e}")
    return experiment


# AUX

# Estimate LED turn-on voltage as first V where I > threshold
# USE: Lab 2b 1.1
def compute_turn_on_voltage(df, current_threshold=1e-3):
    subset = df[df["Current (A)"] > current_threshold]
    if subset.empty:
        return np.nan
    return subset["Voltage (V)"].iloc[0]

# Get PD Isat (under each LED bias)
# USE: Lab 2b 1.4
def compute_saturation_current(df):
    n = max(1, len(df)//10)
    subset = df.nsmallest(n, "Voltage (V)")
    return subset["Current (A)"].mean()


# PARAM EXTRACTION FUNCTIONS

# Extract key params dep on device type
# EDIT: device_type
def extract_parameters(experiment, device_type="photodiode"):
    results = {}
    for key, df in experiment.items():
        if device_type == "LED":
            results[key] = compute_turn_on_voltage(df)
        elif device_type == "photodiode":
            results[key] = compute_saturation_current(df)
        else:
            # generic placeholder for future (LD, FET)
            results[key] = np.nan

    return results

# Estimate Laser I threshold
# USE: Lab 2b 1.5
def find_threshold_current(df, smooth_window=3):
    I = df["Current (A)"].values
    V = df["Voltage (V)"].values
    # Smooth the curve
    V_smooth = np.convolve(V, np.ones(smooth_window)/smooth_window, mode="valid")
    I_smooth = I[:len(V_smooth)]
    dVdI = np.gradient(V_smooth, I_smooth)
    # Threshold = current where slope rises most rapidly
    idx = np.argmax(np.gradient(dVdI))
    return I_smooth[idx]


# PLOTTING

# Plot a single I–V curve
def plot_IV_curve(df, title="I–V Characteristic", log_scale=False):
    plt.figure(figsize=(6,4))
    plt.plot(df["Voltage (V)"], df["Current (A)"], marker='o', markersize=3, linewidth=1)
    plt.title(title)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    if log_scale:
        plt.yscale("log")
        plt.ylabel("Current (A, log scale)")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# Plot multiple I–V curves overlaid
def plot_IV_overlay(
    experiment,
    labels=None,
    title="Overlaid I–V Curves",
    log_scale=False,
    flip_current=False
):
    """
    Plots multiple I–V curves on one figure.
    Used for experiments like LED_LIV or PD under illumination.
    """
    plt.figure(figsize=(6,4))

    for i, (key, df) in enumerate(sorted(experiment.items())):
        label = labels[i] if labels and i < len(labels) else key
        V = df["Voltage (V)"].values
        I = df["Current (A)"].values
        if flip_current:
            I = -I
        plt.plot(V, I, label=label, linewidth=1)

    plt.title(title)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    if log_scale:
        plt.yscale("log")
        plt.ylabel("Current (A, log scale)")

    plt.legend(fontsize=8)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# Plot relationsgips like Isat v I_lED
# USE: Lab 2b 1.4
def plot_relationships(results, x_values=None, xlabel="Drive Current (mA)", ylabel="Saturation Current (µA)", title="L–I Relationship"):
    if x_values is None:
        x_values = np.arange(len(results))

    y_values = list(results.values())

    # plt.scatter(df.iloc[:,0], df.iloc[:,1], s=8)
    # plt.xlabel(df.columns[0])
    # plt.ylabel(df.columns[1])
    # plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(x_values, np.array(y_values)*1e6, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# Plot Lase Diode L-I curve, marks estimated threshold curr
# USE: Lab 2b 1.5
def plot_LI_with_threshold(df, RL_ohm=None, responsivity_A_per_W=None,
                           title="Laser Diode L–I Curve (V ∝ Optical Power)"):
    I_mA = df["Current (A)"].values * 1e3
    Y = df["Voltage (V)"].values
    y_label = "Photodiode Voltage (V)"

    # Optional unit conversion: V → Optical Power (mW)
    if RL_ohm and responsivity_A_per_W:
        Y = (Y / (RL_ohm * responsivity_A_per_W)) * 1e3
        y_label = "Optical Output Power (mW)"

    # Find threshold current
    Ith_A = find_threshold_current(df)
    Ith_mA = Ith_A * 1e3

    plt.figure(figsize=(6, 4))
    plt.scatter(I_mA, Y, s=12, color="tab:red", label="Measured")
    plt.axvline(Ith_mA, ls="--", color="gray",
                alpha=0.7, label=f"Ith ≈ {Ith_mA:.1f} mA")
    plt.title(title)
    plt.xlabel("Laser Diode Current (mA)")
    plt.ylabel(y_label)
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Estimated threshold current ≈ {Ith_mA:.1f} mA")





if __name__ == "__main__":
    # 1.1 LED I–V
    df_led = parse_keithley_csv("lab_2b/LED_IV.csv")
    plot_IV_curve(df_led, title="LED I–V (Linear)")
    plot_IV_curve(df_led, title="LED I–V (Log)", log_scale=True)
    V_on = compute_turn_on_voltage(df_led)
    print(f"LED turn-on ≈ {V_on:.2f} V")

    # 1.2 Laser Diode I–V
    df_ld = parse_keithley_csv("lab_2b/LD_IV.csv")
    plot_IV_curve(df_ld, title="Laser Diode I–V")

    # 1.3 Photodiode (PD)
    df_pd = parse_keithley_csv("lab_2b/PD_dark.csv")
    plot_IV_curve(df_pd, title="Photodiode I–V (Linear)")
    plot_IV_curve(df_pd, title="Photodiode I–V (Log)", log_scale=True)


    # 1.4 LED LIV
    files = {
    "LED 0 mA": "lab_2b/LED_LIV/141.csv",
    "LED 3 mA": "lab_2b/LED_LIV/142.csv",
    "LED 6 mA": "lab_2b/LED_LIV/143.csv"
    }
    experiment = {label: parse_keithley_csv(path) for label, path in files.items()}

    plt.figure(figsize=(6,4))
    for label, df in experiment.items():
        plt.plot(df["Voltage (V)"], df["Current (A)"], label=label)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("Photodiode I–V under LED illumination")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Extract saturation currents for each illumination level
    isats = {label: compute_saturation_current(df) for label, df in experiment.items()}
    baseline = list(isats.values())[0]
    isats_corrected = {k: v - baseline for k, v in isats.items()}

    I_led = [0, 3, 6] # mA corresponding to the illumination files

    # Load LED’s own I–V data for the red voltage curve
    df_led = parse_keithley_csv("lab_2b/LED_IV.csv")

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(I_led, [i*1e6 for i in isats_corrected.values()], 'o-b', label="Optical Power ∝ Isat (µA)")
    ax1.set_xlabel("LED Drive Current (mA)")
    ax1.set_ylabel("Optical Power (µA above dark)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, ls="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(df_led["Current (A)"]*1e3, df_led["Voltage (V)"], 'r-', label="Forward Voltage (V)")
    ax2.set_ylabel("Forward Voltage (V)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("LED L–I–V Characteristics (two-axis, baseline-removed)")
    plt.tight_layout()
    plt.show()


    # 1.5 Laser Diode LIV
    df_ld_liv = parse_keithley_csv("lab_2b/LD_LIV.csv", swap_axes=True)
    plot_LI_with_threshold(df_ld_liv, title="Laser Diode L–I (V ∝ Optical Power)")

    Ith = find_threshold_current(df_ld_liv)
    print(f"Laser threshold ≈ {Ith*1e3:.1f} mA")
