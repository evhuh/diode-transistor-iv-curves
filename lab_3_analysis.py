
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["axes.grid"] = True

vgs_jfet_n = np.linspace(0, -3.0, 21)    # jn1 → jn21
vgs_jfet_p = np.linspace(0,  3.0, 21)    # jp1 → jp21
vgs_nmos    = np.linspace(0,  5.0, 21)  # mn1 → mn21
vgs_pmos    = np.linspace(0, -5.0, 21)  # mp1 → mp21


# KEITHLEY PARSER
def read_keithley_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a Keithley CSV file (like jn1.csv / jp1.csv / mn1.csv / mp1.csv).

    Returns dataframe with:
        - Index (step index)
        - Id    (drain current, A)
        - Vds   (drain-source voltage, V)
    """
    path = Path(path)
    raw = pd.read_csv(path, skiprows=7)

    df = pd.DataFrame({
        "Index": raw["Index"].astype(int),
        "Id": raw["Reading"].astype(float),       # Amp DC
        "Vds": raw["Value"].astype(float),        # Volt DC
    })
    df["file"] = path.name
    return df


# LOAD ALL SWEEPS FOR A DEVICE
def load_device_sweeps_from_folder(folder: Union[str, Path],
                                   vgs_list) -> pd.DataFrame:
    """
    Load all Keithley sweeps from a folder containing 21 CSV files.
    Example folder:
        lab3/jfet_n/
        lab3/jfet_p/
        lab3/nmos/
        lab3/pmos/

    Assumes files end with .csv and are sorted correctly.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))

    if len(files) != len(vgs_list):
        raise ValueError(
            "%s: found %d files, but %d Vgs values provided" %
            (folder, len(files), len(vgs_list))
        )

    dfs = []
    for file, vgs in zip(files, vgs_list):
        df = read_keithley_csv(file)
        df["Vgs"] = float(vgs)
        df["device"] = folder.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# ID–VDS PLOTS
def plot_id_vds_family(df: pd.DataFrame, title=""):
    """
    Plot ID–VDS curves for all VGS sweeps.
    """
    plt.figure()

    for vgs, grp in sorted(df.groupby("Vgs"), key=lambda x: x[0]):
        plt.plot(grp["Vds"], grp["Id"], label="Vgs = %.2f V" % vgs)

    plt.xlabel("Vds (V)")
    plt.ylabel("Id (A)")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


# Extract ID–VGS (take ID at highest VDS)
def extract_id_vs_vgs_from_family(df: pd.DataFrame, vds_target=None):
    """
    From ID–VDS family, extract ID at a fixed VDS for each VGS.
    If vds_target is None, the highest Vds is used.
    """
    if vds_target is None:
        vds_target = df["Vds"].max()

    out = []
    for vgs, grp in df.groupby("Vgs"):
        idx = (grp["Vds"] - vds_target).abs().idxmin()
        row = grp.loc[idx, ["Vgs", "Vds", "Id"]]
        out.append(row)

    out_df = pd.DataFrame(out).sort_values("Vgs").reset_index(drop=True)
    return out_df


def plot_id_vgs(id_vgs: pd.DataFrame, title=""):
    plt.figure()
    plt.plot(id_vgs["Vgs"], id_vgs["Id"], "o-")
    plt.xlabel("Vgs (V)")
    plt.ylabel("Id (A)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_id_vgs_semilog(id_vgs: pd.DataFrame, title=""):
    plt.figure()
    plt.semilogy(id_vgs["Vgs"], np.abs(id_vgs["Id"]), "o-")
    plt.xlabel("Vgs (V)")
    plt.ylabel("|Id| (A)")
    plt.title(title)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


# gm = dId/dVgs
def compute_gm_vs_vgs(id_vgs: pd.DataFrame):
    vgs = id_vgs["Vgs"].values
    id_ = id_vgs["Id"].values
    gm = np.gradient(id_, vgs)
    return pd.DataFrame({"Vgs": vgs, "gm": gm})


def plot_gm_vs_vgs(gm_df: pd.DataFrame, title=""):
    plt.figure()
    plt.plot(gm_df["Vgs"], gm_df["gm"], "o-")
    plt.xlabel("Vgs (V)")
    plt.ylabel("gm (S)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# MOSFET – Extract Vt and k from linear region
def extract_mos_linear_params(df: pd.DataFrame, vds_max_linear=0.1):
    """
    Fit slopes m = dId/dVds at small Vds for each Vgs.
    slopes ≈ k * (Vgs - Vt)
    Fit slope vs Vgs to extract k and Vt.
    """
    slopes = []
    vgs_vals = []

    for vgs, grp in df.groupby("Vgs"):
        lin = grp[grp["Vds"] <= vds_max_linear]
        if len(lin) < 3:
            continue
        m, b = np.polyfit(lin["Vds"], lin["Id"], 1)
        slopes.append(m)
        vgs_vals.append(vgs)

    vgs_vals = np.array(vgs_vals)
    slopes = np.array(slopes)

    # slope = k*Vgs - k*Vt
    k_lin, intercept = np.polyfit(vgs_vals, slopes, 1)
    vt_lin = -intercept / k_lin

    return vt_lin, k_lin, vgs_vals, slopes


# MOSFET – Extract Vt and k from saturation region
def extract_mos_sat_params(df: pd.DataFrame, vds_min_sat=1.0):
    """
    Use highest Vds ≥ vds_min_sat to approximate Id_sat for each Vgs.
    Then fit sqrt(Id_sat) vs Vgs to get Vt and k.
    """
    rows = []

    for vgs, grp in df.groupby("Vgs"):
        sat = grp[grp["Vds"] >= vds_min_sat]
        if len(sat) == 0:
            continue
        row = sat.loc[sat["Vds"].idxmax()]
        rows.append(row[["Vgs", "Id", "Vds"]])

    sat_df = pd.DataFrame(rows).dropna()
    sat_df = sat_df[sat_df["Id"] > 0]

    vgs = sat_df["Vgs"].values
    ids = sat_df["Id"].values
    y = np.sqrt(ids)

    slope, intercept = np.polyfit(vgs, y, 1)
    vt_sat = -intercept / slope
    k_sat = 2 * slope**2

    return vt_sat, k_sat, vgs, ids


# MOSFET – Early Voltage and channel resistance r_o
def estimate_early_voltage(df: pd.DataFrame, vds_min_sat=1.0):
    """
    Fit Id = m*Vds + b in saturation region.
    Extrapolate to Id=0 → intercept = -Va.
    """
    vas = []
    vgs_list = sorted(df["Vgs"].unique())

    for vgs in vgs_list:
        grp = df[(df["Vgs"] == vgs) & (df["Vds"] >= vds_min_sat)]
        if len(grp) < 3:
            continue
        m, b = np.polyfit(grp["Vds"], grp["Id"], 1)
        if m == 0:
            continue
        v_intercept = -b / m
        vas.append(-v_intercept)

    return np.array(vas)


def compute_ro_from_va(id_value, va):
    return va / id_value




if __name__ == "__main__":

    BASE = Path(".") 

    # Load all 4 devices
    jfet_n = load_device_sweeps_from_folder(BASE / "jfet_n", vgs_jfet_n)
    jfet_p = load_device_sweeps_from_folder(BASE / "jfet_p", vgs_jfet_p)
    nmos   = load_device_sweeps_from_folder(BASE / "nmos",    vgs_nmos)
    pmos   = load_device_sweeps_from_folder(BASE / "pmos",    vgs_pmos)

    print("Loaded JFET N:", jfet_n.shape)
    print("Loaded JFET P:", jfet_p.shape)
    print("Loaded NMOS:", nmos.shape)
    print("Loaded PMOS:", pmos.shape)

    # Plot ID–VDS families
    plot_id_vds_family(jfet_n, "JFET N: Id–Vds")
    plot_id_vds_family(jfet_p, "JFET P: Id–Vds")
    plot_id_vds_family(nmos,   "NMOS: Id–Vds")
    plot_id_vds_family(pmos,   "PMOS: Id–Vds")

    # Extract ID–VGS for each device
    jfet_n_idvgs = extract_id_vs_vgs_from_family(jfet_n)
    jfet_p_idvgs = extract_id_vs_vgs_from_family(jfet_p)
    nmos_idvgs   = extract_id_vs_vgs_from_family(nmos)
    pmos_idvgs   = extract_id_vs_vgs_from_family(pmos)

    plot_id_vgs(jfet_n_idvgs, "JFET N: Id–Vgs")
    plot_id_vgs(jfet_p_idvgs, "JFET P: Id–Vgs")
    plot_id_vgs(nmos_idvgs,   "NMOS: Id–Vgs")
    plot_id_vgs(pmos_idvgs,   "PMOS: Id–Vgs")

    # Semilog versions (MOSFET requirement)
    plot_id_vgs_semilog(nmos_idvgs, "NMOS: Id–Vgs (semilog)")
    plot_id_vgs_semilog(pmos_idvgs, "PMOS: Id–Vgs (semilog)")

    # Compute gm
    jfet_n_gm = compute_gm_vs_vgs(jfet_n_idvgs)
    jfet_p_gm = compute_gm_vs_vgs(jfet_p_idvgs)
    nmos_gm   = compute_gm_vs_vgs(nmos_idvgs)
    pmos_gm   = compute_gm_vs_vgs(pmos_idvgs)

    plot_gm_vs_vgs(jfet_n_gm, "JFET N: gm vs Vgs")
    plot_gm_vs_vgs(jfet_p_gm, "JFET P: gm vs Vgs")
    plot_gm_vs_vgs(nmos_gm,   "NMOS: gm vs Vgs")
    plot_gm_vs_vgs(pmos_gm,   "PMOS: gm vs Vgs")

    # Linear region extraction
    vt_lin_nmos, k_lin_nmos, vgs_lin, slopes_lin = extract_mos_linear_params(nmos)
    print("\nNMOS linear-region Vt =", vt_lin_nmos)
    print("NMOS linear-region k  =", k_lin_nmos)

    vt_lin_pmos, k_lin_pmos, _, _ = extract_mos_linear_params(pmos)
    print("\nPMOS linear-region Vt =", vt_lin_pmos)
    print("PMOS linear-region k  =", k_lin_pmos)

    # Saturation region extraction
    vt_sat_nmos, k_sat_nmos, _, _ = extract_mos_sat_params(nmos)
    print("\nNMOS saturation Vt =", vt_sat_nmos)
    print("NMOS saturation k  =", k_sat_nmos)

    vt_sat_pmos, k_sat_pmos, _, _ = extract_mos_sat_params(pmos)
    print("\nPMOS saturation Vt =", vt_sat_pmos)
    print("PMOS saturation k  =", k_sat_pmos)

    vas_nmos = estimate_early_voltage(nmos, vds_min_sat=1.0)
    print("\nNMOS Early voltages (per Vgs):", vas_nmos)
    print("NMOS Average Early voltage Va ≈", np.mean(vas_nmos))
