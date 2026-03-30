"""
==============================================================================
MODULE 3: SPATIAL DYNAMICS
==============================================================================
Perryman et al. (1997) Recreation Pipeline — Module 3

Purpose:
  Given the confirmed Hyades members from Module 2, compute:
    (a) the 3-D cluster centre of mass using mass-weighted positions
    (b) the tidal (Jacobi) radius from Galactic Oort constants

Physics Background
------------------

  CENTRE OF MASS
  ──────────────
  We assign stellar masses via the empirical Main-Sequence mass-luminosity
  relation calibrated on nearby stars.  For solar-type main-sequence stars
  the most reliable analytic approximation (Eker et al. 2018, MNRAS 479)
  in absolute V-band magnitude is:

    log₁₀(M/M☉) = a + b·Mv + c·Mv²  (piecewise by spectral type)

  Since Gaia supplies G-band magnitudes we use the colour transformation:
    G ≈ V − 0.0257 − 0.0924·(B−V) − 0.1623·(B−V)²   (Evans et al. 2018)
  or more simply work directly in G with the Pecaut & Mamajek (2013) ML
  relation parameterised in terms of BP−RP colour.

  TIDAL (JACOBI) RADIUS
  ─────────────────────
  The tidal radius is the distance from the cluster centre at which the
  Galactic tidal acceleration equals the cluster's self-gravity.  From the
  restricted three-body problem in the rotating Galactic frame, the Jacobi
  (Roche-lobe) radius is (King 1962; Binney & Tremaine 2008 §8.3):

    r_t = ( G M_cl / (2 Ω² − ∂²Φ/∂R²) )^(1/3)

  where Ω is the Galactic angular velocity at the cluster's Galactocentric
  radius R_cl, and ∂²Φ/∂R² is the radial epicyclic term.  In terms of the
  Oort constants A and B:

    Ω² − ∂²Φ/∂R² = −4 A(A−B)     [Oort's (1927) result]

  and the standard form becomes:

    r_t = ( G M_cl / (−4 A(A − B) + κ²) )^(1/3)

  More commonly written as (Pinfield et al. 1998, MNRAS 299):

    r_t  =  [ G M_cl / (4 A |A − B|) ]^(1/3)    ... (*)

  with Oort constants (Feast & Whitelock 1997):
    A = +14.82 ± 0.84 km s⁻¹ kpc⁻¹
    B = −12.37 ± 0.64 km s⁻¹ kpc⁻¹

  The Galactocentric distance of the cluster matters because A and B are
  evaluated locally; for the Hyades at R_cl ≈ 8.07 kpc the solar values
  are an excellent approximation.

Dependencies:
  pip install astropy numpy pandas scipy
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from astropy.coordinates import SkyCoord, Galactocentric, ICRS
import astropy.units as u
from astropy.constants import G as G_newt

# --------------------------------------------------------------------------- #
log = logging.getLogger(__name__)

# ── Oort constants (Feast & Whitelock 1997; consistent with Perryman 1997) ── #
# Units: km s⁻¹ kpc⁻¹
A_OORT = 14.82   # km s⁻¹ kpc⁻¹
B_OORT = -12.37  # km s⁻¹ kpc⁻¹

# Solar Galactocentric distance (for converting kpc → pc)
R_SUN_KPC = 8.122   # Gravity Collaboration 2019

# Solar mass in kg (for unit conversion)
M_SUN = 1.989e30  # kg

# ── Mass–luminosity relation coefficients ─────────────────────────────────── #
# Calibrated here in Gaia G-band absolute magnitude using a simple
# polynomial fit to Pecaut & Mamajek (2013) main-sequence data.
# log10(M/Msun) = sum_k c_k * MG^k
# Valid range: MG in [2.5, 12.0]  (roughly spectral types F0–M4)
_ML_COEFFS = np.array([0.507, -0.0906, 0.00256])   # [c0, c1, c2]
_MG_MIN, _MG_MAX = 2.5, 12.0

# Evolved stars (MG < 2.5) — assign a rough post-MS mass of 1.5 M☉
MASS_EVOLVED_MSUN = 1.5


# =========================================================================== #
#  HELPER — mass_from_absolute_magnitude
# =========================================================================== #
def mass_from_absolute_magnitude(mg: np.ndarray) -> np.ndarray:
    """
    Estimate stellar mass [M☉] from Gaia G-band absolute magnitude M_G.

    Uses a second-order polynomial fitted to Pecaut & Mamajek (2013)
    dwarf main-sequence data.  Stars brighter than the turnoff
    (M_G < 2.5) are assigned an evolved-star mass of 1.5 M☉.

    Parameters
    ----------
    mg : np.ndarray
        Gaia G-band absolute magnitudes.

    Returns
    -------
    np.ndarray
        Estimated masses in solar units.
    """
    mg = np.asarray(mg, dtype=float)
    log10_mass = np.polyval(_ML_COEFFS[::-1], mg)   # numpy convention
    mass = 10.0**log10_mass

    # Clip to physical range (0.1 – 4 M☉ for MS + sub-giant regime)
    mass = np.clip(mass, 0.1, 4.0)

    # Evolved stars (giants / subgiants above turnoff)
    evolved = mg < _MG_MIN
    mass[evolved] = MASS_EVOLVED_MSUN

    return mass


# =========================================================================== #
#  FUNCTION 1 — compute_cluster_centre_of_mass
# =========================================================================== #
def compute_cluster_centre_of_mass(
    members_df: pd.DataFrame,
) -> dict:
    """
    Compute the mass-weighted 3-D centre of mass of the Hyades cluster.

    Requires that the member DataFrame contains:
      - ra, dec, parallax       (for 3-D positions)
      - phot_g_mean_mag         (for mass estimation via M-L relation)

    Steps:
      1. Convert parallax → distance; project (ra, dec, d) to ICRS Cartesian
         (X, Y, Z) in parsecs.
      2. Convert to heliocentric Galactic Cartesian (X_g, Y_g, Z_g).
      3. Estimate each star's mass from its absolute G magnitude.
      4. Compute the mass-weighted centroid.

    Parameters
    ----------
    members_df : pd.DataFrame
        Confirmed Hyades members (is_member == True from Module 2).

    Returns
    -------
    dict with keys:
        ra_com, dec_com, dist_com_pc   — sky position and distance of CoM
        x_com, y_com, z_com            — Galactic Cartesian coords [pc]
        total_mass_msun                — total cluster mass [M☉]
        n_stars                        — number of members used
        masses                         — individual mass estimates [M☉]
    """
    df = members_df.copy()

    # ------------------------------------------------------------------ #
    # Step 1 — compute distances and 3-D positions
    # ------------------------------------------------------------------ #
    valid = (df["parallax"] > 0) & df["phot_g_mean_mag"].notna()
    df = df[valid].copy()
    n  = len(df)
    log.info("Computing CoM using %d members with valid astrometry.", n)

    df["dist_pc"] = 1000.0 / df["parallax"]

    # ------------------------------------------------------------------ #
    # Step 2 — absolute magnitude and mass
    # ------------------------------------------------------------------ #
    df["abs_g"] = df["phot_g_mean_mag"] - 5.0 * np.log10(df["dist_pc"] / 10.0)
    df["mass"]  = mass_from_absolute_magnitude(df["abs_g"].values)
    total_mass  = df["mass"].sum()
    log.info(
        "Total cluster mass (M-L estimate): %.1f M☉  (N=%d stars)",
        total_mass, n,
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Galactic Cartesian positions via astropy
    # ------------------------------------------------------------------ #
    coords = SkyCoord(
        ra       = df["ra"].values    * u.deg,
        dec      = df["dec"].values   * u.deg,
        distance = df["dist_pc"].values * u.pc,
        frame    = "icrs",
    )
    galactic = coords.galactic

    # Convert to Galactic Cartesian
    xyz = galactic.cartesian
    x_pc = xyz.x.to(u.pc).value
    y_pc = xyz.y.to(u.pc).value
    z_pc = xyz.z.to(u.pc).value

    # ------------------------------------------------------------------ #
    # Step 4 — mass-weighted centre of mass
    # ------------------------------------------------------------------ #
    weights = df["mass"].values
    x_com   = np.average(x_pc, weights=weights)
    y_com   = np.average(y_pc, weights=weights)
    z_com   = np.average(z_pc, weights=weights)

    # Convert CoM back to (l, b, d) for reporting
    from astropy.coordinates import Galactic as GalCoord
    com_galactic = GalCoord(
        l = np.arctan2(y_com, x_com) * u.rad,
        b = np.arctan2(z_com, np.sqrt(x_com**2 + y_com**2)) * u.rad,
        distance = np.sqrt(x_com**2 + y_com**2 + z_com**2) * u.pc,
    )
    com_icrs = com_galactic.icrs

    result = {
        "ra_com":           float(com_icrs.ra.deg),
        "dec_com":          float(com_icrs.dec.deg),
        "dist_com_pc":      float(com_icrs.distance.to(u.pc).value),
        "l_com_deg":        float(com_galactic.l.deg),
        "b_com_deg":        float(com_galactic.b.deg),
        "x_com_pc":         x_com,
        "y_com_pc":         y_com,
        "z_com_pc":         z_com,
        "total_mass_msun":  total_mass,
        "n_stars":          n,
        "masses":           df["mass"].values,
        "abs_g":            df["abs_g"].values,
    }

    log.info(
        "Cluster CoM:  RA=%.3f°, Dec=%.3f°, d=%.1f pc",
        result["ra_com"], result["dec_com"], result["dist_com_pc"],
    )
    return result


# =========================================================================== #
#  FUNCTION 2 — compute_tidal_radius
# =========================================================================== #
def compute_tidal_radius(
    total_mass_msun: float,
    A_oort: float = A_OORT,
    B_oort: float = B_OORT,
) -> dict:
    """
    Compute the cluster tidal (Jacobi) radius using Galactic Oort constants.

    The Jacobi radius formula (*) from Pinfield et al. (1998):

        r_t  =  [ G M_cl / (4 A |A − B|) ]^(1/3)

    with A and B in km s⁻¹ kpc⁻¹ and M_cl in M☉.

    Full derivation note
    --------------------
    In the rotating frame co-rotating with the cluster, the effective
    potential has saddle points (Lagrange L1/L2) at distance r_t from the
    cluster centre along the Galactocentric axis.  Setting the gradient of
    Φ_eff = 0 and using the epicycle approximation for Φ_Galaxy gives (*).

    Parameters
    ----------
    total_mass_msun : float
        Total cluster mass [M☉] estimated from Module 3, Function 1.
    A_oort, B_oort : float
        Oort constants [km s⁻¹ kpc⁻¹].

    Returns
    -------
    dict with keys:
        r_t_pc         — tidal radius in parsecs
        r_t_deg        — tidal radius as projected angle at cluster distance
        M_cl_msun      — input mass
        A_oort, B_oort — Oort constants used
    """
    # ------------------------------------------------------------------ #
    # Unit conversion:
    #   G in (km/s)² · pc · M☉⁻¹:
    #   G = 6.674e-11 m³ kg⁻¹ s⁻²
    #     = 4.3009e-3 pc M☉⁻¹ (km/s)²    [standard astrophysical unit]
    # ------------------------------------------------------------------ #
    G_pc_msun = 4.3009e-3   # pc (km/s)² M☉⁻¹

    # Convert Oort constants from km s⁻¹ kpc⁻¹ → km s⁻¹ pc⁻¹
    A_kms_pc  = A_oort / 1000.0   # km s⁻¹ pc⁻¹
    B_kms_pc  = B_oort / 1000.0   # km s⁻¹ pc⁻¹

    # Denominator: 4 A |A − B|  in (km/s)² pc⁻²
    denom = 4.0 * A_kms_pc * abs(A_kms_pc - B_kms_pc)   # (km/s)² pc⁻²

    # r_t in pc
    r_t_pc = (G_pc_msun * total_mass_msun / denom) ** (1.0 / 3.0)

    # Convert to angular size (arcmin) at Hyades mean distance (46.5 pc)
    hyades_dist_pc = 46.5
    r_t_deg = np.rad2deg(np.arctan2(r_t_pc, hyades_dist_pc))

    result = {
        "r_t_pc":         r_t_pc,
        "r_t_deg":        r_t_deg,
        "r_t_arcmin":     r_t_deg * 60.0,
        "M_cl_msun":      total_mass_msun,
        "A_oort":         A_oort,
        "B_oort":         B_oort,
    }

    log.info(
        "Tidal radius: r_t = %.2f pc  (%.2f° = %.1f arcmin at %.1f pc)",
        r_t_pc, r_t_deg, r_t_deg * 60, hyades_dist_pc,
    )
    log.info(
        "  Assumptions: G M / (4A|A-B|) with A=%.2f, B=%.2f km/s/kpc, "
        "M_cl=%.1f M☉",
        A_oort, B_oort, total_mass_msun,
    )
    return result


# =========================================================================== #
#  FUNCTION 3 — radial_mass_profile
# =========================================================================== #
def radial_mass_profile(
    members_df: pd.DataFrame,
    com: dict,
    n_bins: int = 15,
) -> pd.DataFrame:
    """
    Compute the radial mass profile (M enclosed vs radius) relative to
    the cluster centre of mass.  Useful for identifying the tidal boundary
    observationally.

    Parameters
    ----------
    members_df : pd.DataFrame
        Member stars (must have ra, dec, parallax, phot_g_mean_mag).
    com : dict
        Output of compute_cluster_centre_of_mass().
    n_bins : int
        Number of radial bins.

    Returns
    -------
    pd.DataFrame
        Columns: r_pc, M_enclosed_msun, N_enclosed, surface_density
    """
    df = members_df.copy()
    df = df[(df["parallax"] > 0) & df["phot_g_mean_mag"].notna()].copy()
    df["dist_pc"] = 1000.0 / df["parallax"]
    df["abs_g"]   = df["phot_g_mean_mag"] - 5.0 * np.log10(df["dist_pc"] / 10.0)
    df["mass"]    = mass_from_absolute_magnitude(df["abs_g"].values)

    coords = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        distance=df["dist_pc"].values * u.pc,
        frame="icrs",
    )
    com_coord = SkyCoord(
        ra=com["ra_com"] * u.deg,
        dec=com["dec_com"] * u.deg,
        distance=com["dist_com_pc"] * u.pc,
        frame="icrs",
    )
    # 3-D separations in pc
    sep_3d = coords.separation_3d(com_coord).to(u.pc).value

    r_max  = sep_3d.max() * 1.05
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_mid  = 0.5 * (r_bins[:-1] + r_bins[1:])

    M_enc = np.zeros(n_bins)
    N_enc = np.zeros(n_bins, dtype=int)
    for k, r_lo, r_hi in zip(range(n_bins), r_bins[:-1], r_bins[1:]):
        inside = sep_3d < r_hi
        M_enc[k] = df.loc[inside, "mass"].sum()
        N_enc[k] = int(inside.sum())

    shell_vol = (4.0 / 3.0) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    dm_shell  = np.diff(np.concatenate([[0], M_enc]))   # mass in shell

    profile_df = pd.DataFrame({
        "r_pc":               r_mid,
        "M_enclosed_msun":    M_enc,
        "N_enclosed":         N_enc,
        "mass_density_msun_pc3": dm_shell / shell_vol,
    })
    return profile_df


# =========================================================================== #
#  CONVENIENCE WRAPPER
# =========================================================================== #
def run_spatial_dynamics(members_df: pd.DataFrame) -> dict:
    """
    Full Module 3 pipeline.

    Returns a results dict with keys:
        com        — centre-of-mass dict
        tidal      — tidal radius dict
        profile    — radial mass profile DataFrame
    """
    com    = compute_cluster_centre_of_mass(members_df)
    tidal  = compute_tidal_radius(com["total_mass_msun"])
    profile = radial_mass_profile(members_df, com)

    log.info("Module 3 complete.")
    log.info("  CoM: (RA=%.3f°, Dec=%.3f°, d=%.1f pc)",
             com["ra_com"], com["dec_com"], com["dist_com_pc"])
    log.info("  Total mass: %.1f M☉", com["total_mass_msun"])
    log.info("  Tidal radius: %.2f pc", tidal["r_t_pc"])

    return {"com": com, "tidal": tidal, "profile": profile}


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Synthetic members for smoke-test
    rng = np.random.default_rng(42)
    N = 100
    mock_members = pd.DataFrame({
        "ra":              rng.normal(66.75, 3.0, N),
        "dec":             rng.normal(15.87, 3.0, N),
        "parallax":        rng.normal(21.5, 0.8, N),
        "parallax_error":  rng.uniform(0.02, 0.1, N),
        "phot_g_mean_mag": rng.uniform(6.5, 12.0, N),
        "bp_rp":           rng.uniform(0.4, 2.0, N),
    })

    results = run_spatial_dynamics(mock_members)
    print("\nCentre of mass:")
    for k, v in results["com"].items():
        if not isinstance(v, np.ndarray):
            print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nTidal radius: {results['tidal']['r_t_pc']:.2f} pc")
    print("\nRadial profile (first 5 bins):")
    print(results["profile"].head().to_string(index=False))
