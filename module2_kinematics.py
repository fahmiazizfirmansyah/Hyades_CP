"""
==============================================================================
MODULE 2: KINEMATICS & MEMBERSHIP SELECTION
==============================================================================
Perryman et al. (1997) Recreation Pipeline — Module 2

Purpose:
  Convert Gaia astrometry into 3-D Galactic space velocities (U, V, W) and
  apply a statistically rigorous membership filter based on the Mahalanobis
  distance (χ² statistic) in velocity space.

╔══════════════════════════════════════════════════════════════════════════════╗
║              ASTROPHYSICS BACKGROUND — READ THIS FIRST                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. FROM OBSERVABLES TO SPACE VELOCITIES  (the "uvw_transform" step)        ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  Gaia gives us, per star:                                                    ║
║    α, δ   — sky position in ICRS (degrees)                                  ║
║    ω       — parallax in mas  →  distance d = 1 / (ω/1000)  pc              ║
║    μα*, μδ — proper motions in mas yr⁻¹  (μα* ≡ μα cos δ)                   ║
║    v_r     — radial velocity in km s⁻¹  (NOT always available in DR3!)      ║
║                                                                              ║
║  The transverse (tangential) velocity components in km s⁻¹ are:            ║
║                                                                              ║
║    v_α  =  A_v · μα* / ω    A_v = 4.74047 km s⁻¹ per (mas yr⁻¹ / mas)     ║
║    v_δ  =  A_v · μδ  / ω                                                    ║
║                                                                              ║
║  where A_v = 1 AU / 1 yr converted to km s⁻¹ =  4.74047 km s⁻¹.           ║
║  This constant arises because 1 pc × 1 mas yr⁻¹ = 4.74047 km s⁻¹.         ║
║                                                                              ║
║  The full 3-D heliocentric velocity vector (v_r, v_α, v_δ) is then rotated  ║
║  into the Galactic frame (U, V, W) via the matrix T, defined by the IAU     ║
║  1958 Galactic coordinate system (Johnson & Soderblom 1987, AJ 93, 864):   ║
║                                                                              ║
║    ⎡U⎤   ⎡−0.0548755604 −0.8734370902 −0.4838350155⎤ ⎡cos δ cos α  ⎤      ║
║    ⎢V⎥ = ⎢+0.4941094279 −0.4448296300 +0.7469822445⎥ ⎢cos δ sin α  ⎥ × v  ║
║    ⎣W⎦   ⎣−0.8676661490 −0.1980763734 +0.4559837762⎦ ⎣   sin δ     ⎦      ║
║                                                                              ║
║  where v = (v_r, v_α, v_δ) in the equatorial frame. astropy wraps this     ║
║  entire transformation natively via the Galactocentric frame.               ║
║                                                                              ║
║  Convention: U is positive toward the Galactic anti-centre, V is positive  ║
║  in the direction of Galactic rotation, W is positive toward the NGP.       ║
║                                                                              ║
║  2. HANDLING MISSING RADIAL VELOCITIES                                       ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  Gaia DR3 provides ground-based RVs for ~33 M stars; the Hyades region has  ║
║  good but incomplete coverage.  For stars lacking v_r we have two options:  ║
║                                                                              ║
║  (a) Moving-cluster method (Perryman 1997):  project the known cluster mean ║
║      velocity onto the star's line of sight.  Requires convergent-point     ║
║      knowledge (Sect. 4 of the paper).                                      ║
║                                                                              ║
║  (b) Marginalize: use only the 2-D proper-motion vector (μα*, μδ) and        ║
║      apply a 2-D Mahalanobis filter in the proper-motion / parallax space.  ║
║                                                                              ║
║  We implement (b) as default and flag (a) as an optional extension.         ║
║                                                                              ║
║  3. MEMBERSHIP STATISTIC — MAHALANOBIS DISTANCE                             ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  Suppose the cluster mean velocity is μ̄ = (Ū, V̄, W̄) and the velocity       ║
║  covariance of a candidate star is Σ_i (3×3 matrix, see below).             ║
║                                                                              ║
║  The Mahalanobis distance (Dempster 1969; see Perryman 1997 §3.2) is:       ║
║                                                                              ║
║    D_M² = (v_i − μ̄)ᵀ · Σ_total⁻¹ · (v_i − μ̄)                              ║
║                                                                              ║
║  where  Σ_total = Σ_i + Σ_cluster                                           ║
║         Σ_i         = propagated observational error covariance for star i  ║
║         Σ_cluster   = intrinsic velocity dispersion matrix of the cluster   ║
║                       (σ_int ≈ 0.3 km s⁻¹ for the Hyades; isotropic)       ║
║                                                                              ║
║  Under the null hypothesis (star belongs to cluster), D_M² follows a χ²    ║
║  distribution with 3 degrees of freedom (or 2 if v_r is missing).          ║
║                                                                              ║
║  Threshold: χ²(3, 0.9973) ≈ 11.83 corresponds to a 3-sigma cut             ║
║             χ²(2, 0.9545) ≈  6.18 corresponds to 2-sigma (2-D case)        ║
║                                                                              ║
║  4. ERROR PROPAGATION FOR Σ_i                                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  The velocity errors are propagated from Gaia's observational errors via    ║
║  the Jacobian J of the transformation (v_r, v_α, v_δ) → (U, V, W):         ║
║                                                                              ║
║    Σ_UVW = J · Σ_obs · Jᵀ                                                   ║
║                                                                              ║
║  where Σ_obs = diag(σ_vr², σ_vα², σ_vδ²)  (cross-terms small; neglected)  ║
║  and J = ∂(U,V,W)/∂(v_r, v_α, v_δ) is the 3×3 rotation matrix T shown      ║
║  above.  Because T is orthogonal (Tᵀ = T⁻¹), in the uncorrelated case      ║
║  the UVW total variance is simply the quadrature sum of the input errors    ║
║  projected through T.                                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies:
  pip install astropy numpy pandas scipy
"""

import logging
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

from astropy.coordinates import (
    SkyCoord, Galactocentric, ICRS, Galactic,
    UnitSphericalRepresentation,
    SphericalRepresentation,
    CartesianRepresentation,
)
from astropy.coordinates import galactocentric_frame_defaults
import astropy.units as u

# --------------------------------------------------------------------------- #
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Physical / statistical constants ──────────────────────────────────────── #

# Conversion factor:  1 AU yr⁻¹ = 4.74047 km s⁻¹
#   (used to convert proper motion × distance → transverse velocity)
A_V = 4.74047  # km s⁻¹ per (mas yr⁻¹ · pc)

# Intrinsic 1-D velocity dispersion of the Hyades (isotropic assumption)
#   Perryman et al. (1997) Table 3: σ_int ≈ 0.3 km s⁻¹
SIGMA_INT_KMS = 0.3

# χ² thresholds for membership (see notes above)
CHI2_THRESH_3D = chi2.ppf(0.9973, df=3)   # ≈ 11.83  (3-sigma, 3 DOF)
CHI2_THRESH_2D = chi2.ppf(0.9545, df=2)   # ≈  6.18  (2-sigma, 2 DOF)

# Hyades mean heliocentric UVW from Perryman (1997) as first-guess seed
HYADES_UVW_SEED = np.array([-6.32, 45.24, 5.30])  # km s⁻¹, (U, V, W)


# =========================================================================== #
#  FUNCTION 1 — compute_space_velocities
# =========================================================================== #
def compute_space_velocities(
    df: pd.DataFrame,
    rv_fill: float | None = None,
) -> pd.DataFrame:
    """
    Convert Gaia astrometry into 3-D Galactic space velocities (U, V, W)
    and their propagated uncertainties for each star.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ra, dec, parallax, parallax_error,
                      pmra, pmra_error, pmdec, pmdec_error,
                      radial_velocity (optional), radial_velocity_error.
    rv_fill : float or None
        If provided, substitute this constant v_r [km s⁻¹] for stars
        lacking a measured radial velocity (option-a imputation).
        If None (default), stars without v_r are assigned NaN velocities
        and receive the 'rv_missing' flag; they are later handled in 2-D.

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with columns:
            dist_pc              — heliocentric distance in parsecs
            U, V, W              — Galactic space velocities in km s⁻¹
            sigma_U, sigma_V, sigma_W — propagated 1-σ uncertainties
            rv_missing           — boolean flag for imputed/missing v_r
            vel_total            — |v| in km s⁻¹ (sanity check)
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # Distance from parallax:  d [pc] = 1000 / ω [mas]
    # Only valid for positive parallax; negative values indicate noise.
    # ------------------------------------------------------------------ #
    with np.errstate(divide="ignore", invalid="ignore"):
        df["dist_pc"] = np.where(
            df["parallax"] > 0,
            1000.0 / df["parallax"],
            np.nan,
        )

    log.info(
        "Distance range for candidates: %.1f – %.1f pc",
        df["dist_pc"].min(), df["dist_pc"].max(),
    )

    # ------------------------------------------------------------------ #
    # Handle missing radial velocities
    # ------------------------------------------------------------------ #
    df["rv_missing"] = df["radial_velocity"].isna()
    n_missing = df["rv_missing"].sum()
    log.info(
        "%d / %d stars lack a radial velocity measurement.",
        n_missing, len(df),
    )

    if rv_fill is not None:
        log.info("  Imputing missing v_r = %.2f km s⁻¹.", rv_fill)
        df["radial_velocity"]       = df["radial_velocity"].fillna(rv_fill)
        df["radial_velocity_error"] = df["radial_velocity_error"].fillna(1.0)

    # ------------------------------------------------------------------ #
    # Compute transverse velocities in km s⁻¹
    #   v_α [km/s] = A_v · μα* [mas/yr] · d [pc] / 1000
    #   (μα* is already the cos-δ-corrected proper motion in Gaia DR3)
    # ------------------------------------------------------------------ #
    df["vt_ra"]  = A_V * df["pmra"]  * df["dist_pc"] / 1000.0  # really A_V * μ / (1000/d)
    # Correction: v_t = A_V * μ [mas/yr] * d [pc]  →  but A_V is defined as
    #   A_V = 4.74047 km/s · pc / (mas yr⁻¹ · pc) so actually:
    #   v_t [km/s] = A_V * μ [mas/yr] * d [pc]  ... let me compute correctly:
    # Actually the relation is v_t [km/s] = 4.74047 * mu [mas/yr] / parallax[mas]
    df["vt_ra"]  = A_V * df["pmra"]  / (df["parallax"] / 1000.0)
    df["vt_dec"] = A_V * df["pmdec"] / (df["parallax"] / 1000.0)

    # ------------------------------------------------------------------ #
    # Use astropy's full transformation for stars with v_r
    # ------------------------------------------------------------------ #
    # Arrays for vectorised astropy calculation
    has_rv = df["radial_velocity"].notna()

    # Initialise output columns
    for col in ["U", "V", "W", "sigma_U", "sigma_V", "sigma_W"]:
        df[col] = np.nan

    if has_rv.any():
        sub = df[has_rv].copy()

        # Build SkyCoord with full 6-D phase space
        coords_6d = SkyCoord(
            ra              = sub["ra"].values       * u.deg,
            dec             = sub["dec"].values      * u.deg,
            distance        = sub["dist_pc"].values  * u.pc,
            pm_ra_cosdec    = sub["pmra"].values     * u.mas / u.yr,
            pm_dec          = sub["pmdec"].values    * u.mas / u.yr,
            radial_velocity = sub["radial_velocity"].values * u.km / u.s,
            frame           = "icrs",
        )

        # Set default Galactocentric parameters (Gravity Collaboration 2019):
        #   R_sun = 8.122 kpc, z_sun = 20.8 pc, v_sun = (12.9, 245.6, 7.78) km/s
        galactocentric_frame_defaults.set("v4.0")
        gc = coords_6d.galactocentric

        # Extract Cartesian velocities — astropy returns them as Quantity arrays
        df.loc[has_rv, "U"] = gc.v_x.to(u.km / u.s).value
        df.loc[has_rv, "V"] = gc.v_y.to(u.km / u.s).value
        df.loc[has_rv, "W"] = gc.v_z.to(u.km / u.s).value

        # ---------------------------------------------------------- #
        # Propagate velocity errors (Monte Carlo or analytic)
        # Here: analytic linearised propagation assuming uncorrelated
        # observational errors.  Full covariance would require the
        # DR3 covariance matrices (available but complex).
        # ---------------------------------------------------------- #
        sigma_vr  = sub["radial_velocity_error"].fillna(1.0).values   # km/s
        sigma_mra = sub["pmra_error"].values                           # mas/yr
        sigma_mdec= sub["pmdec_error"].values                          # mas/yr
        sigma_plx = sub["parallax_error"].values                       # mas
        plx       = sub["parallax"].values                             # mas

        # Transverse velocity errors (km/s):
        #   σ(v_t_α) = A_V * σ(μα*) / ω  ⊕  A_V * μα* * σ(ω) / ω²
        sigma_vt_ra  = np.hypot(
            A_V * sigma_mra  / plx,
            A_V * sub["pmra"].values  * sigma_plx / plx**2,
        )
        sigma_vt_dec = np.hypot(
            A_V * sigma_mdec / plx,
            A_V * sub["pmdec"].values * sigma_plx / plx**2,
        )

        # Rotate errors through the ICRS → Galactic rotation matrix T.
        # T is orthogonal  ⟹  σ²(U,V,W) = T² · σ²(v_r, vt_α, vt_δ)
        # (diagonal input covariance; off-diagonal terms neglected here)
        T = _johnson_soderblom_matrix(
            sub["ra"].values, sub["dec"].values
        )  # shape (N, 3, 3)

        sigma_obs_sq = np.column_stack([
            sigma_vr**2, sigma_vt_ra**2, sigma_vt_dec**2
        ])  # (N, 3)

        # Vectorised: σ²(UVW)_i = Σ_j  T_ij²  σ²(obs)_j
        sigma_uvw_sq = np.einsum("nij,nj->ni", T**2, sigma_obs_sq)  # (N, 3)

        df.loc[has_rv, "sigma_U"] = np.sqrt(sigma_uvw_sq[:, 0])
        df.loc[has_rv, "sigma_V"] = np.sqrt(sigma_uvw_sq[:, 1])
        df.loc[has_rv, "sigma_W"] = np.sqrt(sigma_uvw_sq[:, 2])

    df["vel_total"] = np.sqrt(df["U"]**2 + df["V"]**2 + df["W"]**2)

    log.info(
        "UVW computed for %d stars with RV; %d stars flagged rv_missing.",
        has_rv.sum(), (~has_rv).sum(),
    )
    return df


# =========================================================================== #
#  FUNCTION 2 — membership_selection
# =========================================================================== #
def membership_selection(
    df: pd.DataFrame,
    sigma_threshold: float = 3.0,
    n_iterations: int = 5,
) -> pd.DataFrame:
    """
    Kinematic membership selection via iterative Mahalanobis distance filter.

    Algorithm
    ---------
    1. Seed the cluster mean velocity with the known Hyades UVW.
    2. For each star with full 3-D velocity:
         a. Construct the total covariance  Σ_total = Σ_i + Σ_cluster
         b. Compute D_M² = (v_i − μ̄)ᵀ Σ_total⁻¹ (v_i − μ̄)
         c. Reject if D_M² > χ²(3, p) where p = Φ(sigma_threshold)³
    3. Recompute cluster mean from surviving members; repeat until convergence.
    4. Stars lacking v_r are evaluated in 2-D (proper motion only).

    Parameters
    ----------
    df : pd.DataFrame
        Output of compute_space_velocities().
    sigma_threshold : float
        Number of equivalent Gaussian sigmas for the χ² cut (default 3).
    n_iterations : int
        Number of iterative rejection rounds.

    Returns
    -------
    pd.DataFrame
        Extended with columns:
            chi2_3d        — Mahalanobis χ² for stars with full UVW
            chi2_2d        — Mahalanobis χ² in proper-motion space (no v_r)
            is_member      — boolean membership flag
            membership_prob— approximate membership probability from χ²
    """
    df = df.copy()
    df["chi2_3d"]         = np.nan
    df["chi2_2d"]         = np.nan
    df["is_member"]       = False
    df["membership_prob"] = 0.0

    # χ² threshold for this sigma level  (3-D)
    p_thresh = chi2.cdf(sigma_threshold**2, df=1)  # ≈ 0.9973 for 3σ
    chi2_cut_3d = chi2.ppf(p_thresh**3, df=3)       # ~11.83
    chi2_cut_2d = chi2.ppf(p_thresh**2, df=2)       # ~11.36 at 3-sigma 2D

    log.info(
        "Membership χ² cuts — 3-D: %.2f, 2-D: %.2f  (%s-σ)",
        chi2_cut_3d, chi2_cut_2d, sigma_threshold,
    )

    # ------------------------------------------------------------------ #
    # Stage A: 3-D filter for stars with measured radial velocities
    # ------------------------------------------------------------------ #
    mask_3d = df["U"].notna() & df["V"].notna() & df["W"].notna()
    log.info("Stars with full 3-D velocities: %d", mask_3d.sum())

    if mask_3d.sum() < 10:
        log.warning("Too few stars with 3-D velocities; skipping 3-D filter.")
    else:
        sub3d = df[mask_3d].copy()
        mean_uvw = HYADES_UVW_SEED.copy()   # initial seed

        for iteration in range(n_iterations):
            velocities = sub3d[["U", "V", "W"]].values  # (N, 3)
            delta_v    = velocities - mean_uvw           # (N, 3)

            # Intrinsic cluster covariance (isotropic)
            Sigma_cluster = np.eye(3) * SIGMA_INT_KMS**2

            chi2_vals = np.full(len(sub3d), np.nan)

            for i, (idx, row) in enumerate(sub3d.iterrows()):
                # Per-star observational error covariance (diagonal)
                Sigma_i = np.diag([
                    row["sigma_U"]**2 if not np.isnan(row["sigma_U"]) else 1.0,
                    row["sigma_V"]**2 if not np.isnan(row["sigma_V"]) else 1.0,
                    row["sigma_W"]**2 if not np.isnan(row["sigma_W"]) else 1.0,
                ])
                Sigma_total = Sigma_i + Sigma_cluster

                try:
                    # Mahalanobis distance squared
                    # D_M² = Δv · Σ⁻¹ · Δv
                    Sigma_inv = np.linalg.inv(Sigma_total)
                    dv = delta_v[i]
                    chi2_vals[i] = float(dv @ Sigma_inv @ dv)
                except np.linalg.LinAlgError:
                    chi2_vals[i] = np.inf

            sub3d["chi2_3d"] = chi2_vals
            # Keep only members for this iteration
            keep = chi2_vals < chi2_cut_3d
            log.info(
                "  Iter %d: %d / %d stars survive 3-D χ² < %.2f",
                iteration + 1, keep.sum(), len(sub3d), chi2_cut_3d,
            )

            if keep.sum() < 5:
                log.warning("  Too few survivors; halting iteration early.")
                break

            # Recompute cluster mean from surviving members
            mean_uvw = sub3d.loc[keep, ["U", "V", "W"]].values.mean(axis=0)
            sub3d = sub3d[keep].copy()

        # Write results back
        df.loc[sub3d.index, "chi2_3d"]   = sub3d["chi2_3d"]
        df.loc[sub3d.index, "is_member"] = sub3d["chi2_3d"] < chi2_cut_3d
        df.loc[sub3d.index, "membership_prob"] = (
            1.0 - chi2.cdf(df.loc[sub3d.index, "chi2_3d"], df=3)
        )

        log.info(
            "Final 3-D members: %d  (mean UVW = [%.2f, %.2f, %.2f] km/s)",
            df["is_member"].sum(), *mean_uvw,
        )

    # ------------------------------------------------------------------ #
    # Stage B: 2-D filter for stars lacking radial velocities
    # We work in proper-motion space (μα*, μδ) after subtracting the
    # projection of the cluster mean proper motion.
    # ------------------------------------------------------------------ #
    mask_2d = df["rv_missing"] & df["parallax"].notna()
    log.info("Stars for 2-D proper-motion filter: %d", mask_2d.sum())

    if mask_2d.sum() > 0:
        sub2d = df[mask_2d].copy()

        # Compute expected proper motions from cluster mean UVW at each star's
        # position using the inverse of the UVW transformation.
        # Approximate: use the iteratively refined mean_uvw if available.
        try:
            mean_uvw_final = mean_uvw
        except NameError:
            mean_uvw_final = HYADES_UVW_SEED

        pm_expected = _expected_proper_motions(
            sub2d["ra"].values, sub2d["dec"].values,
            sub2d["parallax"].values, mean_uvw_final,
        )  # (N, 2): [μα*, μδ] in mas yr⁻¹

        delta_pm = sub2d[["pmra", "pmdec"]].values - pm_expected  # (N, 2)

        Sigma_cl_2d = np.eye(2) * (SIGMA_INT_KMS / A_V)**2  # intrinsic in pm space

        chi2_2d_vals = np.full(len(sub2d), np.nan)
        for i, (idx, row) in enumerate(sub2d.iterrows()):
            Sigma_i_pm = np.diag([
                row["pmra_error"]**2,
                row["pmdec_error"]**2,
            ])
            Sigma_total_2d = Sigma_i_pm + Sigma_cl_2d
            try:
                Sigma_inv_2d  = np.linalg.inv(Sigma_total_2d)
                dpm = delta_pm[i]
                chi2_2d_vals[i] = float(dpm @ Sigma_inv_2d @ dpm)
            except np.linalg.LinAlgError:
                chi2_2d_vals[i] = np.inf

        df.loc[sub2d.index, "chi2_2d"]   = chi2_2d_vals
        member_2d = chi2_2d_vals < chi2_cut_2d
        df.loc[sub2d.index, "is_member"] = (
            df.loc[sub2d.index, "is_member"] | member_2d
        )
        df.loc[sub2d.index, "membership_prob"] = (
            1.0 - chi2.cdf(chi2_2d_vals, df=2)
        )
        log.info(
            "2-D proper-motion members (no v_r): %d", member_2d.sum()
        )

    total_members = df["is_member"].sum()
    log.info("Total selected members: %d", total_members)
    return df


# =========================================================================== #
#  INTERNAL HELPERS
# =========================================================================== #
def _johnson_soderblom_matrix(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """
    Compute the 3×3 rotation matrix T from Johnson & Soderblom (1987)
    that transforms heliocentric (v_r, v_α, v_δ) to Galactic (U, V, W).

    Returns shape (N, 3, 3).
    """
    # Galactic pole direction in ICRS (J2000):
    #   α_NGP = 192.85948°, δ_NGP = +27.12825°  (Reid & Brunthaler 2004)
    # The full T matrix for each star depends on (α, δ).

    ra  = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    N   = len(ra)

    # Standard rotation matrix from J&S 1987 (Eq. A1–A3)
    # T = A · R  where A is the standard Galactic transformation matrix
    # and R depends on (α, δ).
    # Row 1: U-axis  (toward Galactic anti-centre in J&S sign convention)
    A = np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [+0.4941094279, -0.4448296300, +0.7469822445],
        [-0.8676661490, -0.1980763734, +0.4559837762],
    ])

    cos_ra, sin_ra   = np.cos(ra),  np.sin(ra)
    cos_dec, sin_dec = np.cos(dec), np.sin(dec)

    # R: equatorial → spherical unit vectors at each (α, δ)
    # Columns: [r̂, α̂, δ̂] evaluated at each star position
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] =  cos_dec * cos_ra    # r̂_x
    R[:, 0, 1] =  cos_dec * sin_ra    # r̂_y
    R[:, 0, 2] =  sin_dec             # r̂_z
    R[:, 1, 0] = -sin_ra              # α̂_x
    R[:, 1, 1] =  cos_ra              # α̂_y
    R[:, 1, 2] =  0.0                 # α̂_z
    R[:, 2, 0] = -sin_dec * cos_ra    # δ̂_x
    R[:, 2, 1] = -sin_dec * sin_ra    # δ̂_y
    R[:, 2, 2] =  cos_dec             # δ̂_z

    # T_i = A @ R_i  →  shape (N, 3, 3)
    T = np.einsum("ij,njk->nik", A, R)
    return T


def _expected_proper_motions(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    parallax_mas: np.ndarray,
    uvw: np.ndarray,
) -> np.ndarray:
    """
    Given cluster mean UVW [km s⁻¹] and star positions, compute the
    expected (μα*, μδ) proper motions in mas yr⁻¹.

    This is the inverse of the Johnson-Soderblom transform:
        v_equatorial = Tᵀ · UVW
    Then extract the transverse components and convert to mas yr⁻¹.
    """
    T = _johnson_soderblom_matrix(ra_deg, dec_deg)  # (N, 3, 3)
    T_T = np.transpose(T, axes=(0, 2, 1))           # (N, 3, 3) transposed

    # v_equatorial = Tᵀ · UVW  →  (N, 3)
    v_eq = np.einsum("nij,j->ni", T_T, uvw)

    # v_eq[:, 1] = v_α (km/s), v_eq[:, 2] = v_δ (km/s)
    # μα* [mas/yr] = v_α / A_V * parallax_mas / 1000
    # Actually: v_t [km/s] = A_V * μ / plx  →  μ = v_t * plx / A_V
    # but plx in mas and A_V in km/s per (mas/yr / mas) so:
    # μ [mas/yr] = v_t [km/s] / A_V * plx [mas]
    # Wait - let's be careful:
    # v_t = A_V * mu / omega  where omega in mas, mu in mas/yr, v_t in km/s
    # => mu = v_t * omega / A_V
    mu_ra   = v_eq[:, 1] * parallax_mas / A_V   # mas yr⁻¹
    mu_dec  = v_eq[:, 2] * parallax_mas / A_V   # mas yr⁻¹
    return np.column_stack([mu_ra, mu_dec])


# =========================================================================== #
#  CONVENIENCE WRAPPER
# =========================================================================== #
def run_kinematics_pipeline(
    df: pd.DataFrame,
    rv_fill: float | None = None,
    sigma_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Full Module 2 pipeline:
        Gaia candidates  →  space velocities  →  membership selection.

    Returns DataFrame with all membership columns appended.
    """
    df = compute_space_velocities(df, rv_fill=rv_fill)
    df = membership_selection(df, sigma_threshold=sigma_threshold)
    members = df[df["is_member"]].copy()
    log.info(
        "Module 2 complete.  %d / %d stars selected as Hyades members.",
        len(members), len(df),
    )
    return df   # return full table; caller can filter by is_member


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # --- Quick smoke-test with a synthetic mini-dataset -------------------- #
    rng = np.random.default_rng(42)
    N   = 200
    mock = pd.DataFrame({
        "ra":                   rng.uniform(55, 80, N),
        "dec":                  rng.uniform(10, 25, N),
        "parallax":             rng.normal(21.5, 1.5, N),
        "parallax_error":       rng.uniform(0.02, 0.1, N),
        "pmra":                 rng.normal(102.0, 2.0, N),   # Hyades-like
        "pmra_error":           rng.uniform(0.05, 0.5, N),
        "pmdec":                rng.normal(-27.5, 2.0, N),
        "pmdec_error":          rng.uniform(0.05, 0.5, N),
        "radial_velocity":      np.where(rng.random(N) > 0.35,
                                         rng.normal(39.4, 0.5, N), np.nan),
        "radial_velocity_error":rng.uniform(0.3, 2.0, N),
        "phot_g_mean_mag":      rng.uniform(6, 14, N),
        "phot_bp_mean_mag":     rng.uniform(7, 15, N),
        "phot_rp_mean_mag":     rng.uniform(5, 13, N),
        "bp_rp":                rng.uniform(0.3, 2.5, N),
    })

    result = run_kinematics_pipeline(mock)
    members = result[result["is_member"]]
    print(f"\nMembers selected: {len(members)} / {len(result)}")
    print(members[["ra", "dec", "parallax", "U", "V", "W",
                   "chi2_3d", "is_member"]].head(10).to_string())
