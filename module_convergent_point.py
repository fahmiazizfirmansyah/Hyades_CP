"""
==============================================================================
MODULE: CONVERGENT POINT (CP) METHOD & KINEMATIC PARALLAX
==============================================================================
Perryman et al. (1997) Recreation Pipeline — Convergent Point Module

Purpose:
  Implement the moving-cluster Convergent Point method to:
    (a) Find the CP via global minimisation of angular residuals Δθ
    (b) Compute kinematic parallax π_cp with rigorous error propagation
    (c) Apply iterative sigma-clipping (outlier rejection) during CP fitting

╔══════════════════════════════════════════════════════════════════════════════╗
║                   ASTROPHYSICS BACKGROUND                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MOVING-CLUSTER PRINCIPLE                                                    ║
║  ────────────────────────                                                    ║
║  All stars in a gravitationally unbound (freely drifting) cluster share      ║
║  the same 3-D space velocity vector V.  In projection on the sky, their      ║
║  proper-motion vectors appear to converge toward (or diverge from) a single  ║
║  point called the Convergent Point (CP).                                      ║
║                                                                              ║
║  For the Hyades, the CP lies at approximately α_cp ≈ 97.4°, δ_cp ≈ +6.9°   ║
║  (Perryman et al. 1997; van Leeuwen 2009).                                   ║
║                                                                              ║
║  SPHERICAL TRIGONOMETRY — POSITION ANGLES                                    ║
║  ──────────────────────────────────────────                                  ║
║  All angle calculations are performed rigorously on the unit sphere.         ║
║  No flat-sky (small-angle) approximations are made.                          ║
║                                                                              ║
║  For each star i at position (αᵢ, δᵢ):                                      ║
║                                                                              ║
║    θ_obs,i  — PA (N→E) of the observed proper-motion vector:                 ║
║                θ_obs = atan2(μα*, μδ)                                        ║
║                                                                              ║
║    θ_c,i   — PA of the great circle from star i toward the CP:              ║
║               (four-parts spherical trig formula)                            ║
║               tan θ_c = sin(Δα) / [cos(δᵢ)·tan(δ_cp) − sin(δᵢ)·cos(Δα)]  ║
║               where Δα = α_cp − αᵢ                                          ║
║                                                                              ║
║    Δθᵢ = θ_obs,i − θ_c,i      should → 0 for true cluster members          ║
║                                                                              ║
║  The great-circle distance λᵢ (star → CP) uses the Haversine formula:       ║
║    λᵢ = 2 arcsin(√[sin²(Δδ/2) + cos(δᵢ)cos(δ_cp)sin²(Δα/2)])             ║
║                                                                              ║
║  KINEMATIC PARALLAX                                                          ║
║  ─────────────────                                                            ║
║  For a moving-cluster star with space velocity magnitude V:                  ║
║    V_r,i = V · cos(λᵢ)        (line-of-sight component)                     ║
║    V_t,i = V · sin(λᵢ)        (transverse component)                        ║
║                                                                              ║
║  Dividing: V_t / V_r = tan(λ)                                               ║
║                                                                              ║
║  The transverse velocity in km s⁻¹ is related to observables by:            ║
║    V_t = A_v · μ / π_cp                                                      ║
║    where A_v = 4.74047 km s⁻¹ · mas / (mas yr⁻¹)  [= 1 AU/yr]             ║
║          μ = √(μα*² + μδ²)   total proper motion [mas yr⁻¹]                ║
║          π_cp                 kinematic parallax [mas]                        ║
║                                                                              ║
║  Combining:  A_v · μ / π_cp = V_r · tan(λ)                                  ║
║                                                                              ║
║                π_cp = A_v · μ / (V_r · tan λ)         [mas]     ← (*)      ║
║                                                                              ║
║  ⚠ NOTE: The formula is often written inverted by mistake.                  ║
║     Dimensional check: [4.74047 · mas/yr / (km/s)] = [mas] ✓               ║
║                                                                              ║
║  ERROR PROPAGATION  (rigorous first-order)                                   ║
║  ──────────────────────────────────────────                                  ║
║  From (*), π_cp is a product/quotient of μ and V_r, so in fractional form:  ║
║                                                                              ║
║    (σ_π / π)² = (σ_μ / μ)² + (σ_Vr / V_r)²                               ║
║                                                                              ║
║  where:                                                                      ║
║    σ_μ  = √[(μα* σ_μα*)² + (μδ σ_μδ)²] / μ     (propagated from Gaia)    ║
║    σ_Vr = Gaia RV measurement error                                          ║
║                                                                              ║
║  This is derived from ∂π/∂μ = A_v/(V_r·tan λ) = π/μ                        ║
║                        ∂π/∂V_r = −A_v·μ/(V_r²·tan λ) = −π/V_r            ║
║                                                                              ║
║  The contribution of uncertainty in λ (from CP position error σ_cp) is:     ║
║    ∂π/∂λ = −A_v·μ / (V_r · sin²λ) = −π · cos λ / sin λ                   ║
║  This term is included if σ_cp is provided.                                  ║
║                                                                              ║
║  GLOBAL OPTIMISATION — AVOIDING LOCAL MINIMA                                 ║
║  ─────────────────────────────────────────────                               ║
║  The cost function Q(α_cp, δ_cp) = Σ wᵢ Δθᵢ² is non-convex.               ║
║  We use a two-phase strategy:                                                ║
║    1. Differential Evolution (DE): stochastic global search over a          ║
║       bounded sky region; immune to local traps by construction.             ║
║    2. Nelder-Mead simplex: high-precision local refinement from DE result.  ║
║  DE with polish=True already does a final L-BFGS-B step, but the explicit   ║
║  Nelder-Mead pass squeezes out additional sub-arcsecond precision.          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

References:
  Perryman et al. 1997, A&A 331, 81
  van Leeuwen 2009, A&A 497, 209
  Smart 1968, Stellar Kinematics, Cambridge

Dependencies:
  pip install numpy scipy astropy pandas
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# A_v = 1 AU yr⁻¹ expressed in km s⁻¹.
# Derivation: 1 pc × 1 mas yr⁻¹ = 1 AU yr⁻¹ = 4.74047 km s⁻¹
# Used in:  V_t [km/s] = A_v × μ [mas/yr] / π [mas]
A_V = 4.74047   # km s⁻¹ per (mas yr⁻¹ / mas)

# Known Hyades Convergent Point from Perryman et al. (1997) Table 1.
# Used as the centre of the DE search region; not used as a fixed prior.
_CP_HYADES_RA_DEG  = 97.4
_CP_HYADES_DEC_DEG =  6.9

# Default search bounds for Differential Evolution (generous ±40° window)
_CP_BOUNDS_RA  = (50.0, 150.0)   # degrees
_CP_BOUNDS_DEC = (-20.0,  30.0)  # degrees


# ============================================================================
#  SECTION 1 — RIGOROUS SPHERICAL TRIGONOMETRY
# ============================================================================

def great_circle_distance_rad(
    ra1: np.ndarray, dec1: np.ndarray,
    ra2: float, dec2: float,
) -> np.ndarray:
    """
    Angular separation λ [radians] from N stars at (ra1, dec1) to a single
    point (ra2, dec2) via the Haversine formula.

    The Haversine formula is used instead of the naive
      cos(λ) = sin(δ₁)sin(δ₂) + cos(δ₁)cos(δ₂)cos(Δα)
    because the naive form suffers catastrophic cancellation for λ ≪ 1 rad.

    Parameters
    ----------
    ra1, dec1 : array_like, degrees
        Star equatorial coordinates (length N).
    ra2, dec2 : float, degrees
        Reference point (the Convergent Point).

    Returns
    -------
    lambda_rad : np.ndarray, shape (N,), radians
    """
    a1 = np.deg2rad(np.asarray(ra1,  dtype=float))
    d1 = np.deg2rad(np.asarray(dec1, dtype=float))
    a2 = np.deg2rad(float(ra2))
    d2 = np.deg2rad(float(dec2))

    dra  = a2 - a1
    ddec = d2 - d1

    # Haversine kernel: h = sin²(Δδ/2) + cos(δ₁)cos(δ₂)sin²(Δα/2)
    h = np.sin(ddec * 0.5)**2 + np.cos(d1) * np.cos(d2) * np.sin(dra * 0.5)**2
    h = np.clip(h, 0.0, 1.0)   # guard against floating-point overshoot
    return 2.0 * np.arcsin(np.sqrt(h))


def position_angle_to_cp_rad(
    ra_star: np.ndarray, dec_star: np.ndarray,
    ra_cp: float, dec_cp: float,
) -> np.ndarray:
    """
    Position angle θ_c [radians, N through E] of the great-circle arc
    from each star toward the Convergent Point.

    Derivation — four-parts formula (rigorous spherical trigonometry)
    ─────────────────────────────────────────────────────────────────
    On the unit sphere consider the spherical triangle formed by:
      • The North Celestial Pole (NP)
      • The star S at (αᵢ, δᵢ)
      • The Convergent Point C at (α_cp, δ_cp)

    The position angle θ_c is the interior angle at vertex S.
    Applying the four-parts (co-sides) formula to this triangle gives:

        tan(θ_c) = sin(Δα) / [cos(δ_S) tan(δ_C) − sin(δ_S) cos(Δα)]

    where Δα = α_cp − αᵢ.

    This is exact (no small-angle approximation) and returns θ_c in (−π, π]
    measured Eastward from North, consistent with the IAU position-angle
    convention and with θ_obs computed from atan2(μα*, μδ).

    Edge cases:
      • Δα → 0, δ_C > δ_S : θ_c → 0   (CP is due North)
      • Δα → 0, δ_C < δ_S : θ_c → ±π  (CP is due South)
      • δ_S → ±90°          : handled numerically by atan2

    Parameters
    ----------
    ra_star, dec_star : array_like, degrees
    ra_cp, dec_cp     : float, degrees

    Returns
    -------
    theta_c : np.ndarray, shape (N,), radians ∈ (−π, π]
    """
    a_s = np.deg2rad(np.asarray(ra_star,  dtype=float))
    d_s = np.deg2rad(np.asarray(dec_star, dtype=float))
    a_c = np.deg2rad(float(ra_cp))
    d_c = np.deg2rad(float(dec_cp))

    dra = a_c - a_s

    num = np.sin(dra)
    den = np.cos(d_s) * np.tan(d_c) - np.sin(d_s) * np.cos(dra)

    return np.arctan2(num, den)


def position_angle_proper_motion_rad(
    pmra: np.ndarray,
    pmdec: np.ndarray,
) -> np.ndarray:
    """
    Observed position angle θ_obs [radians] of the proper-motion vector.

    Convention: θ = 0 → North, θ = +π/2 → East (IAU standard).

    For the Gaia DR3 proper-motion components (μα* = μα cos δ, μδ):
        θ_obs = atan2(μα*, μδ)

    This is exact for any proper-motion magnitude and direction.
    Note that μα* is already the cos-δ-corrected component in Gaia DR3,
    so no additional projection factor is required.

    Parameters
    ----------
    pmra  : array_like  [mas yr⁻¹]   μα* (East component, cos-δ corrected)
    pmdec : array_like  [mas yr⁻¹]   μδ  (North component)

    Returns
    -------
    theta_obs : np.ndarray, shape (N,), radians ∈ (−π, π]
    """
    return np.arctan2(
        np.asarray(pmra,  dtype=float),
        np.asarray(pmdec, dtype=float),
    )


def angular_residuals_rad(
    ra: np.ndarray, dec: np.ndarray,
    pmra: np.ndarray, pmdec: np.ndarray,
    ra_cp: float, dec_cp: float,
) -> np.ndarray:
    """
    Δθᵢ = θ_obs,i − θ_c,i [radians], wrapped to (−π, π].

    A small |Δθ| means the star's proper-motion vector aligns with the
    great circle toward the CP, as expected for a true cluster member.

    Parameters
    ----------
    ra, dec : array_like, degrees
    pmra, pmdec : array_like, mas yr⁻¹
    ra_cp, dec_cp : float, degrees

    Returns
    -------
    delta : np.ndarray, shape (N,), radians
    """
    theta_obs = position_angle_proper_motion_rad(pmra, pmdec)
    theta_c   = position_angle_to_cp_rad(ra, dec, ra_cp, dec_cp)
    delta     = theta_obs - theta_c
    # Wrap to (−π, +π]
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def pm_angle_uncertainty_rad(
    pmra: np.ndarray, pmdec: np.ndarray,
    sigma_pmra: np.ndarray, sigma_pmdec: np.ndarray,
) -> np.ndarray:
    """
    1-σ uncertainty on θ_obs [radians] propagated from Gaia PM errors.

    Derivation (first-order propagation)
    ─────────────────────────────────────
    θ_obs = atan2(μα*, μδ),  μ² = μα*² + μδ²

    ∂θ/∂μα* =  μδ  / μ²
    ∂θ/∂μδ  = −μα* / μ²

    σ²(θ) = (μδ / μ²)² σ²(μα*) + (μα* / μ²)² σ²(μδ)
           = [(μδ σ_μα*)² + (μα* σ_μδ)²] / μ⁴

    For equal errors σ_μα* = σ_μδ = σ_μ  →  σ(θ) = σ_μ / μ  (classical form)

    Parameters
    ----------
    pmra, pmdec, sigma_pmra, sigma_pmdec : array_like, mas yr⁻¹

    Returns
    -------
    sigma_theta : np.ndarray, shape (N,), radians
    """
    pmra  = np.asarray(pmra,  dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    sp    = np.asarray(sigma_pmra,  dtype=float)
    sd    = np.asarray(sigma_pmdec, dtype=float)

    mu_sq = pmra**2 + pmdec**2
    mu_sq = np.where(mu_sq < 1e-12, 1e-12, mu_sq)   # guard against μ ≈ 0

    sigma_sq = ((pmdec * sp)**2 + (pmra * sd)**2) / mu_sq**2
    return np.sqrt(sigma_sq)


# ============================================================================
#  SECTION 2 — CONVERGENT POINT FINDER WITH ITERATIVE SIGMA-CLIPPING
# ============================================================================

def _cp_cost(
    params: np.ndarray,
    ra: np.ndarray, dec: np.ndarray,
    pmra: np.ndarray, pmdec: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted sum of squared angular residuals: Q = Σ wᵢ Δθᵢ².

    This is the objective function minimised to find the Convergent Point.

    Parameters
    ----------
    params : [ra_cp, dec_cp]  degrees
    ra, dec, pmra, pmdec : star data arrays
    weights : per-star weights wᵢ = 1 / σ²(θ_obs,i)

    Returns
    -------
    Q : float  (dimensionless, radians²)
    """
    ra_cp, dec_cp = params
    delta = angular_residuals_rad(ra, dec, pmra, pmdec, ra_cp, dec_cp)
    return float(np.sum(weights * delta**2))


def find_convergent_point(
    ra: np.ndarray,
    dec: np.ndarray,
    pmra: np.ndarray,
    pmdec: np.ndarray,
    sigma_pmra: np.ndarray,
    sigma_pmdec: np.ndarray,
    sigma_clip: float = 3.0,
    max_iter: int = 20,
    de_seed: int = 42,
    bounds_ra:  Tuple[float, float] = _CP_BOUNDS_RA,
    bounds_dec: Tuple[float, float] = _CP_BOUNDS_DEC,
) -> dict:
    """
    Find the Convergent Point (α_cp, δ_cp) via global optimisation with
    iterative sigma-clipping outlier rejection.

    Algorithm overview
    ------------------
    Each iteration consists of three sub-steps:

      A. Global minimum (Differential Evolution)
         DE samples (α_cp, δ_cp) stochastically across the full search region,
         making it immune to local minima of the non-convex cost function Q.
         The 'polish' flag performs an internal L-BFGS-B refinement after DE.

      B. Local refinement (Nelder-Mead)
         Starting from the DE solution, Nelder-Mead squeezes out sub-arcsecond
         precision without requiring gradient information (robust for the
         wrapping discontinuities in Δθ near ±π).

      C. Sigma-clipping
         Residuals Δθᵢ are computed for all currently active stars.
         Stars with |Δθᵢ| > sigma_clip × std(Δθ) are rejected.
         The loop repeats until no stars are rejected (convergence).

    Why Differential Evolution avoids local minima
    ───────────────────────────────────────────────
    The cost Q = Σ wᵢ Δθᵢ² can have multiple local minima because:
      • Δθ is a non-linear, trigonometric function of (α_cp, δ_cp)
      • The wrapping modulo 2π creates non-smooth ridges in Q
    DE explores the full bounded search region with a population of candidate
    solutions that mutate and cross-breed, preventing early convergence to a
    sub-optimal solution.  With popsize=20 and 2000 generations the probability
    of missing the global minimum is negligible for a 2-D problem of this kind.

    Parameters
    ----------
    ra, dec : array_like, degrees
        Equatorial coordinates of candidate stars.
    pmra, pmdec : array_like, mas yr⁻¹
        Proper-motion components (μα*, μδ).  μα* must be cos-δ corrected.
    sigma_pmra, sigma_pmdec : array_like, mas yr⁻¹
        Gaia 1-σ proper-motion uncertainties.
    sigma_clip : float
        Outlier rejection threshold in units of std(Δθ)  (default 3.0).
    max_iter : int
        Maximum number of sigma-clipping iterations (default 20).
    de_seed : int
        Random seed for Differential Evolution (reproducibility).
    bounds_ra, bounds_dec : tuple of (float, float)
        Search bounds in degrees.

    Returns
    -------
    dict:
        ra_cp, dec_cp         — best-fit CP coordinates [degrees]
        sigma_ra_cp, sigma_dec_cp — formal 1-σ uncertainties [degrees]
          (estimated from the numerical Hessian of Q at the minimum)
        n_stars_used          — stars remaining after outlier rejection
        n_rejected            — stars rejected
        n_iterations          — iterations actually performed
        mask_used             — bool array of length N: True = star kept
        residuals_deg         — Δθ [degrees] for kept stars (final iteration)
        cost_final            — value of Q at convergence
        cp_history            — list of (ra_cp, dec_cp) per iteration
    """
    ra    = np.asarray(ra,    dtype=float)
    dec   = np.asarray(dec,   dtype=float)
    pmra  = np.asarray(pmra,  dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    sp    = np.asarray(sigma_pmra,  dtype=float)
    sd    = np.asarray(sigma_pmdec, dtype=float)

    N_total = len(ra)
    log.info(
        "CP search: N=%d stars, sigma_clip=%.1f, max_iter=%d, "
        "RA bounds=%s, Dec bounds=%s",
        N_total, sigma_clip, max_iter, bounds_ra, bounds_dec,
    )

    mask       = np.ones(N_total, dtype=bool)   # starts with all stars active
    cp_history = []
    n_iter     = 0
    ra_cp_cur  = _CP_HYADES_RA_DEG    # will be updated each iteration
    dec_cp_cur = _CP_HYADES_DEC_DEG

    for iteration in range(max_iter):
        n_iter = iteration + 1
        idx    = np.where(mask)[0]
        n_act  = int(len(idx))

        if n_act < 10:
            log.warning(
                "Iter %d: only %d stars remain — stopping early.", n_iter, n_act
            )
            break

        # Slice active stars
        ra_a  = ra[idx];   dec_a = dec[idx]
        pm_a  = pmra[idx]; pd_a  = pmdec[idx]
        sp_a  = sp[idx];   sd_a  = sd[idx]

        # Per-star weights wᵢ = 1 / σ²(θ_obs)
        sigma_theta = pm_angle_uncertainty_rad(pm_a, pd_a, sp_a, sd_a)
        sigma_theta = np.where(sigma_theta < 1e-6, 1e-6, sigma_theta)
        weights     = 1.0 / sigma_theta**2

        # ── Phase A: Global minimum via Differential Evolution ──────────── #
        de_result = differential_evolution(
            _cp_cost,
            bounds=[bounds_ra, bounds_dec],
            args=(ra_a, dec_a, pm_a, pd_a, weights),
            seed=de_seed,
            strategy="best1bin",
            maxiter=2000,
            tol=1e-9,
            popsize=20,
            polish=True,      # internal L-BFGS-B polish step
            workers=1,
        )
        if not de_result.success:
            log.warning("DE did not fully converge at iteration %d.", n_iter)

        # ── Phase B: Local refinement with Nelder-Mead ───────────────────── #
        nm_result = minimize(
            _cp_cost,
            x0=de_result.x,
            args=(ra_a, dec_a, pm_a, pd_a, weights),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 100_000},
        )
        ra_cp_cur, dec_cp_cur = float(nm_result.x[0]), float(nm_result.x[1])
        cp_history.append((ra_cp_cur, dec_cp_cur))

        # ── Phase C: Sigma-clipping ───────────────────────────────────────── #
        delta     = angular_residuals_rad(ra_a, dec_a, pm_a, pd_a, ra_cp_cur, dec_cp_cur)
        std_delta = float(np.std(delta, ddof=1)) if n_act > 2 else 1.0
        outliers  = np.abs(delta) > sigma_clip * std_delta
        n_out     = int(outliers.sum())

        log.info(
            "  Iter %2d: CP=(RA=%.4f°, Dec=%.4f°),  std(Δθ)=%.4f°,  "
            "removed=%d / %d",
            n_iter, ra_cp_cur, dec_cp_cur,
            np.rad2deg(std_delta), n_out, n_act,
        )

        # ── Step 4: If no outliers, we have converged ─────────────────────── #
        if n_out == 0:
            log.info("  Sigma-clipping converged after %d iteration(s).", n_iter)
            break

        mask[idx[outliers]] = False   # reject outliers globally

    # ── Formal uncertainty from the numerical Hessian of Q ─────────────── #
    idx_f  = np.where(mask)[0]
    ra_f   = ra[idx_f];   dec_f = dec[idx_f]
    pm_f   = pmra[idx_f]; pd_f  = pmdec[idx_f]
    sp_f   = sp[idx_f];   sd_f  = sd[idx_f]

    sigma_theta_f = pm_angle_uncertainty_rad(pm_f, pd_f, sp_f, sd_f)
    sigma_theta_f = np.where(sigma_theta_f < 1e-6, 1e-6, sigma_theta_f)
    weights_f     = 1.0 / sigma_theta_f**2

    delta_f = angular_residuals_rad(ra_f, dec_f, pm_f, pd_f, ra_cp_cur, dec_cp_cur)
    Q_final = float(np.sum(weights_f * delta_f**2))

    sigma_ra_cp, sigma_dec_cp = _hessian_uncertainty(
        np.array([ra_cp_cur, dec_cp_cur]),
        ra_f, dec_f, pm_f, pd_f, weights_f,
        bounds_ra, bounds_dec,
    )

    log.info(
        "Final CP: RA = %.4f ± %.4f°,  Dec = %.4f ± %.4f°  "
        "| stars used=%d, rejected=%d",
        ra_cp_cur, sigma_ra_cp,
        dec_cp_cur, sigma_dec_cp,
        len(idx_f), N_total - len(idx_f),
    )

    return {
        "ra_cp":          ra_cp_cur,
        "dec_cp":         dec_cp_cur,
        "sigma_ra_cp":    sigma_ra_cp,
        "sigma_dec_cp":   sigma_dec_cp,
        "n_stars_used":   int(len(idx_f)),
        "n_rejected":     int(N_total - len(idx_f)),
        "n_iterations":   n_iter,
        "mask_used":      mask.copy(),
        "residuals_deg":  np.rad2deg(delta_f),
        "cost_final":     Q_final,
        "cp_history":     cp_history,
    }


def _hessian_uncertainty(
    x0: np.ndarray,
    ra: np.ndarray, dec: np.ndarray,
    pmra: np.ndarray, pmdec: np.ndarray,
    weights: np.ndarray,
    bounds_ra: Tuple[float, float],
    bounds_dec: Tuple[float, float],
    eps: float = 1e-4,
) -> Tuple[float, float]:
    """
    Estimate 1-σ uncertainties on (α_cp, δ_cp) from the numerical Hessian of Q.

    The curvature matrix H = ∂²Q/∂θ² evaluated at the minimum gives the
    inverse covariance: Cov(α_cp, δ_cp) ≈ H⁻¹.
    So σ²(α_cp) = (H⁻¹)₀₀ and σ²(δ_cp) = (H⁻¹)₁₁.

    The Hessian is estimated with the second-order finite-difference formula:
      ∂²Q/∂xᵢ∂xⱼ ≈ [Q(x+εᵢ+εⱼ) − Q(x+εᵢ) − Q(x+εⱼ) + Q(x)] / ε²

    Parameters
    ----------
    x0      : [ra_cp, dec_cp] at the minimum
    eps     : finite-difference step size [degrees]

    Returns
    -------
    sigma_ra, sigma_dec : float, degrees
    """
    try:
        f0 = _cp_cost(x0, ra, dec, pmra, pmdec, weights)
        H  = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                ei = np.zeros(2); ei[i] = eps
                ej = np.zeros(2); ej[j] = eps
                fij = _cp_cost(x0 + ei + ej, ra, dec, pmra, pmdec, weights)
                fi  = _cp_cost(x0 + ei,       ra, dec, pmra, pmdec, weights)
                fj  = _cp_cost(x0 + ej,       ra, dec, pmra, pmdec, weights)
                H[i, j] = (fij - fi - fj + f0) / (eps**2)

        cov = np.linalg.inv(H)
        sigma_ra  = float(np.sqrt(max(cov[0, 0], 0.0)))
        sigma_dec = float(np.sqrt(max(cov[1, 1], 0.0)))
        return sigma_ra, sigma_dec

    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        log.warning("Hessian inversion failed; uncertainties set to NaN.")
        return float("nan"), float("nan")


# ============================================================================
#  SECTION 3 — KINEMATIC PARALLAX WITH RIGOROUS ERROR PROPAGATION
# ============================================================================

def compute_kinematic_parallax(
    ra: np.ndarray,
    dec: np.ndarray,
    pmra: np.ndarray,
    pmdec: np.ndarray,
    sigma_pmra: np.ndarray,
    sigma_pmdec: np.ndarray,
    radial_velocity: np.ndarray,
    sigma_rv: np.ndarray,
    ra_cp: float,
    dec_cp: float,
    sigma_cp_deg: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute kinematic parallax π_cp [mas] and uncertainty σ(π_cp) per star.

    The Fundamental Formula (Perryman 1997, corrected sign)
    ───────────────────────────────────────────────────────
    From the moving-cluster geometry (see module docstring):

        π_cp = A_v · μ / (V_r · tan λ)         ... (*)

    where:
      A_v = 4.74047  [km s⁻¹ · mas · yr / mas]   (= 1 AU/yr in km/s units)
      μ   = √(μα*² + μδ²)                         [mas yr⁻¹]
      V_r = heliocentric radial velocity           [km s⁻¹]
      λ   = great-circle distance (star → CP)      [radians]

    ⚠  The equivalent formula  π = V_r·tan(λ) / (A_v·μ)  is WRONG.
       Dimensional analysis of (*): [km/s · mas·yr/mas · mas/yr / km/s] = [mas] ✓

    Full Error Propagation (first-order, partial derivatives)
    ──────────────────────────────────────────────────────────
    Let  f(μ, V_r, λ) = A_v · μ / (V_r · tan λ)  = π_cp

    Partial derivatives:
      ∂π/∂μ   =  A_v / (V_r · tan λ)        =  π_cp / μ
      ∂π/∂V_r = −A_v · μ / (V_r² · tan λ)   = −π_cp / V_r
      ∂π/∂λ   = −A_v · μ / (V_r · sin²λ)    = −π_cp · cos(λ) / sin(λ)

    Propagated uncertainty:
      σ²(π_cp) = (∂π/∂μ)² σ²(μ) + (∂π/∂V_r)² σ²(V_r) + (∂π/∂λ)² σ²(λ)

    where:
      σ²(μ)  = [(μα* σ_μα*)² + (μδ σ_μδ)²] / μ²     (from Gaia)
      σ²(V_r) = (Gaia RV error)²
      σ²(λ)   ≈ σ²_cp [degrees → radians]   (from CP position uncertainty)
                 omitted if sigma_cp_deg is None

    Geometric exclusion criteria
    ────────────────────────────
    Stars are excluded from kinematic-parallax computation if:
      • λ < 1°          (star is too close to CP; tan λ → 0 → π_cp → ∞)
      • |90° − λ| < 1°  (star is near the perpendicular; π_cp → 0 and
                          the method loses sensitivity)
      • |V_r| < 0.5 km/s (V_r too small; zero-crossing degeneracy)

    Parameters
    ----------
    ra, dec : array_like, degrees
    pmra, pmdec : array_like, mas yr⁻¹
    sigma_pmra, sigma_pmdec : array_like, mas yr⁻¹
    radial_velocity : array_like, km s⁻¹
    sigma_rv : array_like, km s⁻¹
    ra_cp, dec_cp : float, degrees
    sigma_cp_deg : float or None
        Combined 1-σ uncertainty on the CP position [degrees].
        If provided, the λ uncertainty term is included in σ(π_cp).
        Estimated as  σ_cp ≈ √(σ_α_cp² cos²δ_cp + σ_δ_cp²)  for the arc
        relevant to each star.  Pass None to omit this term.
    mask : array_like of bool or None
        Pre-filter mask (e.g., stars surviving sigma-clipping).
        If None, all stars with finite V_r are processed.

    Returns
    -------
    pd.DataFrame, one row per input star, columns:
        pi_cp         — kinematic parallax [mas]  (NaN if excluded)
        sigma_pi_cp   — 1-σ uncertainty [mas]
        lambda_deg    — great-circle distance star→CP [degrees]
        mu_total      — √(μα*² + μδ²)  [mas yr⁻¹]
        sigma_mu      — propagated proper-motion uncertainty [mas yr⁻¹]
        frac_mu       — σ_μ / μ  (fractional PM uncertainty)
        frac_rv       — σ_Vr / |V_r|  (fractional RV uncertainty)
        frac_lambda   — |∂π/∂λ · σ_λ| / π   (fractional λ uncertainty,
                         0 if sigma_cp_deg=None)
        geom_valid    — bool: star passes geometric exclusion criteria
        kinpar_valid  — bool: π_cp > 0 and all quantities finite
    """
    ra    = np.asarray(ra,    dtype=float)
    dec   = np.asarray(dec,   dtype=float)
    pmra  = np.asarray(pmra,  dtype=float)
    pmdec = np.asarray(pmdec, dtype=float)
    sp    = np.asarray(sigma_pmra,  dtype=float)
    sd    = np.asarray(sigma_pmdec, dtype=float)
    rv    = np.asarray(radial_velocity, dtype=float)
    srv   = np.asarray(sigma_rv,        dtype=float)

    N = len(ra)

    if mask is None:
        mask = np.isfinite(rv)
    mask = np.asarray(mask, dtype=bool)

    # ── λ: great-circle distance star → CP ──────────────────────────────── #
    lambda_rad = great_circle_distance_rad(ra, dec, ra_cp, dec_cp)
    lambda_deg = np.rad2deg(lambda_rad)

    # ── μ and σ(μ): total proper motion and propagated uncertainty ─────── #
    mu_total = np.sqrt(pmra**2 + pmdec**2)
    mu_safe  = np.where(mu_total < 1e-10, 1e-10, mu_total)

    # σ(μ) from ∂μ/∂μα* = μα*/μ,  ∂μ/∂μδ = μδ/μ
    sigma_mu = np.sqrt(
        (pmra  / mu_safe * sp)**2 +
        (pmdec / mu_safe * sd)**2
    )

    # ── Geometric exclusion mask ──────────────────────────────────────────── #
    geom_valid = (
        mask &
        np.isfinite(rv) &
        (np.abs(rv) >= 0.5) &              # V_r not too close to zero
        (lambda_deg >= 1.0) &              # not too close to CP
        (np.abs(90.0 - lambda_deg) >= 1.0) # not near perpendicular
    )

    # ── Initialise output arrays with NaN ─────────────────────────────────── #
    pi_cp       = np.full(N, np.nan)
    sigma_pi_cp = np.full(N, np.nan)
    frac_mu_arr  = np.full(N, np.nan)
    frac_rv_arr  = np.full(N, np.nan)
    frac_lam_arr = np.full(N, 0.0)

    idx = np.where(geom_valid)[0]
    if len(idx) == 0:
        log.warning("No stars with valid geometry for kinematic parallax.")
    else:
        rv_i  = rv[idx]
        srv_i = srv[idx]
        mu_i  = mu_total[idx]
        smu_i = sigma_mu[idx]
        mu_s  = mu_safe[idx]
        lam_i = lambda_rad[idx]

        tan_lam = np.tan(lam_i)

        # π_cp = A_v · μ / (V_r · tan λ)
        pi_cp[idx] = A_V * mu_i / (rv_i * tan_lam)

        # Fractional uncertainty contributions
        frac_mu  = smu_i / mu_s
        frac_rv  = srv_i / np.abs(rv_i)

        frac_mu_arr[idx] = frac_mu
        frac_rv_arr[idx] = frac_rv

        # Partial derivative term for λ: (∂π/∂λ)² σ²(λ) / π²
        # = (cos λ / sin λ)² σ²(λ) = (1/tan λ)² σ²(λ)
        frac_lam_sq = np.zeros(len(idx))
        if sigma_cp_deg is not None:
            # Approximate: σ(λ) ≈ σ_cp [degrees → radians]
            sigma_lambda_rad = np.deg2rad(sigma_cp_deg)
            frac_lam_sq = (np.cos(lam_i) / np.sin(lam_i))**2 * sigma_lambda_rad**2
            frac_lam_arr[idx] = np.sqrt(frac_lam_sq)

        # Combined σ(π_cp) from all terms in quadrature
        frac_total_sq = frac_mu**2 + frac_rv**2 + frac_lam_sq
        sigma_pi_cp[idx] = np.abs(pi_cp[idx]) * np.sqrt(frac_total_sq)

        log.info(
            "Kinematic parallax: %d stars,  median π_cp = %.3f mas,  "
            "median σ(π) = %.3f mas",
            len(idx),
            float(np.nanmedian(pi_cp[idx])),
            float(np.nanmedian(sigma_pi_cp[idx])),
        )

    kinpar_valid = geom_valid & np.isfinite(pi_cp) & (pi_cp > 0) & np.isfinite(sigma_pi_cp)

    return pd.DataFrame({
        "pi_cp":       pi_cp,
        "sigma_pi_cp": sigma_pi_cp,
        "lambda_deg":  lambda_deg,
        "mu_total":    mu_total,
        "sigma_mu":    sigma_mu,
        "frac_mu":     frac_mu_arr,
        "frac_rv":     frac_rv_arr,
        "frac_lambda": frac_lam_arr,
        "geom_valid":  geom_valid,
        "kinpar_valid": kinpar_valid,
    })


# ============================================================================
#  SECTION 4 — CONVENIENCE WRAPPER
# ============================================================================

def run_convergent_point_pipeline(
    df: pd.DataFrame,
    sigma_clip: float = 3.0,
    max_iter: int = 20,
    include_lambda_error: bool = True,
) -> dict:
    """
    Full CP pipeline: find CP → iterative sigma-clipping → kinematic parallaxes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
          ra, dec, pmra, pmra_error, pmdec, pmdec_error,
          radial_velocity (optional), radial_velocity_error (optional).
    sigma_clip : float
        Sigma-clipping threshold for CP fit (default 3.0).
    max_iter : int
        Maximum sigma-clipping iterations (default 20).
    include_lambda_error : bool
        Whether to propagate CP position uncertainty into σ(π_cp).

    Returns
    -------
    dict:
        cp     — dict from find_convergent_point()
        kinpar — pd.DataFrame from compute_kinematic_parallax()
                 (None if fewer than 3 stars have radial velocities)
    """
    # Require valid proper motions
    mu = np.sqrt(df["pmra"].values**2 + df["pmdec"].values**2)
    valid_pm = (mu > 0.1) & df["pmra"].notna() & df["pmdec"].notna()
    sub = df[valid_pm].copy().reset_index(drop=True)

    if len(sub) < 10:
        log.error("Too few stars with valid proper motions (%d < 10).", len(sub))
        return {"cp": None, "kinpar": None}

    log.info("CP pipeline: %d stars with valid proper motions.", len(sub))

    # ── Find CP ─────────────────────────────────────────────────────────── #
    cp = find_convergent_point(
        ra          = sub["ra"].values,
        dec         = sub["dec"].values,
        pmra        = sub["pmra"].values,
        pmdec       = sub["pmdec"].values,
        sigma_pmra  = sub["pmra_error"].values,
        sigma_pmdec = sub["pmdec_error"].values,
        sigma_clip  = sigma_clip,
        max_iter    = max_iter,
    )

    # ── Kinematic parallax (stars with RV only) ─────────────────────────── #
    has_rv = sub["radial_velocity"].notna() if "radial_velocity" in sub.columns else pd.Series(False, index=sub.index)

    if has_rv.sum() < 3:
        log.warning("Too few stars with RV (%d); skipping kinematic parallax.", has_rv.sum())
        kinpar = None
    else:
        # Combine CP position uncertainty into a single arc uncertainty
        sigma_cp_deg = None
        if include_lambda_error:
            sra  = cp.get("sigma_ra_cp",  0.0) or 0.0
            sdec = cp.get("sigma_dec_cp", 0.0) or 0.0
            if np.isfinite(sra) and np.isfinite(sdec):
                dec_cp_rad  = np.deg2rad(cp["dec_cp"])
                sigma_cp_deg = float(np.sqrt(
                    (sra * np.cos(dec_cp_rad))**2 + sdec**2
                ))

        srv = (
            sub["radial_velocity_error"].fillna(1.0).values
            if "radial_velocity_error" in sub.columns
            else np.ones(len(sub))
        )

        kinpar = compute_kinematic_parallax(
            ra              = sub["ra"].values,
            dec             = sub["dec"].values,
            pmra            = sub["pmra"].values,
            pmdec           = sub["pmdec"].values,
            sigma_pmra      = sub["pmra_error"].values,
            sigma_pmdec     = sub["pmdec_error"].values,
            radial_velocity = sub["radial_velocity"].values,
            sigma_rv        = srv,
            ra_cp           = cp["ra_cp"],
            dec_cp          = cp["dec_cp"],
            sigma_cp_deg    = sigma_cp_deg,
            mask            = has_rv.values,
        )

    return {"cp": cp, "kinpar": kinpar}


# ============================================================================
#  SMOKE TEST
# ============================================================================
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s  %(message)s")

    rng = np.random.default_rng(7)
    N = 200

    # Synthetic Hyades-like stars with proper motions pointed toward the true CP
    CP_TRUE_RA  = 97.4
    CP_TRUE_DEC =  6.9
    PLX_TRUE    = 21.53   # mas (Perryman 1997 mean)
    VR_MEAN     = 39.4    # km/s

    ra_cl  = rng.normal(66.75, 4.5, N)
    dec_cl = rng.normal(15.87, 4.5, N)

    # Perfect proper motions: point toward CP
    theta_true = position_angle_to_cp_rad(ra_cl, dec_cl, CP_TRUE_RA, CP_TRUE_DEC)
    lam_true   = great_circle_distance_rad(ra_cl, dec_cl, CP_TRUE_RA, CP_TRUE_DEC)

    # μ magnitude consistent with known kinematics: μ = A_v * plx * tan(lam) / Vr * plx
    # From π_cp = A_v * μ / (Vr * tan λ)  →  μ = π * Vr * tan(λ) / A_v
    mu_mag = PLX_TRUE * VR_MEAN * np.tan(lam_true) / A_V + rng.normal(0, 0.3, N)
    mu_mag = np.clip(mu_mag, 0.5, None)

    # Add scatter to simulate measurement noise
    noise_pm = 0.5  # mas/yr noise
    pmra_cl  = mu_mag * np.sin(theta_true) + rng.normal(0, noise_pm, N)
    pmdec_cl = mu_mag * np.cos(theta_true) + rng.normal(0, noise_pm, N)

    # 10% non-members with random PMs (outliers)
    n_out = N // 10
    pmra_cl[:n_out]  = rng.normal(0, 50, n_out)
    pmdec_cl[:n_out] = rng.normal(0, 50, n_out)

    rv_vals = np.where(
        rng.random(N) > 0.35,
        rng.normal(VR_MEAN, 0.5, N),
        np.nan,
    )

    df_test = pd.DataFrame({
        "ra":                    ra_cl,
        "dec":                   dec_cl,
        "pmra":                  pmra_cl,
        "pmra_error":            rng.uniform(0.1, 0.5, N),
        "pmdec":                 pmdec_cl,
        "pmdec_error":           rng.uniform(0.1, 0.5, N),
        "radial_velocity":       rv_vals,
        "radial_velocity_error": np.full(N, 0.5),
    })

    print(f"\nSmoke test: {N} stars ({n_out} injected non-members)")
    print(f"True CP: RA={CP_TRUE_RA:.4f}°,  Dec={CP_TRUE_DEC:.4f}°")
    print(f"True parallax (Hyades mean): {PLX_TRUE:.3f} mas\n")

    results = run_convergent_point_pipeline(df_test, sigma_clip=3.0)
    cp = results["cp"]

    print(f"Found CP: RA  = {cp['ra_cp']:.4f}° ± {cp['sigma_ra_cp']:.4f}°")
    print(f"         Dec  = {cp['dec_cp']:.4f}° ± {cp['sigma_dec_cp']:.4f}°")
    print(f"Stars used: {cp['n_stars_used']}  |  Rejected: {cp['n_rejected']}  "
          f"|  Iterations: {cp['n_iterations']}")

    kp = results["kinpar"]
    if kp is not None:
        v = kp["kinpar_valid"]
        print(f"\nKinematic parallax ({v.sum()} valid stars):")
        print(f"  Median π_cp  = {kp.loc[v, 'pi_cp'].median():.3f} mas  "
              f"(true = {PLX_TRUE:.3f} mas)")
        print(f"  Median σ(π)  = {kp.loc[v, 'sigma_pi_cp'].median():.3f} mas")
        print(f"  Median σ_μ/μ = {kp.loc[v, 'frac_mu'].median():.4f}")
        print(f"  Median σ_Vr/Vr = {kp.loc[v, 'frac_rv'].median():.4f}")
