"""
==============================================================================
MASTER PIPELINE — Perryman et al. (1997) Hyades Replication
==============================================================================
Orchestrates Modules 1–4 into a single end-to-end run.

Usage
-----
  python run_pipeline.py                  # full Gaia live query
  python run_pipeline.py --from-csv data.csv   # skip Gaia query
  python run_pipeline.py --smoke-test     # synthetic data only (no network)

Quick-start (recommended first run):
  python run_pipeline.py --smoke-test

File layout:
  hyades_pipeline/
  ├── module1_data_retrieval.py
  ├── module2_kinematics.py
  ├── module3_spatial_dynamics.py
  ├── module4_hr_diagram.py
  └── run_pipeline.py              ← this file
==============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Set up logging before any imports that might emit their own ─────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hyades_pipeline.log", mode="w"),
    ],
)
log = logging.getLogger("run_pipeline")

# ── Local module imports ─────────────────────────────────────────────────────
from module1_data_retrieval    import retrieve_and_flag, query_gaia_hyades_candidates
from module2_kinematics        import run_kinematics_pipeline
from module3_spatial_dynamics  import run_spatial_dynamics
from module4_hr_diagram        import run_hr_pipeline, compute_absolute_magnitudes
from module_convergent_point   import run_convergent_point_pipeline


# =========================================================================== #
#  SMOKE-TEST DATA GENERATOR
# =========================================================================== #
def make_synthetic_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic Hyades-like dataset for offline testing.
    Uses known cluster parameters from Perryman et al. (1997):
      - Mean parallax: 21.5 mas (d ≈ 46.5 pc)
      - Mean PM: (μα*, μδ) ≈ (102, −27.5) mas yr⁻¹
      - Mean radial velocity: ~39.4 km s⁻¹
      - ~35% of stars lack a measured RV (simulates DR3 completeness)
    """
    rng = np.random.default_rng(seed)

    # True cluster members (2/3 of sample)
    n_mem = int(0.67 * n)
    # Random field stars (1/3)
    n_fld = n - n_mem

    # --- Members ---
    parallax_mem   = rng.normal(21.5, 1.2, n_mem)
    pmra_mem       = rng.normal(102.0, 1.5, n_mem)
    pmdec_mem      = rng.normal(-27.5, 1.5, n_mem)
    rv_mem_full    = rng.normal(39.4, 0.4, n_mem)
    rv_has_mem     = rng.random(n_mem) > 0.35       # ~65% have RV
    rv_mem         = np.where(rv_has_mem, rv_mem_full, np.nan)

    # Cluster extent: ~10 deg radius on sky
    ra_mem   = rng.normal(66.75, 5.0, n_mem)
    dec_mem  = rng.normal(15.87, 5.0, n_mem)
    dist_mem = 1000.0 / parallax_mem

    # G-band magnitudes from a realistic luminosity function
    abs_g_mem = np.concatenate([
        rng.uniform(0.5, 3.5,  int(0.08 * n_mem)),   # giants/subgiants
        rng.uniform(3.5, 7.5,  int(0.45 * n_mem)),   # upper MS
        rng.uniform(7.5, 12.0, int(0.47 * n_mem)),   # lower MS
    ])[:n_mem]
    app_g_mem = abs_g_mem + 5 * np.log10(dist_mem / 10.0)

    # BP-RP from abs_g via inverse ML (rough)
    bprp_mem = np.clip((abs_g_mem + 0.48) / 3.82 + rng.normal(0, 0.05, n_mem),
                       0.2, 3.2)

    # --- Field stars ---
    parallax_fld = rng.uniform(10, 35, n_fld)
    ra_fld  = rng.uniform(47, 87, n_fld)
    dec_fld = rng.uniform(6, 26, n_fld)
    dist_fld = 1000.0 / parallax_fld

    # Field: random proper motions and RVs
    pmra_fld  = rng.normal(0, 30, n_fld)
    pmdec_fld = rng.normal(0, 20, n_fld)
    rv_fld    = np.where(rng.random(n_fld) > 0.5, rng.normal(0, 40, n_fld), np.nan)
    abs_g_fld = rng.uniform(4, 13, n_fld)
    app_g_fld = abs_g_fld + 5 * np.log10(dist_fld / 10.0)
    bprp_fld  = rng.uniform(0.3, 3.0, n_fld)

    # --- Combine ---
    df = pd.DataFrame({
        "source_id":              np.arange(n, dtype=np.int64),
        "ra":                     np.concatenate([ra_mem,  ra_fld]),
        "ra_error":               rng.uniform(0.01, 0.05, n),
        "dec":                    np.concatenate([dec_mem, dec_fld]),
        "dec_error":              rng.uniform(0.01, 0.05, n),
        "parallax":               np.concatenate([parallax_mem, parallax_fld]),
        "parallax_error":         rng.uniform(0.02, 0.15, n),
        "pmra":                   np.concatenate([pmra_mem,  pmra_fld]),
        "pmra_error":             rng.uniform(0.05, 0.50, n),
        "pmdec":                  np.concatenate([pmdec_mem, pmdec_fld]),
        "pmdec_error":            rng.uniform(0.05, 0.50, n),
        "radial_velocity":        np.concatenate([rv_mem, rv_fld]),
        "radial_velocity_error":  rng.uniform(0.3, 3.0, n),
        "phot_g_mean_mag":        np.concatenate([app_g_mem,  app_g_fld]),
        "phot_bp_mean_mag":       np.concatenate([app_g_mem,  app_g_fld]) + bprp_mem[:n] * 0.5,
        "phot_rp_mean_mag":       np.concatenate([app_g_mem,  app_g_fld]) - bprp_mem[:n] * 0.5,
        "bp_rp":                  np.concatenate([bprp_mem, bprp_fld]),
        "ruwe":                   rng.uniform(0.9, 1.35, n),
        "astrometric_excess_noise": rng.uniform(0, 0.3, n),
        # MAST flags (simulated sparse)
        "has_hst_obs":  np.where(rng.random(n) > 0.92, True, False),
        "has_jwst_obs": np.where(rng.random(n) > 0.97, True, False),
        "mast_matched_instruments": "",
        # Ground truth labels for validation
        "_true_member": np.concatenate([
            np.ones(n_mem, dtype=bool), np.zeros(n_fld, dtype=bool)
        ]),
    })

    log.info(
        "Synthetic dataset: %d total (%d true members + %d field stars).",
        n, n_mem, n_fld,
    )
    return df


# =========================================================================== #
#  MAIN PIPELINE ORCHESTRATOR
# =========================================================================== #
def run_full_pipeline(
    from_csv:    str | None = None,
    smoke_test:  bool       = False,
    save_results: bool      = True,
    iso_ages_myr: list[float] = [600, 625, 650],
    iso_filepaths: list | None = None,
) -> dict:
    """
    End-to-end Hyades analysis pipeline.

    Parameters
    ----------
    from_csv : str or None
        If provided, skip Module 1 and load candidates from this CSV.
    smoke_test : bool
        If True, use synthetic data (no network access required).
    save_results : bool
        Whether to write outputs (CSV, figures) to disk.
    iso_ages_myr : list of float
        Isochrone ages in Myr to overlay on the HR diagram.
    iso_filepaths : list or None
        Paths to MIST .iso.cmd files; None entries use analytic placeholders.

    Returns
    -------
    dict with keys: candidates, members, spatial, fig
    """
    results = {}
    output_dir = Path(".")

    # ================================================================== #
    #  MODULE 1 — Data Retrieval
    # ================================================================== #
    log.info("=" * 68)
    log.info("MODULE 1 — Data Retrieval")
    log.info("=" * 68)

    if smoke_test:
        log.info("SMOKE TEST MODE: using synthetic dataset.")
        candidates_df = make_synthetic_dataset(n=350)
    elif from_csv:
        log.info("Loading candidates from '%s'.", from_csv)
        candidates_df = pd.read_csv(from_csv)
    else:
        log.info("Querying Gaia DR3 (live network access required).")
        from module1_data_retrieval import retrieve_and_flag
        csv_out = output_dir / "hyades_gaia_mast.csv"
        candidates_df = retrieve_and_flag(save_csv=str(csv_out))

    results["candidates"] = candidates_df
    log.info(
        "Module 1 done.  %d candidate stars.  "
        "HST flags: %d, JWST flags: %d.",
        len(candidates_df),
        candidates_df.get("has_hst_obs", pd.Series([False]*len(candidates_df))).sum(),
        candidates_df.get("has_jwst_obs", pd.Series([False]*len(candidates_df))).sum(),
    )

    # ================================================================== #
    #  MODULE 2 — Kinematics & Membership
    # ================================================================== #
    log.info("=" * 68)
    log.info("MODULE 2 — Kinematics & Membership Selection")
    log.info("=" * 68)

    full_kinematic_df = run_kinematics_pipeline(
        candidates_df,
        rv_fill=None,           # do not impute missing v_r
        sigma_threshold=3.0,    # 3-sigma cut
    )
    members_df = full_kinematic_df[full_kinematic_df["is_member"]].copy()
    results["kinematic_full"] = full_kinematic_df
    results["members"] = members_df

    log.info(
        "Module 2 done.  Members selected: %d / %d  (%.1f%%)",
        len(members_df), len(full_kinematic_df),
        100 * len(members_df) / max(len(full_kinematic_df), 1),
    )

    # If ground truth is available (smoke-test), report purity/completeness
    if "_true_member" in members_df.columns:
        tp = members_df["_true_member"].sum()
        fp = (~members_df["_true_member"]).sum()
        fn = (full_kinematic_df["_true_member"] & ~full_kinematic_df["is_member"]).sum()
        log.info(
            "  Smoke-test validation:  TP=%d  FP=%d  FN=%d  "
            "Purity=%.1f%%  Completeness=%.1f%%",
            tp, fp, fn,
            100 * tp / max(tp + fp, 1),
            100 * tp / max(tp + fn, 1),
        )

    if save_results:
        members_path = output_dir / "hyades_members.csv"
        members_df.to_csv(members_path, index=False)
        log.info("Members saved to '%s'.", members_path)

    # ================================================================== #
    #  MODULE 2b — Convergent Point & Kinematic Parallax
    # ================================================================== #
    log.info("=" * 68)
    log.info("MODULE 2b — Convergent Point & Kinematic Parallax")
    log.info("=" * 68)

    if len(members_df) < 10:
        log.warning("Too few members for CP analysis; skipping.")
        cp_results = {"cp": None, "kinpar": None}
    else:
        cp_results = run_convergent_point_pipeline(
            members_df,
            sigma_clip=3.0,
            max_iter=20,
        )
        if cp_results["cp"] is not None:
            cp = cp_results["cp"]
            log.info(
                "CP found: RA = %.4f ± %.4f°,  Dec = %.4f ± %.4f°",
                cp["ra_cp"], cp["sigma_ra_cp"],
                cp["dec_cp"], cp["sigma_dec_cp"],
            )
            log.info(
                "Stars used for CP: %d,  rejected: %d,  iterations: %d",
                cp["n_stars_used"], cp["n_rejected"], cp["n_iterations"],
            )
        if cp_results["kinpar"] is not None:
            kp = cp_results["kinpar"]
            valid = kp["kinpar_valid"]
            log.info(
                "Kinematic parallax: %d stars,  median π_cp = %.3f mas,  "
                "median σ(π) = %.3f mas",
                valid.sum(),
                kp.loc[valid, "pi_cp"].median(),
                kp.loc[valid, "sigma_pi_cp"].median(),
            )

    results["cp_results"] = cp_results

    # ================================================================== #
    #  MODULE 3 — Spatial Dynamics
    # ================================================================== #
    log.info("=" * 68)
    log.info("MODULE 3 — Spatial Dynamics")
    log.info("=" * 68)

    if len(members_df) < 5:
        log.warning("Too few members for spatial analysis; skipping Module 3.")
        spatial = {"com": None, "tidal": None, "profile": None}
    else:
        spatial = run_spatial_dynamics(members_df)

    results["spatial"] = spatial

    if spatial["tidal"] is not None:
        log.info("Module 3 done.")
        log.info(
            "  CoM: (RA=%.3f°, Dec=%.3f°, d=%.1f pc)",
            spatial["com"]["ra_com"],
            spatial["com"]["dec_com"],
            spatial["com"]["dist_com_pc"],
        )
        log.info(
            "  Total mass: %.1f M☉   |   Tidal radius: %.2f pc",
            spatial["com"]["total_mass_msun"],
            spatial["tidal"]["r_t_pc"],
        )

    # ================================================================== #
    #  MODULE 4 — HR Diagram & Age Fitting
    # ================================================================== #
    log.info("=" * 68)
    log.info("MODULE 4 — HR Diagram & Age Fitting")
    log.info("=" * 68)

    # Ensure abs_g is computed (might not be in smoke-test data)
    if "abs_g" not in members_df.columns:
        members_df = compute_absolute_magnitudes(members_df)

    hr_save = str(output_dir / "hyades_hr_diagram.pdf") if save_results else None

    tidal_n_inside = None
    tidal_r_pc     = None
    if spatial["tidal"] is not None and spatial["com"] is not None:
        tidal_r_pc = spatial["tidal"]["r_t_pc"]
        # Count stars within tidal radius
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        com = spatial["com"]
        mem_valid = members_df[(members_df["parallax"] > 0)].copy()
        mem_valid["dist_pc"] = 1000.0 / mem_valid["parallax"]
        coords = SkyCoord(
            ra       = mem_valid["ra"].values    * u.deg,
            dec      = mem_valid["dec"].values   * u.deg,
            distance = mem_valid["dist_pc"].values * u.pc,
        )
        com_c = SkyCoord(
            ra       = com["ra_com"]    * u.deg,
            dec      = com["dec_com"]   * u.deg,
            distance = com["dist_com_pc"] * u.pc,
        )
        sep_3d_pc = coords.separation_3d(com_c).to(u.pc).value
        tidal_n_inside = int((sep_3d_pc < tidal_r_pc).sum())

    fig, ax = run_hr_pipeline(
        members_df,
        iso_ages_myr  = iso_ages_myr,
        iso_filepaths = iso_filepaths,
        save_path     = hr_save,
        tidal_radius_pc = tidal_r_pc,
        tidal_n_inside  = tidal_n_inside,
    )
    results["fig"] = fig

    log.info("Module 4 done.  HR diagram saved to '%s'.", hr_save)

    # ================================================================== #
    #  SUMMARY TABLE
    # ================================================================== #
    log.info("")
    log.info("=" * 68)
    log.info("  PIPELINE SUMMARY — Hyades Replication (Perryman et al. 1997)")
    log.info("=" * 68)
    log.info("  Candidate stars queried:  %d", len(candidates_df))
    log.info("  Kinematic members:        %d", len(members_df))
    if spatial["com"]:
        log.info("  Cluster distance:         %.1f pc", spatial["com"]["dist_com_pc"])
        log.info("  Total (estimated) mass:   %.0f M☉",  spatial["com"]["total_mass_msun"])
        log.info("  Tidal radius:             %.2f pc",  spatial["tidal"]["r_t_pc"])
        if tidal_n_inside is not None:
            log.info("  Stars within r_t:         %d",      tidal_n_inside)
    log.info("  Isochrone ages tested:    %s Myr", iso_ages_myr)
    log.info("=" * 68)

    return results


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyades cluster analysis pipeline (Perryman+ 1997 replication)"
    )
    parser.add_argument(
        "--from-csv", metavar="FILE",
        help="Load Gaia candidates from a previously saved CSV (skips Module 1 query)."
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run on synthetic data only — no network access required."
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write any output files."
    )
    parser.add_argument(
        "--iso-files", nargs="+", metavar="FILE",
        help="Paths to MIST .iso.cmd files for 600, 625, 650 Myr."
    )
    args = parser.parse_args()

    iso_fp = args.iso_files if args.iso_files else None

    run_full_pipeline(
        from_csv     = args.from_csv,
        smoke_test   = args.smoke_test,
        save_results = not args.no_save,
        iso_filepaths = iso_fp,
    )

    plt.show()
