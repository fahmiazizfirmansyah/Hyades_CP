"""
==============================================================================
MODULE 1: DATA RETRIEVAL
==============================================================================
Perryman et al. (1997) - "The Hyades: distance, structure, dynamics, and age"
Recreation Pipeline — Module 1

Purpose:
  Query Gaia DR3 for candidate Hyades members and cross-match with MAST
  to flag stars with HST or JWST photometric observations.

Key references:
  - Gaia DR3: Gaia Collaboration et al. (2022), A&A, 674, A1
  - Perryman et al. (1997): A&A, 331, 81–120
  - Hyades centre (J2000): RA = 66.75°, Dec = +15.87°  (in Taurus)

Dependencies:
  pip install astropy astroquery
==============================================================================
"""

import warnings
import logging
import numpy as np
import pandas as pd

from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.mast import Observations

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Silence the verbose astroquery tap warnings unless we explicitly want them
warnings.filterwarnings("ignore", category=UserWarning, module="astroquery")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Hyades cluster centre (J2000) — Perryman et al. (1997), Table 1
HYADES_RA_DEG  = 66.75    # degrees
HYADES_DEC_DEG = 15.87    # degrees
SEARCH_RADIUS_DEG = 20.0  # degrees — generous first-pass cone

# Parallax cuts (rough prior before kinematic membership):
#   Hyades mean parallax ≈ 21.5 mas  (distance ≈ 46.5 pc)
#   Keep stars with ω ∈ [10, 35] mas  (≈ 29–100 pc) to avoid heavy pruning
PARALLAX_MIN_MAS = 10.0
PARALLAX_MAX_MAS = 35.0

# Gaia DR3 release name recognised by astroquery
GAIA_RELEASE = "gaiadr3"


# =========================================================================== #
#  FUNCTION 1 — query_gaia_hyades_candidates
# =========================================================================== #
def query_gaia_hyades_candidates(
    ra_center: float  = HYADES_RA_DEG,
    dec_center: float = HYADES_DEC_DEG,
    radius_deg: float = SEARCH_RADIUS_DEG,
    parallax_min: float = PARALLAX_MIN_MAS,
    parallax_max: float = PARALLAX_MAX_MAS,
    row_limit: int = 50_000,
) -> pd.DataFrame:
    """
    Query the Gaia DR3 TAP service for candidate Hyades members.

    The query uses an ADQL cone search combined with a loose parallax window
    to reduce the result set before kinematic filtering (Module 2).

    Columns retrieved
    -----------------
    Astrometry:
        source_id, ra, dec, parallax (±error), proper motions (±error),
        ruwe (astrometric quality flag)
    Radial velocity:
        radial_velocity, radial_velocity_error  — *sparse* in DR3
    Photometry:
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        bp_rp (= BP − RP colour), bp_g, g_rp

    Parameters
    ----------
    ra_center, dec_center : float
        Cluster centre in decimal degrees (ICRS / J2016.0 Gaia frame).
    radius_deg : float
        Cone-search radius in degrees.
    parallax_min, parallax_max : float
        Loose parallax bounds in milli-arcseconds to pre-filter background /
        foreground stars before proper kinematic membership selection.
    row_limit : int
        Safety cap on returned rows (−1 = unlimited, not recommended).

    Returns
    -------
    pd.DataFrame
        Table of candidate stars with all retrieved columns.
        NaN is preserved for missing radial velocity entries.

    Notes on ADQL
    -------------
    CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', ra0, dec0, r)) is the
    standard ADQL cone-search idiom used by the Gaia archive TAP service.
    """
    log.info(
        "Querying Gaia DR3 — centre (%.3f°, %.3f°), radius %.1f°",
        ra_center, dec_center, radius_deg,
    )

    # ------------------------------------------------------------------ #
    # Build the ADQL query
    # RUWE < 1.4 is the Gaia-recommended threshold for clean astrometry.
    # We keep RUWE as a column so the user can tighten or relax later.
    # ------------------------------------------------------------------ #
    adql = f"""
    SELECT
        source_id,
        ra,  ra_error,
        dec, dec_error,
        parallax,         parallax_error,
        pmra,             pmra_error,
        pmdec,            pmdec_error,
        radial_velocity,  radial_velocity_error,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        bp_rp,
        bp_g,
        g_rp,
        ruwe,
        astrometric_excess_noise
    FROM
        {GAIA_RELEASE}.gaia_source
    WHERE
        CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_center}, {dec_center}, {radius_deg})
        ) = 1
        AND parallax BETWEEN {parallax_min} AND {parallax_max}
        AND parallax_over_error > 5          -- S/N > 5 on parallax
    """

    # astroquery.gaia wraps the synchronous TAP endpoint; for >2000 rows we
    # use launch_job (asynchronous) which is more robust.
    Gaia.ROW_LIMIT = row_limit
    try:
        job    = Gaia.launch_job_async(adql, verbose=False)
        result = job.get_results()
    except Exception as exc:
        log.error("Gaia TAP query failed: %s", exc)
        raise

    df = result.to_pandas()
    log.info("Gaia query returned %d candidate stars.", len(df))

    # ------------------------------------------------------------------ #
    # Diagnostics — radial velocity coverage
    # ------------------------------------------------------------------ #
    n_with_rv = df["radial_velocity"].notna().sum()
    log.info(
        "Radial velocities available for %d / %d stars (%.1f%%).",
        n_with_rv, len(df), 100 * n_with_rv / max(len(df), 1),
    )

    return df


# =========================================================================== #
#  FUNCTION 2 — crossmatch_with_mast
# =========================================================================== #
def crossmatch_with_mast(
    gaia_df: pd.DataFrame,
    match_radius_arcsec: float = 1.0,
    missions: tuple = ("HST", "JWST"),
) -> pd.DataFrame:
    """
    Cross-match a table of Gaia stars against MAST to flag sources that have
    archival HST (WFC3 / ACS) or JWST (NIRCam) photometric observations.

    The match uses a simple nearest-neighbour sky-coordinate match via
    astropy.coordinates.SkyCoord.match_to_catalog_sky().

    Strategy
    --------
    For each target mission we query MAST for all observations within the
    bounding box of our candidate list, then do an in-memory positional
    cross-match (avoids one TAP call per star, which would be very slow).

    Parameters
    ----------
    gaia_df : pd.DataFrame
        Output of query_gaia_hyades_candidates(); must contain 'ra' and 'dec'.
    match_radius_arcsec : float
        Maximum on-sky separation for a positive cross-match, in arcseconds.
    missions : tuple of str
        Mission names as recognised by MAST (e.g. 'HST', 'JWST').

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional boolean flag columns:
            has_hst_obs  — True if any HST WFC3/ACS imaging found
            has_jwst_obs — True if any JWST NIRCam imaging found
        and a string column 'mast_matched_instruments' listing matched
        instrument/filter combinations.
    """
    if gaia_df.empty:
        log.warning("crossmatch_with_mast received an empty DataFrame.")
        gaia_df["has_hst_obs"]  = False
        gaia_df["has_jwst_obs"] = False
        gaia_df["mast_matched_instruments"] = ""
        return gaia_df

    log.info(
        "Cross-matching %d Gaia sources against MAST (missions: %s) …",
        len(gaia_df), missions,
    )

    # Build a SkyCoord for the entire Gaia candidate list (vectorised)
    gaia_coords = SkyCoord(
        ra  = gaia_df["ra"].values  * u.deg,
        dec = gaia_df["dec"].values * u.deg,
        frame = "icrs",
    )

    # Initialise flag columns
    gaia_df = gaia_df.copy()
    gaia_df["has_hst_obs"]  = False
    gaia_df["has_jwst_obs"] = False
    gaia_df["mast_matched_instruments"] = ""

    # Compute bounding box for the MAST query (saves bandwidth)
    ra_min, ra_max   = gaia_df["ra"].min(),  gaia_df["ra"].max()
    dec_min, dec_max = gaia_df["dec"].min(), gaia_df["dec"].max()

    for mission in missions:
        log.info("  Querying MAST for mission = %s …", mission)
        try:
            # Query MAST for all observations in the bounding region
            obs_table = Observations.query_criteria(
                obs_collection = mission,
                s_ra  = [ra_min,  ra_max],
                s_dec = [dec_min, dec_max],
                dataproduct_type = ["image"],  # photometric data only
            )
        except Exception as exc:
            log.warning("  MAST query for %s failed: %s — skipping.", mission, exc)
            continue

        if obs_table is None or len(obs_table) == 0:
            log.info("  No %s observations found in this sky region.", mission)
            continue

        log.info("  MAST returned %d %s observations.", len(obs_table), mission)

        # Filter to science instruments relevant to photometry
        if mission == "HST":
            instr_mask = np.isin(
                [str(i).upper() for i in obs_table["instrument_name"]],
                ["ACS/WFC", "ACS/HRC", "WFC3/UVIS", "WFC3/IR"],
            )
            obs_table = obs_table[instr_mask]
        elif mission == "JWST":
            instr_mask = np.array(
                ["NIRCAM" in str(i).upper() for i in obs_table["instrument_name"]]
            )
            obs_table = obs_table[instr_mask]

        if len(obs_table) == 0:
            log.info("  No relevant instrument data after filtering for %s.", mission)
            continue

        # Build SkyCoord for MAST observations
        mast_coords = SkyCoord(
            ra  = obs_table["s_ra"].data.astype(float)  * u.deg,
            dec = obs_table["s_dec"].data.astype(float) * u.deg,
            frame = "icrs",
        )

        # Nearest-neighbour match: for each Gaia star find closest MAST obs
        idx, sep2d, _ = gaia_coords.match_to_catalog_sky(mast_coords)
        matched = sep2d < (match_radius_arcsec * u.arcsec)

        flag_col = "has_hst_obs" if mission == "HST" else "has_jwst_obs"
        gaia_df.loc[matched, flag_col] = True

        # Record instrument names for matched stars
        for gaia_idx in np.where(matched)[0]:
            mast_idx = idx[gaia_idx]
            instr = str(obs_table["instrument_name"][mast_idx])
            filt  = str(obs_table["filters"][mast_idx]) if "filters" in obs_table.colnames else ""
            entry = f"{mission}/{instr}/{filt}"
            existing = gaia_df.at[gaia_idx, "mast_matched_instruments"]
            gaia_df.at[gaia_idx, "mast_matched_instruments"] = (
                existing + ("; " if existing else "") + entry
            )

    n_hst  = gaia_df["has_hst_obs"].sum()
    n_jwst = gaia_df["has_jwst_obs"].sum()
    log.info(
        "MAST cross-match complete — HST flags: %d, JWST flags: %d.",
        n_hst, n_jwst,
    )
    return gaia_df


# =========================================================================== #
#  CONVENIENCE WRAPPER — run module standalone
# =========================================================================== #
def retrieve_and_flag(save_csv: str = "hyades_gaia_mast.csv") -> pd.DataFrame:
    """
    Full Module 1 pipeline:  Gaia query  →  MAST cross-match  →  CSV save.

    Returns the combined DataFrame for passing to Module 2.
    """
    df = query_gaia_hyades_candidates()
    df = crossmatch_with_mast(df)

    df.to_csv(save_csv, index=False)
    log.info("Module 1 complete.  Results saved to '%s'.", save_csv)
    return df


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    data = retrieve_and_flag()
    print(data.head())
    print(f"\nShape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
