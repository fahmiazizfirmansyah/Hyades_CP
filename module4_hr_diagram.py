"""
==============================================================================
MODULE 4: HR DIAGRAM & AGE FITTING
==============================================================================
Perryman et al. (1997) Recreation Pipeline — Module 4

Purpose:
  Construct a publication-ready Hertzsprung-Russell diagram (CMD) for
  confirmed Hyades members using Gaia photometry, and overlay theoretical
  MESA/MIST isochrones for age determination.

Physics Background
------------------

  ABSOLUTE MAGNITUDE
  ──────────────────
  The distance modulus converts apparent to absolute magnitude:

    M_G  =  m_G  −  5 log₁₀(d / 10 pc)
          =  m_G  +  5 log₁₀(ω / 100 mas)    [parallax form]
          =  m_G  +  5 log₁₀(ω [mas]) − 10.0

  For the Hyades, interstellar extinction is negligible (E(B−V) < 0.01;
  Taylor 2006) because the cluster is only 46.5 pc away.  We therefore
  set A_G = 0.  For completeness the extinction correction would be:
    M_G_corr = M_G − A_G    where  A_G ≈ 2.740 × E(B−V)  (Cardelli 1989)

  ISOCHRONE FITTING
  ─────────────────
  Stellar evolution theory predicts the locus of stars in the CMD at a
  given age.  The MESA (Modules and Experiments in Stellar Astrophysics)
  code generates these isochrones for a specified metallicity [Fe/H] and
  age τ.  For the Hyades:
    [Fe/H] = +0.14 ± 0.05  (Taylor 2006; approximately solar)
    Age τ  = 625 ± 50 Myr   (Perryman 1997 via main-sequence turnoff)

  In practice, MESA/MIST isochrones are distributed as tables of
  (initial mass → effective temperature T_eff, log g, luminosity, and
  synthetic magnitudes in various filter systems).  We download them from:
    https://waps.cfa.harvard.edu/MIST/iso_form.php
  or use the isochrones Python package (Dotter 2016, ApJS 222, 8).

  Here we provide:
    (a) a full plotting function that reads pre-downloaded MIST .iso.cmd files
    (b) a placeholder generator that produces analytic approximations to
        illustrate how the overlay would look

Dependencies:
  pip install astropy numpy pandas matplotlib scipy
  (optional for real isochrones): pip install isochrones
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
# Publication plot style parameters
# --------------------------------------------------------------------------- #
# We define everything explicitly rather than relying on a style file so the
# module is self-contained and reproducible on any machine.

_FONT_FAMILY   = "serif"
_FONT_SIZE_BASE = 13
_TITLE_SIZE    = 15
_LABEL_SIZE    = 14
_TICK_SIZE     = 12
_LEGEND_SIZE   = 11

# Colour palette for the CMD point colours (Gaia BP-RP → surface temperature proxy)
_CMAP_CMD = LinearSegmentedColormap.from_list(
    "spectral_cmd",
    ["#1a0a5c", "#1460a8", "#3498db", "#aee8f8",   # hot blue end
     "#ffffcc", "#ffd966", "#ffaa00",               # solar
     "#e05000", "#8b0000"],                         # cool red end
    N=512,
)

# Isochrone ages and their visual styles
_ISO_AGES_MYR = [600, 625, 650]
_ISO_COLORS   = ["#e74c3c", "#2ecc71", "#3498db"]   # red, green, blue
_ISO_STYLES   = ["--", "-", ":"]
_ISO_WIDTHS   = [1.5, 2.5, 1.5]


# =========================================================================== #
#  FUNCTION 1 — compute_absolute_magnitudes
# =========================================================================== #
def compute_absolute_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Gaia G-band absolute magnitude M_G from parallax and
    apparent magnitude.

    Distance modulus:  μ = m_G − M_G = 5 log₁₀(d/10 pc)
      ⟹  M_G = m_G − 5 log₁₀(1000/(ω·10)) = m_G + 5 log₁₀(ω[mas]) − 10

    Also propagates the M_G error from parallax and photometric errors.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: phot_g_mean_mag, parallax, parallax_error,
                      bp_rp  (Gaia colour index).

    Returns
    -------
    pd.DataFrame with added columns:
        abs_g          — absolute G magnitude
        abs_g_err      — propagated uncertainty
        dist_mod       — distance modulus
    """
    df = df.copy()

    valid = (df["parallax"] > 0) & df["phot_g_mean_mag"].notna()
    df["abs_g"]     = np.nan
    df["abs_g_err"] = np.nan
    df["dist_mod"]  = np.nan

    # M_G = m_G + 5 log₁₀(ω) − 10  with ω in mas
    plx = df.loc[valid, "parallax"].values
    mg  = df.loc[valid, "phot_g_mean_mag"].values

    df.loc[valid, "abs_g"]    = mg + 5.0 * np.log10(plx) - 10.0
    df.loc[valid, "dist_mod"] = mg - df.loc[valid, "abs_g"]

    # Error propagation:  σ(M_G) = sqrt(σ_m² + (5 σ_ω / (ω ln 10))²)
    sigma_plx = df.loc[valid, "parallax_error"].values
    sigma_phot = 0.002   # typical Gaia G-band photometric error (mag)

    df.loc[valid, "abs_g_err"] = np.sqrt(
        sigma_phot**2 + (5.0 * sigma_plx / (plx * np.log(10.0)))**2
    )

    log.info(
        "Absolute magnitudes computed for %d / %d stars.",
        valid.sum(), len(df),
    )
    log.info(
        "  M_G range: [%.2f, %.2f];  BP-RP range: [%.2f, %.2f]",
        df["abs_g"].min(), df["abs_g"].max(),
        df["bp_rp"].min(),  df["bp_rp"].max(),
    )
    return df


# =========================================================================== #
#  FUNCTION 2 — generate_placeholder_isochrone
# =========================================================================== #
def generate_placeholder_isochrone(
    age_myr: float,
    metallicity_feh: float = 0.14,
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Generate an ANALYTIC APPROXIMATION isochrone for illustration purposes.

    THIS IS A PLACEHOLDER — for science use, replace with real MESA/MIST
    isochrone files (see load_mist_isochrone() below).

    The MS locus is approximated using a polynomial fit to the empirical
    Pecaut & Mamajek (2013) main-sequence colour-magnitude relation, with a
    small age-dependent shift at the turnoff point.

    Physics:
      The main-sequence turnoff luminosity scales as:
        M_turnoff  ∝  M_ZAMS^4 / t     (MS lifetime ∝ M/L ∝ M^(-3))
      For a Salpeter IMF the turnoff mass at age t [Gyr] is approximately:
        M_TO [M☉] ≈ (10.0 / t_Gyr)^0.4
      and the turnoff in absolute G mag shifts by ~0.1–0.2 mag per 50 Myr
      in this age range.

    Parameters
    ----------
    age_myr : float
        Isochrone age in Myr.
    metallicity_feh : float
        Metallicity [Fe/H] — used for a small colour shift.
    n_points : int
        Number of points along the isochrone.

    Returns
    -------
    pd.DataFrame with columns: bp_rp, abs_g, phase
        phase: 'MS' (main sequence), 'SG' (subgiant), 'RGB' (red giant)
    """
    # ------------------------------------------------------------------ #
    # Empirical main sequence polynomial (Pecaut & Mamajek 2013)
    # BP-RP from 0.2 (hot F stars) to 3.0 (cool M dwarfs)
    # M_G = f(BP-RP) fitted to colour-magnitude data.
    # Polynomial coefficients for M_G vs (BP-RP):
    #   M_G ≈ a + b·(BP-RP) + c·(BP-RP)²
    # Approximate fit: blueward anchored at solar analog
    # ------------------------------------------------------------------ #
    a_ms = -0.48
    b_ms =  3.82
    c_ms = -0.10   # slight MS curvature

    # Main sequence locus (cool end)
    bprp_ms = np.linspace(0.25, 3.0, n_points)
    abs_g_ms = a_ms + b_ms * bprp_ms + c_ms * bprp_ms**2

    # ------------------------------------------------------------------ #
    # Age-dependent turnoff:
    #   Approximate turnoff: M_TO [M☉] ≈ (10 / (age_myr/1000))^0.4
    #   Convert to M_G using the ML relation from Module 3
    # ------------------------------------------------------------------ #
    age_gyr = age_myr / 1000.0
    m_to    = (10.0 / age_gyr) ** 0.4           # turnoff mass in M☉
    # Approximate turnoff M_G from log10(M) = 0.507 - 0.0906*MG + 0.00256*MG²
    # Solve for MG numerically
    from numpy.polynomial import polynomial as P
    # log10(m_to) = 0.507 - 0.0906*MG + 0.00256*MG²
    # 0.00256*MG² - 0.0906*MG + (0.507 - log10(m_to)) = 0
    a2, b2 = 0.00256, -0.0906
    c2 = 0.507 - np.log10(max(m_to, 0.5))
    discriminant = b2**2 - 4*a2*c2
    if discriminant >= 0:
        mg_to = (-b2 - np.sqrt(discriminant)) / (2 * a2)
    else:
        mg_to = 3.0  # fallback

    # Clip MS to below turnoff
    bprp_to = (mg_to - a_ms) / b_ms   # approx turnoff colour
    ms_below_to = bprp_ms[bprp_ms >= max(bprp_to - 0.1, 0.25)]
    abs_g_below = a_ms + b_ms * ms_below_to + c_ms * ms_below_to**2

    # ------------------------------------------------------------------ #
    # Subgiant branch: hook above the MS at the turnoff
    # ------------------------------------------------------------------ #
    bprp_sg  = np.linspace(bprp_to - 0.1, bprp_to + 0.5, 30)
    abs_g_sg = (
        a_ms + b_ms * bprp_sg + c_ms * bprp_sg**2
        - 0.5 * np.exp(-3.0 * (bprp_sg - bprp_to)**2)  # slight hook upward
    )

    # ------------------------------------------------------------------ #
    # Red-giant branch: nearly vertical in colour, rising in luminosity
    # ------------------------------------------------------------------ #
    abs_g_rgb = np.linspace(abs_g_sg[-1], -2.0, 40)
    bprp_rgb  = bprp_sg[-1] + 0.15 * (abs_g_sg[-1] - abs_g_rgb) / abs_g_sg[-1]

    # Apply metallicity shift: higher [Fe/H] → redder colour by ~0.05 mag
    delta_colour = 0.05 * metallicity_feh / 0.1

    # Compile full isochrone
    bprp_full  = np.concatenate([ms_below_to, bprp_sg, bprp_rgb]) + delta_colour
    abs_g_full = np.concatenate([abs_g_below, abs_g_sg, abs_g_rgb])
    phase_full = (
        ["MS"] * len(ms_below_to)
        + ["SG"] * len(bprp_sg)
        + ["RGB"] * len(abs_g_rgb)
    )

    return pd.DataFrame({
        "bp_rp": bprp_full,
        "abs_g": abs_g_full,
        "phase": phase_full,
    })


# =========================================================================== #
#  FUNCTION 3 — load_mist_isochrone (real data path)
# =========================================================================== #
def load_mist_isochrone(
    filepath: str | Path,
    age_myr: float,
    age_col: str = "star_age",
    eep_col: str = "EEP",
    bprp_col: str = "Gaia_BP-RP",
    mg_col: str = "Gaia_G",
) -> pd.DataFrame:
    """
    Load a single isochrone from a MIST .iso.cmd file and extract the
    age-slice corresponding to age_myr.

    MIST files are produced by the MIST isochrone interpolator:
    https://waps.cfa.harvard.edu/MIST/iso_form.php

    Select:
      - Photometric system: Gaia DR2 / EDR3
      - Version:  v1.2  (or latest)
      - Output:   .iso.cmd

    The file format is a series of blocks, one per age point, preceded by
    a header line "#  age =  X.XX   [log yr]".

    Parameters
    ----------
    filepath : str or Path
        Path to the downloaded .iso.cmd file.
    age_myr : float
        Target age in Myr.  The closest age block is selected.
    age_col, eep_col, bprp_col, mg_col : str
        Column names in the MIST file (check file header).

    Returns
    -------
    pd.DataFrame with columns bp_rp, abs_g, phase (EEP-based)
        EEP < 454  → main sequence
        EEP 454–631→ subgiant + red giant
        EEP > 631  → horizontal branch / AGB
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"MIST isochrone file not found: {filepath}\n"
            "Download from https://waps.cfa.harvard.edu/MIST/iso_form.php\n"
            "or use generate_placeholder_isochrone() for illustration."
        )

    log.info("Loading MIST isochrone from %s", filepath)

    # ------------------------------------------------------------------ #
    # Parse the MIST .iso.cmd file format
    # ------------------------------------------------------------------ #
    target_log_age = np.log10(age_myr * 1e6)   # MIST uses log10(age/yr)

    all_blocks  = []
    header_cols = None
    current_age = None
    current_rows = []

    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n")

            # Header line for a new age block
            if line.startswith("#  age =") or line.startswith("# age ="):
                # Save previous block
                if current_rows and header_cols:
                    block_df = pd.DataFrame(current_rows, columns=header_cols)
                    block_df["_log_age"] = current_age
                    all_blocks.append(block_df)
                parts = line.split("=")
                current_age = float(parts[1].split()[0])
                current_rows = []
                continue

            # Column header line
            if line.startswith("#") and not line.startswith("# age"):
                cols = line.lstrip("# ").split()
                if len(cols) > 3:
                    header_cols = cols
                continue

            # Data line
            if not line.startswith("#") and line.strip():
                try:
                    values = [float(x) for x in line.split()]
                    current_rows.append(values)
                except ValueError:
                    continue

        # Save last block
        if current_rows and header_cols:
            block_df = pd.DataFrame(current_rows, columns=header_cols)
            block_df["_log_age"] = current_age
            all_blocks.append(block_df)

    if not all_blocks:
        raise ValueError("No isochrone blocks parsed from file.")

    full_df = pd.concat(all_blocks, ignore_index=True)

    # Select the age block closest to target
    unique_ages = full_df["_log_age"].unique()
    closest_age = unique_ages[np.argmin(np.abs(unique_ages - target_log_age))]
    log.info(
        "Requested log_age=%.3f; using closest block log_age=%.3f  (%.1f Myr)",
        target_log_age, closest_age, 10**closest_age / 1e6,
    )

    iso_df = full_df[full_df["_log_age"] == closest_age].copy()

    # EEP-based phase classification
    eep = iso_df[eep_col].values
    phase = np.where(eep < 454, "MS", np.where(eep < 631, "SG/RGB", "HB/AGB"))
    iso_df["phase"] = phase

    # Rename to standard column names
    iso_df = iso_df.rename(columns={bprp_col: "bp_rp", mg_col: "abs_g"})

    return iso_df[["bp_rp", "abs_g", "phase"]].dropna()


# =========================================================================== #
#  FUNCTION 4 — plot_hr_diagram  (MAIN PLOTTING FUNCTION)
# =========================================================================== #
def plot_hr_diagram(
    members_df: pd.DataFrame,
    iso_ages_myr: list[float] = _ISO_AGES_MYR,
    iso_filepaths: list[str | None] | None = None,
    metallicity_feh: float = 0.14,
    save_path: str | Path | None = "hyades_hr_diagram.pdf",
    tidal_n_inside: int | None = None,
    tidal_radius_pc: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a publication-quality Hertzsprung-Russell diagram (CMD).

    Features:
      - Colour-coded scatter by BP-RP colour (spectral class proxy)
      - Error bars on M_G from parallax errors
      - Isochrone overlays (real MIST if files provided, otherwise analytic)
      - Annotated regions (main sequence, subgiant branch, giants)
      - Publication-ready typography and layout
      - Optional tidal membership annotation

    Parameters
    ----------
    members_df : pd.DataFrame
        Confirmed Hyades members; must have abs_g, bp_rp, abs_g_err.
        Compute abs_g first via compute_absolute_magnitudes().
    iso_ages_myr : list of float
        Isochrone ages to overlay [Myr].
    iso_filepaths : list of str or None, or None
        Paths to MIST .iso.cmd files; use None entries for placeholder.
        If None entirely, all isochrones use analytic placeholder.
    metallicity_feh : float
        Cluster metallicity for isochrone label.
    save_path : str, Path, or None
        Save figure to this path (PDF recommended for publication).
        If None, figure is not saved automatically.
    tidal_n_inside, tidal_radius_pc : optional
        If provided, adds annotation about tidal radius census.

    Returns
    -------
    (fig, ax) — matplotlib Figure and Axes objects.
    """
    # ------------------------------------------------------------------ #
    # Resolve which stars to plot
    # ------------------------------------------------------------------ #
    plot_df = members_df[members_df["abs_g"].notna() & members_df["bp_rp"].notna()].copy()
    log.info("Plotting HR diagram with %d member stars.", len(plot_df))

    # ------------------------------------------------------------------ #
    # Global matplotlib settings (publication style)
    # ------------------------------------------------------------------ #
    plt.rcParams.update({
        "font.family":        _FONT_FAMILY,
        "font.size":          _FONT_SIZE_BASE,
        "axes.labelsize":     _LABEL_SIZE,
        "axes.titlesize":     _TITLE_SIZE,
        "xtick.labelsize":    _TICK_SIZE,
        "ytick.labelsize":    _TICK_SIZE,
        "legend.fontsize":    _LEGEND_SIZE,
        "axes.linewidth":     1.2,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top":          True,
        "ytick.right":        True,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })

    # ------------------------------------------------------------------ #
    # Figure layout
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(9, 11))
    # Main CMD axis
    ax = fig.add_axes([0.13, 0.10, 0.74, 0.78])
    # Colour bar axis on right
    cax = fig.add_axes([0.88, 0.10, 0.025, 0.78])

    # ------------------------------------------------------------------ #
    # Scatter plot — colour-coded by BP-RP
    # ------------------------------------------------------------------ #
    bprp_min, bprp_max = 0.2, 3.2   # display range
    norm = Normalize(vmin=bprp_min, vmax=bprp_max)

    # Sort by colour so redder stars draw on top (nicer aesthetics)
    plot_df = plot_df.sort_values("bp_rp")

    sc = ax.scatter(
        plot_df["bp_rp"].values,
        plot_df["abs_g"].values,
        c         = plot_df["bp_rp"].values,
        cmap      = _CMAP_CMD,
        norm      = norm,
        s         = 28,
        alpha     = 0.85,
        linewidths= 0.3,
        edgecolors= "k",
        zorder    = 5,
        label     = "Hyades members (Gaia DR3)",
    )

    # ------------------------------------------------------------------ #
    # Error bars on M_G
    # ------------------------------------------------------------------ #
    if "abs_g_err" in plot_df.columns:
        ax.errorbar(
            plot_df["bp_rp"].values,
            plot_df["abs_g"].values,
            yerr   = plot_df["abs_g_err"].fillna(0).values,
            fmt    = "none",
            ecolor = "0.5",
            elinewidth = 0.4,
            capsize    = 0,
            alpha  = 0.5,
            zorder = 4,
        )

    # ------------------------------------------------------------------ #
    # Isochrone overlays
    # ------------------------------------------------------------------ #
    if iso_filepaths is None:
        iso_filepaths = [None] * len(iso_ages_myr)

    for age_myr, filepath, color, ls, lw in zip(
        iso_ages_myr, iso_filepaths,
        _ISO_COLORS[:len(iso_ages_myr)],
        _ISO_STYLES[:len(iso_ages_myr)],
        _ISO_WIDTHS[:len(iso_ages_myr)],
    ):
        # Try real MIST file; fall back to placeholder
        try:
            if filepath is not None:
                iso_df = load_mist_isochrone(filepath, age_myr)
                source = "MIST"
            else:
                raise FileNotFoundError("No file provided; using placeholder.")
        except FileNotFoundError as exc:
            log.info("  %s — using analytic placeholder isochrone.", exc)
            iso_df = generate_placeholder_isochrone(
                age_myr, metallicity_feh=metallicity_feh
            )
            source = "analytic approx."

        # Clip isochrone to plot range
        clip = (
            (iso_df["bp_rp"] >= bprp_min - 0.1)
            & (iso_df["bp_rp"] <= bprp_max + 0.1)
        )
        iso_clip = iso_df[clip]

        if len(iso_clip) < 2:
            log.warning("Isochrone at %d Myr has no points in plot range.", age_myr)
            continue

        ax.plot(
            iso_clip["bp_rp"].values,
            iso_clip["abs_g"].values,
            color     = color,
            linestyle = ls,
            linewidth = lw,
            label     = f"{age_myr} Myr  ({source})",
            zorder    = 6,
        )

    # ------------------------------------------------------------------ #
    # Region annotations
    # ------------------------------------------------------------------ #
    annotation_kw = dict(
        fontsize=10, fontstyle="italic", color="0.35",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
    )
    ax.annotate("Main Sequence",   xy=(0.85, 8.0),   **annotation_kw)
    ax.annotate("Subgiant Branch", xy=(0.80, 3.8),   **annotation_kw)
    ax.annotate("Giant Branch",    xy=(1.55, 1.2),   **annotation_kw)
    ax.annotate("White Dwarfs",    xy=(0.40, 13.5),  **annotation_kw)

    # Spectral type labels at top
    _spectral_labels = [
        (0.30, "A"),
        (0.60, "F"),
        (0.85, "G"),
        (1.40, "K"),
        (2.30, "M"),
    ]
    for bprp_val, sp_type in _spectral_labels:
        ax.annotate(
            sp_type, xy=(bprp_val, ax.get_ylim()[0] if ax.get_ylim() else -3),
            xycoords=("data", "axes fraction"),
            xytext=(bprp_val, 1.025), textcoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="0.3",
        )

    # ------------------------------------------------------------------ #
    # Axes formatting
    # ------------------------------------------------------------------ #
    # Standard HR diagram convention: y-axis INVERTED (bright stars at top)
    ax.invert_yaxis()
    ax.set_xlim(bprp_min, bprp_max)

    mg_all = plot_df["abs_g"].dropna()
    if len(mg_all):
        ax.set_ylim(
            max(mg_all.max() + 1.5, 15.0),
            min(mg_all.min() - 1.5, -3.0),
        )

    ax.set_xlabel(r"Colour  $G_{\rm BP} - G_{\rm RP}$  [mag]", labelpad=8)
    ax.set_ylabel(r"Absolute magnitude  $M_G$  [mag]",          labelpad=8)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2.0))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

    # ------------------------------------------------------------------ #
    # Colour bar
    # ------------------------------------------------------------------ #
    sm = ScalarMappable(cmap=_CMAP_CMD, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$G_{\rm BP} - G_{\rm RP}$  [mag]", fontsize=11)
    cbar.ax.invert_yaxis()    # hot (blue) at top of colour bar
    cbar.ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.2))
    cbar.ax.tick_params(labelsize=9)

    # ------------------------------------------------------------------ #
    # Title and legend
    # ------------------------------------------------------------------ #
    n_members = len(plot_df)
    title_str = (
        f"Hertzsprung–Russell Diagram — Hyades Cluster\n"
        f"Gaia DR3 ·  N = {n_members} members  "
        f"·  [Fe/H] = {metallicity_feh:+.2f}"
    )
    ax.set_title(title_str, fontsize=_TITLE_SIZE, pad=14)

    legend = ax.legend(
        loc            = "lower right",
        framealpha     = 0.92,
        edgecolor      = "0.6",
        handlelength   = 2.4,
        borderpad      = 0.8,
        labelspacing   = 0.45,
        title          = "Isochrones (MIST, [Fe/H]={:+.2f})".format(metallicity_feh),
        title_fontsize = 9,
    )
    legend.get_frame().set_linewidth(0.8)

    # ------------------------------------------------------------------ #
    # Optional tidal radius annotation
    # ------------------------------------------------------------------ #
    if tidal_radius_pc is not None and tidal_n_inside is not None:
        ax.text(
            0.04, 0.04,
            f"Stars within tidal radius ({tidal_radius_pc:.1f} pc): {tidal_n_inside}",
            transform=ax.transAxes, fontsize=9,
            color="0.4", ha="left",
        )

    # ------------------------------------------------------------------ #
    # Perryman reference annotation
    # ------------------------------------------------------------------ #
    ax.text(
        0.04, 0.98,
        "After Perryman et al. (1997, A&A 331, 81)\nGaia DR3 replication",
        transform=ax.transAxes, fontsize=8,
        color="0.5", ha="left", va="top", fontstyle="italic",
    )

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info("HR diagram saved to '%s'.", save_path)

    plt.tight_layout()
    return fig, ax


# =========================================================================== #
#  CONVENIENCE WRAPPER
# =========================================================================== #
def run_hr_pipeline(
    members_df: pd.DataFrame,
    iso_ages_myr: list[float] = _ISO_AGES_MYR,
    iso_filepaths: list[str | None] | None = None,
    save_path: str | Path | None = "hyades_hr_diagram.pdf",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Full Module 4 pipeline:
        members  →  absolute magnitudes  →  HR diagram + isochrones.
    """
    df = compute_absolute_magnitudes(members_df)
    fig, ax = plot_hr_diagram(
        df,
        iso_ages_myr  = iso_ages_myr,
        iso_filepaths = iso_filepaths,
        save_path     = save_path,
        **kwargs,
    )
    log.info("Module 4 complete.")
    return fig, ax


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N   = 150

    # ── Synthesise a Hyades-like CMD locus ──────────────────────────────── #
    # Main sequence (dominant)
    n_ms  = 110
    bprp_ms = rng.uniform(0.35, 2.8, n_ms)
    abs_g_ms = (-0.48 + 3.82 * bprp_ms - 0.10 * bprp_ms**2
                + rng.normal(0, 0.10, n_ms))

    # Subgiant / giant branch
    n_sg  = 25
    bprp_sg  = rng.uniform(0.70, 1.60, n_sg)
    abs_g_sg = rng.uniform(1.5, 4.5, n_sg)

    # White dwarf candidates
    n_wd  = 5
    bprp_wd  = rng.uniform(0.1, 0.5, n_wd)
    abs_g_wd = rng.uniform(11.5, 14.5, n_wd)

    bprp_all = np.concatenate([bprp_ms, bprp_sg, bprp_wd])
    absg_all = np.concatenate([abs_g_ms, abs_g_sg, abs_g_wd])
    plx_all  = rng.normal(21.5, 0.3, N)

    mock = pd.DataFrame({
        "bp_rp":           bprp_all,
        "abs_g":           absg_all,
        "phot_g_mean_mag": absg_all + 5 * np.log10(1000.0 / plx_all) - 5,
        "parallax":        plx_all,
        "parallax_error":  rng.uniform(0.03, 0.15, N),
        "phot_bp_mean_mag": np.zeros(N),   # placeholder
        "phot_rp_mean_mag": np.zeros(N),   # placeholder
        "abs_g_err":       rng.uniform(0.01, 0.08, N),
    })

    fig, ax = run_hr_pipeline(
        mock,
        iso_ages_myr=[600, 625, 650],
        save_path="hyades_hr_diagram.pdf",
        tidal_radius_pc=9.0,
        tidal_n_inside=124,
    )
    plt.show()
    print("Module 4 complete — HR diagram generated.")
