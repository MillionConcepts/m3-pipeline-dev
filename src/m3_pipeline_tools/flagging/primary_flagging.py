""" Functions for flagging L0 data. """

import numpy as np
from pathlib import Path


def build_bde_mask(
        dark_path: Path,
        obs_type,
        readout_cols: list,
        filter_seam_rows: list
):
    """
    Make a BDE mask for a target or global observation. Thresholds named below
    are deduced from calculating the mean and std for the L0 dark and comparing
    to their mission-produced BDE file. The order in which flags were assigned
    is unknown, and there are unexplained apparent differences in flag meanings
    for observation type (ie target vs global).

    It was also discovered there is one pixel in the global flat that is always
    flagged as flag 1. There could be more like this, but I haven't found them
    yet.

    Global
    0: < 2.2 DN std, < 1000 DN mean
    1: < 2.2 DN std, > 1000 DN mean
    2: > 2.2 DN std, < 1000 DN mean
    3: > 2.2 DN std, > 1000 DN mean
    4:  Filter seams

    Target
    0: < 5.0 DN std, 1000 > x > 300 DN mean
    1: < 5.0 DN std, > 1000 DN mean (could just be mean?)
    2: < 5.0 DN std, < 300 DN mean
    3: > 5.0 DN std, x < 1000 DN mean
    4:  Filter seams, > 5.0 DN std, > 1000 DN mean
    """
    from ..utils.loader import load_fits_into_frame

    # could do the following with make_dark_signal_image but would need
    # to decide how to treat negative values for all other dark signal uses
    # (although most of those were never processed to L1B bc they were
    # very hot)

    # crop beginning and end of loaded dark
    dark = load_fits_into_frame(dark_path)[1: -1, :, :]
    # remove -2 rollover values bc they mess with the stats
    dark[dark < 0.0] = np.nan

    # DPSIS says they used mean not median
    mean = np.nanmean(dark, axis=0)
    std = np.nanstd(dark, axis=0)
    has_nan = np.any(np.isnan(dark), axis=0)

    # has to be float type bc for whatever reason negative values are flag 1.5
    mask = np.zeros(mean.shape, dtype=float)

    if obs_type == "G":
        mask[(std < 2.2) & (mean < 1000)] = 0
        mask[(std < 2.2) & (mean > 1000)] = 1
        mask[(std >= 2.2) & (mean < 1000)] = 2
        mask[(std >= 2.2) & (mean >= 1000)] = 3

        # special case from looking at their masks + flat field
        mask[11, 256] = 1
    elif obs_type == "T":
        mask[(std < 5.0) & (mean > 300) & (mean < 1000)] = 0
        mask[(std < 5.0) & (mean > 1000)] = 1
        mask[(std < 5.0) & (mean < 300)] = 2
        mask[(std >= 5.0) & (mean < 1000)] = 3
        mask[(std >= 5.0) & (mean >= 1000)] = 4

    if readout_cols is not None:
        mask[:, readout_cols] = 1

    if filter_seam_rows is not None:
        mask[filter_seam_rows, :] = 4

    mask[has_nan] = 1.5

    return mask


def call_build_bde_mask(
        obs_id: str,
        local_root: str = "data",
):
    """
    Just a convenience function for testing only mission-based BDE mask
    building without running the whole pipeline.
    """
    from ..reference import PipeManager, check_observation

    obs_warn, obs_error, metadata = check_observation(obs_id)

    moonager = PipeManager(
        obs_id=obs_id,
        metadata=metadata,
        local_root=local_root,
        verbose=True,
    )
    return build_bde_mask(moonager.dark_path,
                          moonager.mode,
                          moonager.read_out_cols,
                          moonager.filter_seam_rows,
                          )


def get_l0_negative_flags(
        l0_path: Path,
):
    """
    Load L0, identify negative pixels, return obs as a bool array where
    True = negative in L1B shape and orientation.
    """
    from ..utils.loader import load_fits_into_frame

    l0 = load_fits_into_frame(l0_path)
    return l0 < 0
