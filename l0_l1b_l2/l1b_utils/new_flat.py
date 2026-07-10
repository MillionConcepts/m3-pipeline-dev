"""
Exploring different ideas for how to make a flat field to avoid using two
flat fields like in the original mission pipeline (lab flat and obs-based
flat). 
The mission made their obs-based flat by: 
'Image based flat field correction values were derived by averaging the longest
 on-orbit data sets and then dividing by the cross-track average value. This 
 simple approach also removed the cross-track photometric signal. To retain 
 the cross-track photometry, a two-dimensional plane is fit to the image based 
 flat field and retained in the image based flat field correction factor.
 To suppress the impact of features on the lunar surface in the image based
 flat field, a smoothed spectral average of the function is divided out in a
 final step.'

Given that we know the ripples at the top of the lab flat field 'move' with
different observation angles, having a single flat is preferable to two flats
where the second flat is undoing some portion of the first lab flat. Making a
single flat from an observation is complicated by the impact of lunar surface
features as noted above.
"""
from typing import Literal
import numpy as np


def make_relative_gain_flat(
        obs_image: np.ndarray,
        direction: Literal["left", "right"],
        left_cutoff: int,
        right_cutoff: int,
        readout_cols: list,
        saturation=None,
        min_valid_frac=0.5,
        min_dynamic_range=None
) -> np.ndarray:
    """
    Use the relative difference between adjacent columns to make a flat field
    normalized to the gain (mult factor) between columns being 1 per channel.
    Args:
        obs_image: Full observation, could be in DN or radiance. Must be dark
            signal subtracted otherwise the results will be dominated by dark
            signal.
        direction: Which adjacent column to use for relative diff (left or
            right). Defaults to left. NOT USED RN.
        left_cutoff: Left side of detector to mask (not illuminated).
        right_cutoff: Right side of detector to mask (not illuminated).
        readout_cols: Readout cols to mask.
        saturation: Mask high values above this threshold.
        min_valid_frac: What fraction of lines should be valid (e.g. not
            saturated or 0 std) to use in the flat?
        min_dynamic_range: We should ignore columns with variance below this.
    """
    nlines, nbands, ncols = obs_image.shape
    gain = np.ones((nbands, ncols), dtype=np.float64)

    for b in range(nbands):
        band_data = obs_image[:, b, :]  # copy per band for masking
        # mask bad pixels
        bad = ~np.isfinite(band_data) | (band_data <= 0)
        # mask "saturated" pixels
        if saturation is not None:
            bad |= (band_data >= saturation)
        # mask bad cols, cutoff dark signal / scattered light edges
        bad[:, :left_cutoff] = True
        bad[:, right_cutoff:] = True
        if readout_cols:
            bad[:, readout_cols] = True
        band_data = np.where(bad, np.nan, band_data)

        # we put the DN in log space bc log x - log y = log(x/y)
        log_data = np.log(band_data)
        log_ratio_pairs = log_data[:, 1:] - log_data[:, :-1]

        # don't use columns with low std? we don't want noise to be
        # interpreted as real, reoccurring column to column differences
        if min_dynamic_range is not None:
            local_std = np.nanstd(log_data, axis=1, keepdims=True)
            flat_rows = local_std < min_dynamic_range
            log_ratio_pairs = np.where(flat_rows, np.nan, log_ratio_pairs)

        # Median across lines, ignoring NaNs
        valid_frac = np.mean(np.isfinite(log_ratio_pairs), axis=0)
        with np.errstate(invalid="ignore"):
            median_log_ratio = np.nanmedian(log_ratio_pairs, axis=0)

        # use 0.0 at bad pixels / where there are not enough downline samples
        # to get a good estimate of relative gain
        median_log_ratio = np.where(valid_frac >= min_valid_frac,
                                    median_log_ratio, 0.0)
        median_log_ratio = np.nan_to_num(median_log_ratio, nan=0.0)

        combined_gains = np.concatenate((
                [0.0],
                np.cumsum(median_log_ratio)
            ))

        # normalize to median gain of 1
        combined_gains -= np.median(combined_gains)
        gain[b, :] = np.exp(combined_gains)
    return gain


def get_relative_gain_flat(obs_image: np.ndarray, moonager) -> np.ndarray:
    """
    Take the average of two flats based on the relative gain to the left and
    to the right of each column.

    Kind of like histogram matching but not using the actual histogram of each
    column, instead using each column's relationship to its neighbors?

    This will not work if there are long, vertical, geographic / geologic
    features on the Moon that are the width of a M3 column and extend for most
    of an observation.
    """

    flat_1 = make_relative_gain_flat(
        obs_image,
        direction='left',
        left_cutoff=moonager.left_col_cutoff,
        right_cutoff=moonager.right_col_cutoff,
        readout_cols=moonager.read_out_cols,
        min_valid_frac=0.5,
    )

    # flat_2 = make_relative_gain_flat(
    #     obs_image,
    #     direction='right',
    #     left_cutoff=moonager.left_col_cutoff,
    #     right_cutoff=moonager.right_col_cutoff,
    #     readout_cols=moonager.read_out_cols,
    #     min_valid_frac=0.5,
    # )

    #return ((1.0 / flat_1) + flat_2) / 2.0

    return 1.0 / flat_1
