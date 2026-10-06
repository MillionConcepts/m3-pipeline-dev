import numpy as np
# import warnings
# from scipy import signal
# from scipy.ndimage import label
from l0_l1b_l2.reference import check_shape
from typing import Literal
from pathlib import Path


# flag bad columns that span the whole observation
# this is a very simple heuristic: is the majority of this column
# (>60% of pixels) offset by more than median diffs + .5 std of image diffs


def neighbor_diff(
        image: np.ndarray,
        axis: int = 1,
        side: Literal["left", "right"] = "left"
):
    """
    Subtract neigboring columns for col-col offsets (good for flat evaluation
    and flagging).
    """
    shift = 1 if side == "left" else -1
    neighbor = np.roll(image, shift, axis=axis)
    return image - neighbor


def flag_by_std(diff_image: np.ndarray, sigma: float):
    """
    Flag all pixels in band col difference image based on median + sigma
    threshold.
    """
    std = abs(np.std(diff_image))
    median = abs(np.median(diff_image))
    flagged_pixels = abs(diff_image) > (median + std * sigma)
    return flagged_pixels.astype(int)


def flag_side(
        obs_band: np.ndarray,
        sigma: float,
        side: Literal["left", "right"]
):
    """ Call neighbor diff and run flagging. """
    oneway = neighbor_diff(obs_band, side=side)
    return flag_by_std(oneway, sigma)


def combo_side_flags(obs_band: np.ndarray, sigma: float):
    """
    Get flags for left col subtraction and right col subtraction, then combine.
    """
    # left side
    flagged_left = flag_side(obs_band, sigma, "left")
    # right side
    flagged_right = flag_side(obs_band, sigma, "right")
    return flagged_left + flagged_right


def flag_whole_cols(
        obs_band: np.ndarray,
        sigma: float,
        flag_col_ratio: float,
        band: int,
):
    """
    Returns indices of columns for a single band that are 'bad' for a specified
    minimum percentage of the column (60-70% is a good setting I think). 'Bad'
    means offset from the left and right columns by more than the median plus a
    specified sigma.
    """
    combo_flags = combo_side_flags(obs_band, sigma)
    counts_per_col = combo_flags.sum(axis=0)
    image_len = obs_band.shape[0]
    # percent of col that is flagged
    flag_ratios = (counts_per_col / 2) / image_len
    if np.percentile(flag_ratios, 96) > flag_col_ratio:
        print(
            f"The cutoff flag ratio given for col flagging, {flag_col_ratio}, "
            f"is lower than the 96th percentile pixels flagged per column, "
            f"{np.percentile(flag_ratios, 96)} for band {band}.")
    cutoff = max(flag_col_ratio, np.percentile(flag_ratios, 96))
    return np.where(flag_ratios > cutoff)[0], flag_ratios


def build_bad_col_map(
        obs_image: np.ndarray,
        sigma: float = .5,
        flag_col_ratio: float = .6
):
    """
    Build col x band flag map where flagged pixels are 'bad' across most lines
    of observation. Uses difference between neighboring cols, not ratios.

    Args:
        obs_image: Obs image data, could be rdn or DN.
        sigma: Sigma for threshold at which to flag.
        flag_col_ratio: What ratio of pixels in a col should be flagged before
            the col is flagged.
    """
    _, bands, cols = obs_image.shape
    flag_map = np.zeros((bands, cols), dtype=int)
    for band in range(bands):
        indices, ratios = flag_whole_cols(
            obs_image[:, band, :],
            sigma,
            flag_col_ratio,
            band
        )
        # make an image of the resulting flag map
        # could propagate ratios from flag_whole_cols? instead of just 0/1/2

        # adjacent flagged pixels are usually the result of one high contrast
        # col neighboring two normal ones. we could do something more complex
        # and expand the diff window and rerun the process, but for now
        # merely flagging standalone and highest ratio flagged pixels as
        # 2 ('real' bad pixels) and the neighboring pixels as 1 seems to
        # suffice
        if indices.size == 0:
            continue
        groups = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        for group in groups:
            flag_map[band, group] = 1
            # below means default 2 for groups of 1 pixel as a result
            flag_map[band, group[np.nanargmax(ratios[group])]] = 2
    return flag_map


def get_l0_negative_flags(
        l0_path: Path,
):
    """
    Load L0, identify negative pixels, return obs as a bool array where
    True = negative in L1B shape and orientation.
    """
    from .loader import load_fits_into_frame

    l0 = load_fits_into_frame(l0_path)
    return l0 < 0


def flag_l1b(
        paths: dict,
        omitted_channels: list[int],
        left_col_cutoff: int,
        right_col_cutoff: int,
        filter_seams: list[int],
        read_out_cols: list[int],
):
    """
    From L1B data, and optionally L0 data and mission BDE files, build an
    L1B-shaped flag map with in the shape lines x bands x samples.

    dict paths keys: "l1b_label", "l0_path", "l1b_path",  "flag_path"
    """
    from astropy.io import fits
    from l0_l1b_l2.reference import check_l1b_label
    from .make_backplanes import _bit
    from .loader import load_fits_into_frame

    # flag meanings
    nan = 1
    bde = 2
    block_col = 4
    bad_col = 8

    # If the L1B label or L0 data are not present, we make flag map arr based
    # on L1B shape. We check here to avoid having L0 data and L1B data loaded
    # at the same time later on. Otherwise, flags will get made when we check
    # L0 for negative pixels.
    if not (paths['l1b_label'] and paths['l0_path']):
        hdr = fits.getheader(paths['l1b_path'])
        flags = np.zeros(
            (hdr['NAXIS3'], hdr['NAXIS2'], hdr['NAXIS1']),
            dtype=np.uint8,
        )
    # Check label for orientation of L1B, so we can orient the original flag
    # map and L0 the same way
    if paths['l1b_label']:
        reverse_lines, reverse_samples = check_l1b_label(paths['l1b_label'])

        # bit 0: NaN in obs, set to NaN because of negative values in L0
        if paths['l0_path']:
            # Must load L0, flag negative, and then trim and orient correctly
            flags = get_l0_negative_flags(l0_path=paths['l0_path'])
            flags = flags[
                    :,
                    np.max(omitted_channels) + 1:,
                    left_col_cutoff:right_col_cutoff
                    ]
            if reverse_lines:
                flags = flags[::-1, :, :]
            if reverse_samples:
                flags = flags[:, :, ::-1]
            flags = _bit(flags==True, nan)

        # bit 1: interpolated pixel in BDE map
        if paths['flag_path']:
            # Need to load and trim / orient to L1B s
            bde_map = np.asarray(load_fits_into_frame(paths['flag_path']))
            bde_map = bde_map[
                      np.max(omitted_channels) + 1:,
                      left_col_cutoff:right_col_cutoff
                      ]
            if reverse_samples:
                bde_map = bde_map[:, ::-1]
            flags |= _bit(bde_map > 0, bde)[np.newaxis, :, :]

        # flagging for the following bits requires the L1B data
        obs_image = load_fits_into_frame(paths['l1b_path'])

        # bit 2: variable column block
        # bad col group map shape lines, samples
        # flags |= _bit(bad_col_group_map > 0, block_col)[:, np.newaxis, :]

        # bit 3: bad columns (ie bad flat or missed bright / dark pixel)
        # bad cols shape band, samples
        bad_cols = build_bad_col_map(obs_image)
        flags |= _bit(bad_cols > 1, bad_col)[np.newaxis, :, :]

    # return in original view of l1b
    return flags.transpose(1, 0, 2)
