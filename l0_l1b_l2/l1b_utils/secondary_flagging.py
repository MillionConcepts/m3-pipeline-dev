import numpy as np
import warnings
from scipy import signal
from scipy.ndimage import label
from typing import Literal

# Functions below copied from Million Concept's Moonbow project


def binceil(value):
    rep = bin(value)
    sign = -1 if rep.startswith("-") else 1
    exponent = rep[3:] if sign == -1 else rep[2:]
    return 2 ** len(exponent) * sign


def make_fastkde_axis(array, nunique, downsample=1, padding=0.01):
    # lazily avoiding overflow issues for signed dtypes
    # w/negative values
    vrange = int(array.min()), int(array.max())
    pad = (vrange[1] - vrange[0]) * padding / 2
    padrange = int(vrange[0] - pad), int(vrange[1] + pad)
    return np.linspace(
        padrange[0], padrange[1], binceil(int(nunique / downsample)) + 1
    )


def make_floatkde_axis(array, padding, downsample, nunique):
    vrange = array.min(), array.max()
    pad = (vrange[1] - vrange[0]) * padding / 2
    return np.linspace(
        vrange[0] - pad,
        vrange[1] + pad,
        binceil(int(nunique / downsample)) + 1,
    )


def find_connected_1d(array, split_threshold=1, window_threshold=5):
    splits = np.uint8(
        ~np.concatenate([np.array([False]), np.diff(array) > split_threshold])
    )
    groups, group_ix = [], []
    labels, _ = label(splits)
    for lab, count in zip(*np.unique(labels, return_counts=True)):
        if lab == 0:
            continue
        if count < window_threshold:
            continue
        group_ix.append(np.nonzero(labels == lab)[0])
        groups.append(array[group_ix[-1]])
    return groups, group_ix


def make_kde(array, kernel_downsampling="auto", kernel_padding=0.05, uc=None):
    if array.ndim > 1:
        array = array.ravel()
    if uc is None:
        uc = np.unique(array, return_counts=True)
    if kernel_downsampling == "auto":
        kernel_downsampling = np.ceil(len(uc[0]) / 2500)
    if array.dtype.kind == "f":
        sample_points = make_floatkde_axis(
            array, kernel_padding, kernel_downsampling, len(uc[0])
        )
    else:
        sample_points = make_fastkde_axis(
            array,
            len(uc[0]),
            kernel_downsampling,
            kernel_padding,
        )
    from fastkde.fastKDE import fastKDE

    kde = fastKDE(array, axes=[sample_points])
    return array, kde, uc


def find_peak_stats(values, cutoff=0.01):
    peaks = signal.find_peaks(values, prominence=cutoff)[0]
    match = np.isin(np.arange(len(values)), peaks)
    y, x = np.zeros(len(values)), np.arange(len(values))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prom, _, _ = signal.peak_prominences(values, peaks)
        width, height, _, _ = signal.peak_widths(values, peaks)
    py = y.copy()
    py[np.nonzero(match)] = prom
    wy = y.copy()
    wy[np.nonzero(match)] = width
    hy = y.copy()
    hy[np.nonzero(match)] = height
    # fprom, fwidth, fheight, fprom_y, fwidth_y, fheight_y, fx
    return {
        "prom": np.flip(np.argsort(py)),
        "width": np.flip(np.argsort(wy)),
        "height": np.flip(np.argsort(hy)),
        "prom_y": py,
        "width_y": wy,
        "height_y": hy,
        "x": x,
    }


def kernel_spikes(
    array,
    uc=None,
    diff=None,
    # raw_sigma=2,
    # raw_abs=None,
    h_sigma=2,
    kernel_padding=0.1,
    kernel_downsampling="auto",
    median_threshold=10,
    max_consecutive=3,
    return_kde=False,
):
    array, kde, uc = make_kde(array, kernel_downsampling, kernel_padding, uc)
    x, pdf = kde.axes[0], kde.pdf
    kde_bins = np.digitize(uc[0], x, right=True)
    # but this will find _outliers_ too....
    # ...maybe not if we tune it high enough.
    peak_stats = find_peak_stats(pdf, 0.0001)
    outlier_pred = np.nonzero(
        peak_stats["height"] > pdf.mean() + h_sigma * pdf.std()
    )[0]
    match = uc[0][np.isin(kde_bins, outlier_pred)]
    if max_consecutive is not None:
        if diff is None:
            diff = np.diff(array)
        runs = find_connected_1d(
            match,
            split_threshold=np.median(np.abs(diff[diff != 0]))
            * median_threshold,
            window_threshold=max_consecutive,
        )[0]
        if len(runs) > 0:
            match = match[~np.isin(match, np.concatenate(runs))]
    #         return runs, None
    if return_kde is True:
        return match, kde
    return match

############

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
    return np.where(flag_ratios > cutoff)[0]


def build_bad_col_map(
        obs_image: np.ndarray,
        sigma: float,
        flag_col_ratio: float
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
        indices = flag_whole_cols(
            obs_image[:, band, :],
            sigma,
            flag_col_ratio,
            band
        )
        # make an image of the resulting flag map
        # could propagate ratios from flag_whole_cols? instead of just 0 or 1
        # (not bool here just for easy save to fits)
        flag_map[band, indices] = 1
    return flag_map
