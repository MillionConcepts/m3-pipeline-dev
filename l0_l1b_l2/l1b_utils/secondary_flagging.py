import numpy as np
import warnings
from scipy import signal
from scipy.ndimage import label

############ Functions below copied from Million Concept's Moonbow project


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


def neighboring_col_diff(obs_image, axis=1, side="left"):
    """
    Subtract neighboring column from image for flagging purposes.
    """
    shift = 1 if side == "left" else -1
    neighbor = np.roll(obs_image, shift, axis=axis)
    return obs_image - neighbor
