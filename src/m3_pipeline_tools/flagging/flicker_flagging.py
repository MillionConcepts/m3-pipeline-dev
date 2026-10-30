"""
Functions for identifying flickering electronic background noise that occurs
in the same sets of detector columns across all bands during an observation.
"""

from typing import Literal
import warnings
import numpy as np
import pandas as pd


def transform_bad_column_groups(
        bad_col_groups: dict,
        reverse_samples: bool,
        left_col_cutoff: int,
        l1b_samples: int,
) -> dict:
    """
    Bad col groups are stored in the reference file / moonager in L0 indices
    and orientation. To check for bad cols in L1B files we have to transform
    them.

    Args:
        bad_col_groups: Arrays of hand-identified bad col indices that flash
            on / off in the detector background.
        reverse_samples: Sample orientation in L1B.
        left_col_cutoff: Sample cutoff in L1B.
        l1b_samples: Samples in L1B.
    """
    l1b_bad_col_groups = {}
    for key, idx in bad_col_groups.items():
        # trim to L1B width
        idx = np.asarray(idx) - left_col_cutoff - 1
        # flip if reverse_samples is True
        if reverse_samples:
            idx = l1b_samples - 1 - idx
        # drop cols outside L1B bounds
        l1b_bad_col_groups[key] = idx[(idx >= 0) & (idx < l1b_samples)]
    return l1b_bad_col_groups


def get_median(a, axis=None):
    """ Ignore NaN warnings that would happen for very hot observations. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def get_per_line_step(
        per_line_signal: np.ndarray,
        win_size: int
) -> np.ndarray:
    """ For each line, subtract median of window after and before. """
    # could use scipy ndimage median filter but would need to mask NaNs
    from numpy.lib.stride_tricks import sliding_window_view

    lines = len(per_line_signal)
    padded_signal = np.pad(per_line_signal, win_size, constant_values=np.nan)
    win = sliding_window_view(padded_signal, win_size)
    prev = get_median(win[:lines], axis=1)
    nxt = get_median(win[win_size:win_size + lines], axis=1)
    return nxt - prev


def find_start_stop(
        per_line_flicker_signal: np.ndarray,
        peak: int,
        window: int
) -> int:
    """
    Found peaks identify the start / stop area of a flicker on / off, but
    can be a few pixels off from the exact stop / start index.
    Here we look for the exact start / stop index by finding the index with
    the largest offset to either side of the per line flicker signal.
    """
    lines = len(per_line_flicker_signal)
    step_pos, step_offset = peak, 0

    # search half a window size either side of current peak for the best peak
    for pos in range(max(peak - window // 2, 1),
                     min(peak + window // 2, lines - 1) + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # subtract left and right side of current possible peak index, pos
            offset = abs(
                np.nanmean(per_line_flicker_signal[pos:pos + window]) -
                np.nanmean(per_line_flicker_signal[max(pos - window, 0):pos])
            )
        if np.isfinite(offset) and offset > step_offset:
            # largest offset to either side = most likely the real stop / start
            # of a flicker group
            step_pos, step_offset = pos, offset
    return step_pos


def neighborhood_diff(
        detection_band: np.ndarray,
        cols: np.ndarray,
        n_neighbors: int,
        neighbor_stat: Literal["mean", "median"],
) -> np.ndarray:
    """
    Get difference of bad cols we're interested in with cols to either side.
    Can use average or median of neighbors. If n_neighbors = 1, median == mean.

    secondary_flagging has a similar function called neighbor_diff, but that
    gets the difference for all cols in the array with only the closest col.

    Args:
        detection_band: Image of a band or mean of multiple bands for
            flickering detection.
        cols: Array of col indices.
        n_neighbors: How many cols to either side of each 'bad' col to compare
            it with.
        neighbor_stat: Use mean or median to get neighborhood value of
            n_neighbors. If n_neighbors == 1, mean == median.

    Returns:
         array of shape lines x num bad cols.
    """
    n_lines, n_cols = detection_band.shape

    # median more robust to outliers like crater edges etc
    stat = np.nanmean if neighbor_stat == "mean" else np.nanmedian

    # lines x num bad cols in a group
    out = np.full((n_lines, len(cols)), np.nan)

    # there will be NaNs in very hot observations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j, c in enumerate(cols):
            # only use neighboring cols that exist in the image
            neighbors = [k for k in range(max(c - n_neighbors, 0),
                                          min(c + n_neighbors + 1, n_cols))
                         if k != c]
            if neighbors:
                out[:, j] = (detection_band[:, c]
                             - stat(detection_band[:, neighbors], axis=1))
    return out


def mask_neighborhood_diffs(
        detection_band: np.ndarray,
        normal_diffs: np.ndarray,
        normal_cols: np.ndarray,
        diffs: np.ndarray,
        cols: np.ndarray,
        bad_bright: float,
        clip_k: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mask diffs for scene noise and super bright stuff. Uses both the flickering
    col diffs and the normal col diffs. Need the normal col diffs to set the
    level at which to clip values.
    """
    from scipy import stats

    # 1.4826 x MAD
    robust_sigma = stats.median_abs_deviation(
        normal_diffs,
        scale='normal',
        axis=None,
        nan_policy='omit'
    )
    clip = clip_k * robust_sigma

    diffs[detection_band[:, cols] > bad_bright] = np.nan
    diffs[np.abs(diffs) > clip] = np.nan

    normal_diffs[detection_band[:, normal_cols] > bad_bright] = np.nan
    normal_diffs[np.abs(normal_diffs) > clip] = np.nan

    return diffs, normal_diffs


def check_flicker_group(
        group_name: str,
        detection_band: np.ndarray,
        cols: np.array,
        normal_cols: np.array,
        bad_bright: float,
        n_neighbors: int = 2,
        neighbor_stat: Literal["mean", "median"] = "median",
        window: int = 25,
) -> dict:
    """
    Identify lines in the detection band where the given group of cols is
    flickering on or off. Also find the magnitude of the flicker. Flickering
    is determined by comparison with a set of 'normal' cols at each line.

    Args:
        group_name: Name of col group being checked.
        detection_band: Image of a band or mean of multiple bands for
            flickering detection.
        cols: Array of flickering col indices.
        normal_cols: Array of col indices with no known flickering.
        bad_bright: 99.5 percentile of detection band.
        n_neighbors: How many cols to either side of each 'bad' col to compare
            it with.
        neighbor_stat: Use mean or median to get neighborhood value of
            n_neighbors. If n_neighbors == 1, mean == median.
        window: How many lines before and after each line to compare with.
    """
    from scipy import stats
    from scipy.signal import find_peaks

    n_lines, n_cols = detection_band.shape

    # get offsets with neighboring columns for the known bad group and its
    # normal/control group
    diffs = neighborhood_diff(
        detection_band=detection_band,
        cols=cols,
        n_neighbors=n_neighbors,
        neighbor_stat=neighbor_stat
    )
    normal_diffs = neighborhood_diff(
        detection_band=detection_band,
        cols=normal_cols,
        n_neighbors=n_neighbors,
        neighbor_stat=neighbor_stat
    )

    # mask diffs for scene noise and super bright stuff
    diffs, normal_diffs = mask_neighborhood_diffs(
        detection_band=detection_band,
        normal_diffs=normal_diffs,
        normal_cols=normal_cols,
        diffs=diffs,
        cols=cols,
        bad_bright=bad_bright,
    )

    # median of cols across each line (once for the flickering col group and
    # once for the 'normal' group)
    per_line_flicker_signal = get_median(diffs, axis=1)
    per_line_normal_signal = get_median(normal_diffs, axis=1)

    # find points where the flicker signal steps up or down
    flicker_signal_steps = get_per_line_step(per_line_flicker_signal, window)
    normal_signal_steps = get_per_line_step(per_line_normal_signal, window)

    # can't identify groups at very ends of observation because of window size
    for s_ in (flicker_signal_steps, normal_signal_steps):
        s_[:window // 2] = np.nan
        s_[n_lines - window // 2:] = np.nan

    # steps ID'd in the 'normal' signal are not real flicker, therefore real
    # flicker steps will be bigger than the steps in the normal signal.
    # here we set 6.0 x (1.4826 x MAD of normal step sizes) for a step
    # threshold. maybe move up or down after testing on hot obs.
    min_step_size = 6.0 * stats.median_abs_deviation(
        normal_signal_steps,
        scale='normal',
        axis=None,
        nan_policy='omit'
    )

    # ID peaks in the flicker signal that have step size > than the threshold
    # based on the normal signal steps, defined above. can't find peaks within
    # same window area.
    peaks, _ = find_peaks(
        np.nan_to_num(np.abs(flicker_signal_steps)),
        height=min_step_size,
        distance=window
    )

    # find start / stop of flickering
    peaks = np.array([
        find_start_stop(per_line_flicker_signal, peak, window)
        for peak in peaks],
        dtype=int
    )

    # starts and stops of all flicker segments
    starts = np.r_[0, peaks]
    ends = np.r_[peaks, n_lines]

    # median signal over all lines in a flicker on / off segment
    flicker_mag = np.array([get_median(per_line_flicker_signal[s:e])
                            for s, e in zip(starts, ends)])

    # threshold for flagging segment if super noisy
    norm_line_noisiness = 4.0 * stats.median_abs_deviation(
        per_line_normal_signal,
        scale='normal',
        nan_policy='omit'
    )

    return pd.DataFrame({
        "group": group_name,  # same as in reference / moonager
        "start_line": starts,
        "end_line": ends - 1,
        "n_lines": ends - starts,
        "flicker_mag": flicker_mag,  # median for all lines / cols in group
        "flagged": np.abs(flicker_mag) > norm_line_noisiness,
    })


def get_normal_cols(cols: np.array, n_cols: int):
    """
    Apply random-ish offset to col group indices to get a set of 'normal' cols
    for offset comparison. Like a control set of cols that we know don't
    show the same flickering pattern.

    Args:
        cols: Flickering col group.
        n_cols: Offset, in pixels, to add to col group.
    """
    normal_cols = cols + 5
    # add check that we don't overlap another bad col group?
    # but at the moment I know we don't so for now it's fine unless more
    # groups get defined in the future
    cut = normal_cols >= n_cols
    normal_cols = normal_cols[~cut]
    return normal_cols


def find_background_flicker(
        detection_band: np.ndarray,
        col_groups: dict,
) -> pd.DataFrame:
    """
    Identifies where groups of columns intermittently turn on and off in
    the background across all bands. Here we try to identify the change points
    at which each group of cols flickers on or off. That info can be used for
    flagging or to try and remove the effect.

    About this flickering effect:
        - Previously in this pipeline they were referred to as 'bad column
          groups'.
        - The change happens simultaneously across sets of columns for multiple
          lines (generally a few to several hundred lines long).
        - These columns are equally spaced across the detector.
        - They can be brighter or darker than the surrounding detector
          background.
        - Because this pattern varies with time, it cannot be removed by flat
          fielding or dark signal subtraction.
        - This is probably an electronic effect.
        - The DN magnitude of the offset is close to the same across all bands
          (about -5 to 5 or so DN), however in L1B due to calibration the
          magnitude and visibility of the effect is not the same across all
          bands.
        - This flickering effect is most evident in cool observations
          (less noise) and low bands (less 'real' data, lower scene DNs).
        - Because we think it is part of the electronic background, this is an
          ADDITIVE effect, not multiplicative.
        - Sometimes it is (unfortunately) propagated to the flat field.

    Args:
        detection_band: Observation shape array, average of multiple bands or
            just one.
        col_groups: Dict of flickering bad col groups that have been
            transformed to the shape / orientation of the detection band.
    """

    _, n_cols = detection_band.shape  # changes with L0 or L1B input

    # ignore very bright parts of the image
    bad_bright = np.nanpercentile(detection_band, 99.5)

    # for each group of possible flickering columns, identify flickering
    # segments using an offset group of 'normal' columns
    segments = [
        check_flicker_group(
            group_name=group_name,
            detection_band=detection_band,
            cols=cols,
            normal_cols=get_normal_cols(cols, n_cols),
            bad_bright=bad_bright,
        )
        for group_name, cols in col_groups.items()
    ]

    return pd.concat(segments, ignore_index=True)


def make_flicker_map(
        segments: pd.DataFrame,
        col_groups: dict,
        n_lines: int,
        n_cols: int,
) -> np.ndarray:
    """
    Based on identified flickering, make a flag map.

    0 = not a bad-group column
    1 = bad-group column but no flicker detected for this line
    2 = bump
    3 = drop

    All cols in a group get the same flag value for each line.

    Args:
        segments: Identified groups of flickering line segments
        col_groups: Indices of bad cols in each group of flickering segments.
        n_lines: Lines in observation.
        n_cols: Samples in observation.
    """
    from make_flag_backplanes import NO_FLICKER, FLICKER_DROP, FLICKER_BUMP

    flag_map = np.full((n_lines, n_cols), 0, dtype=np.uint8)
    # we flag even if there's no detected offset
    for cols in col_groups.values():
        flag_map[:, cols] = NO_FLICKER
    # for each flagged segment, indicate if bump or drop
    for seg in segments[segments.flagged].itertuples():
        rows = slice(seg.start_line, seg.end_line + 1)
        flag_map[rows, col_groups[seg.group]] = (
            FLICKER_BUMP if seg.flicker_mag > 0.0 else FLICKER_DROP)
    return flag_map


def make_offset_map(
        segments: pd.DataFrame,
        col_groups: dict,
        n_lines: int,
        n_cols: int,
) -> np.ndarray:
    """
    Based on identified flickering, make an offset map using median offset
    per flickering group.

    All cols in a group get the same value for each line within a flicker.

    Args:
        segments: Identified groups of flickering line segments
        col_groups: Indices of bad cols in each group of flickering segments.
        n_lines: Lines in observation.
        n_cols: Samples in observation.
    """
    offset_map = np.full((n_lines, n_cols), 0, dtype=np.float32)
    for seg in segments[segments.flagged].itertuples():
        rows = slice(seg.start_line, seg.end_line + 1)
        offset_map[rows, col_groups[seg.group]] = seg.flicker_mag
    return offset_map


def get_flicker_maps(
        obs_image: np.ndarray,
        bad_col_groups: dict,
        offset: bool,
        reverse_samples: bool,
        left_col_cutoff: int,
        l1b_samples: int,
        bands: slice = slice(1, 3)
):
    """
    Identify flickering columns in L1B or L0 images and make a flag map from
    them.

    If you wanted to get a map or offset values for every band, this would
    need to be run per band.

    Args:
        obs_image: Observation data in lines x bands x samples/cols. Can be
            L1B or L0 shape.
        bad_col_groups: Arrays of hand-identified bad col indices that flash
            on / off in the detector background.
        offset (bool): If true, output offset map as well that gives additive
            values to try and remove the flickering.
        reverse_samples: Sample orientation in L1B.
        left_col_cutoff: Sample cutoff in L1B.
        l1b_samples: Samples in L1B.
        bands: Slice of band indices to use for making flicker flag maps.
    """
    n_lines, n_bands, n_cols = obs_image.shape

    detection_band = np.mean(obs_image[:, bands, :], axis=1)

    # check if it's l1b based on shape for now, change bad col groups to l1b
    # size and orientation if true
    if n_cols == l1b_samples:
        bad_col_groups = transform_bad_column_groups(
            bad_col_groups=bad_col_groups,
            reverse_samples=reverse_samples,
            left_col_cutoff=left_col_cutoff,
            l1b_samples=l1b_samples,
        )

    # ID flickering in given groups using a set of bands
    segments = find_background_flicker(
        detection_band=detection_band,
        col_groups=bad_col_groups,
    )

    # make flicker map
    flicker_map = make_flicker_map(
        segments=segments,
        col_groups=bad_col_groups,
        n_lines=n_lines,
        n_cols=n_cols,
    )

    # optionally make additive offset map for adding to image
    # probably not a good idea to use for l1b UNLESS running per band, and
    # even then questionable
    if offset:
        offset_map = make_offset_map(
            segments=segments,
            col_groups=bad_col_groups,
            n_lines=n_lines,
            n_cols=n_cols,
        )
        return flicker_map, offset_map

    return flicker_map, None
