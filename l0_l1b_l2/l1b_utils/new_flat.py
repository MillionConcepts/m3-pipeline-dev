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

    # return ((1.0 / flat_1) + flat_2) / 2.0

    return 1.0 / flat_1


def find_variable_column_blocks(
        obs_band_combo: np.ndarray,
        col_groups: dict,
        comp_window=1,
        min_offset=.6,
        max_offset=10.0,
        min_cols_frac=0.3,
        offset_type="auto",
        max_gap=10,
        min_block_size=20,
        end_threshold=1.0,
        end_window=2,
) -> np.ndarray:
    """
    Some evenly spaced groups of columns (typically separated by 16, 32, or 48
    pixels in global mode) are simultaneously higher or lower during an
    observation. They do not get flagged using the dark signal observations
    because their on / off flickering is usually 200-800 lines long. Dark
    observations are around 260 lines long.

    All columns in a group simultaneously increase or decrease at the same line
    and are 1-4 DN higher than neighboring columns across all bands (sometimes
    more at higher bands).

    Returns dictionary with all blocks of relatively high or low col behavior.

    Args:
        obs_band_combo: Median across bands of full observation, could be in DN
            or radiance. Must be dark signal subtracted otherwise the results
            will be dominated by dark signal.
        #TODO: is sum or mean better than median? could do band by band but
            #noisier then
        col_groups: Dictionary with lists of bad columns.
        comp_window: Which neighboring columns to compare bad col values to.
        min_offset: Minimum offset with neighboring columns to be considered
            a bump or dip.
        max_offset: Maximum offset with neighboring columns to be considered
            a bump or dip.
        min_cols_frac: What % of cols in the group must fit the criteria for
            bump or dip for it be considered a "block"
        offset_type: Designate if we are looking for drops, bumps or either.
            String.
        max_gap: How many lines can we skip within a block before it's not a
            block anymore? Most useful in a highly variable terrain context.
        min_block_size: Minimum number of lines in a block.
        end_threshold: Change in DN within a column to designate end of block.
            NOT relative to neighboring columns.
        end_window: Area within which the end could occur for all group cols.
    """
    n_lines, n_cols = obs_band_combo.shape

    block_results = {}

    for group_name, indices in col_groups.items():
        # unlikely but in case of empty groups
        if not indices:
            block_results[group_name] = {}
            continue

        # cols within a group are 1 indexed bc that's how they are in ds9
        cols = np.array(indices) - 1

        if isinstance(offset_type, str):
            offset_types = [offset_type] * len(cols)
        else:
            offset_types = offset_type

        # positive = drop (neighbors cols higher than col)
        # negative = bump (neighboring cols lower than col)
        offsets = np.full((n_lines, len(cols)), np.nan)

        for j, c in enumerate(cols):
            lo = max(c - comp_window, 0)
            hi = min(c + comp_window + 1, n_cols)
            neighbor_cols = [k for k in range(lo, hi) if k != c]
            # mean or median? median meaningless if it's two cols
            neighbor_mean = obs_band_combo[:, neighbor_cols].mean(axis=1)
            target_val = obs_band_combo[:, c]
            offsets[:, j] = neighbor_mean - target_val

        # vectorize?
        matches = np.zeros_like(offsets, dtype=bool)
        for j, d in enumerate(offset_types):
            col_offset = offsets[:, j]
            mag = np.abs(col_offset)
            in_range = (mag >= min_offset) & (mag <= max_offset)

            # we do know some cols tend to be higher rather than lower but
            # IDK if that's 100% true all the time
            if d == "drop":
                sign_ok = col_offset > 0
            elif d == "bump":
                sign_ok = col_offset < 0
            elif d == "auto":
                sign_ok = np.ones_like(col_offset, dtype=bool)
            else:
                raise ValueError(
                    f"invalid dir type '{d}'")

            matches[:, j] = in_range & sign_ok

        match_fraction = matches.mean(axis=1)
        line_flags = match_fraction >= min_cols_frac

        flagged_idx = np.where(line_flags)[0]
        if len(flagged_idx) == 0:
            block_results[group_name] = {}
            continue

        magnitude_matrix = np.abs(offsets)

        raw_blocks = []
        block_start = flagged_idx[0]
        prev = flagged_idx[0]
        for idx in flagged_idx[1:]:
            if idx - prev > max_gap:
                raw_blocks.append((block_start, prev))
                block_start = idx
            prev = idx
        raw_blocks.append((block_start, prev))

        # look for where the DN suddenly changes across all cols (within the
        # cols themselves, not relative to neighbors)
        delta = np.abs(
            magnitude_matrix[end_window:, :] - magnitude_matrix[:-end_window:])
        cols_jumped_frac = np.mean(delta >= end_threshold, axis=1)
        cols_jumped_frac = np.concatenate(
            [np.zeros(end_window), cols_jumped_frac])
        split_points = set(np.where(cols_jumped_frac >= min_cols_frac)[0])

        final_ranges = []
        for (s, e) in raw_blocks:
            seg_start = s
            for line in range(s + 1, e + 1):
                if line in split_points:
                    final_ranges.append((seg_start, line - 1))
                    seg_start = line
            final_ranges.append((seg_start, e))

        final_ranges = [(s, e) for s, e in final_ranges if
                        (e - s + 1) >= min_block_size]

        group_result = {}
        for i, (s, e) in enumerate(final_ranges):
            seg = offsets[s:e + 1, :]

            mean_per_col = np.nanmean(seg, axis=0)
            median_per_col = np.nanmedian(seg, axis=0)

            # get offset_types per column
            offset_per_col = {}
            for j, idx in enumerate(indices):
                col_vals = seg[:, j]
                col_vals = col_vals[~np.isnan(col_vals)]
                n_drop = np.sum(col_vals > 0)
                n_bump = np.sum(col_vals < 0)
                if len(col_vals) == 0:
                    offset_per_col[idx] = "unknown"
                elif n_drop > 0 and n_bump > 0:
                    offset_per_col[idx] = "mixed"
                elif n_drop > 0:
                    offset_per_col[idx] = "drop"
                else:
                    offset_per_col[idx] = "bump"

            # determine offset_type for the whole block
            # may only need this, not to determine it per col. eventually.
            # although for now it's useful.
            flat_vals = seg.flatten()
            flat_vals = flat_vals[~np.isnan(flat_vals)]
            n_drop_block = np.sum(flat_vals > 0)
            n_bump_block = np.sum(flat_vals < 0)
            if len(flat_vals) == 0:
                offset_block = "unknown"
            elif n_drop_block > 0 and n_bump_block > 0:
                offset_block = "mixed"
            elif n_drop_block > 0:
                offset_block = "drop"
            else:
                offset_block = "bump"

            group_result[i] = {
                "start_line": s,
                "end_line": e,
                "n_lines": e - s + 1,
                "mean_offset_per_col": {idx: mean_per_col[j] for
                                        j, idx in enumerate(indices)},
                "median_offset_per_col": {idx: median_per_col[j]
                                          for j, idx in
                                          enumerate(indices)},
                "offset_type_per_col": offset_per_col,
                "mean_offset_block": np.nanmean(seg),
                "median_offset_block": np.nanmedian(seg),
                "offset_type_block": offset_block,
            }

        block_results[group_name] = group_result

    return block_results


def apply_variable_column_correction(
        obs_image: np.ndarray,
        block_results: dict,
        stat: Literal["median", "mean"] = "median",
        method: Literal["column", "block"] = "column",
        skip_mixed_type: bool = False,
) -> np.ndarray:
    """
    Apply the offsets found in find_variable_column_blocks to each bad col
    block. Same offset is applied to all bands of each col, optional to
    apply a different offset per col within a block or the same offset to all
    cols within a block. obs_image edited in place.

    Args:
        obs_image: Input image, in bands x lines x cols. Could be rdn / rfl or
            DN but need to change settings for anything not DN.
        block_results: Dict of block info.
        stat: Correct the columns using either the mean or median value.
        method: "column" to correct using column-level stats within the block,
            or using block level stats, "block".
        skip_mixed_type: if True, columns/blocks with offset_type_per_col
            "mixed" or "unknown" are not corrected.
    """

    # correction offset amount stored in dict per block / col group
    stat_name = f"{stat}_offset_per_col" if method == "column" \
        else f"{stat}_offset_block "

    for group_name, blocks in block_results.items():

        if not blocks:
            print('No bad column blocks identified.')
            continue

        # run fix per block of ID'd bad lines, using one offset value per block
        # or one offset value per column per block
        for block_idx, block in blocks.items():
            s = block["start_line"]
            e = block["end_line"] + 1
            types_per_col = block["offset_type_per_col"]

            offset_vals = block[stat_name]

            if method == "block":
                if np.isnan(offset_vals):
                    continue
                if skip_mixed_type and block["offset_type_block"] in (
                        "mixed", "unknown"):
                    # skipping whole block
                    continue
                for col_idx in types_per_col.keys():
                    c = col_idx - 1
                    obs_image[:, s:e, c] += offset_vals

            elif method == "column":
                for col_idx, offset in offset_vals.items():
                    if skip_mixed_type and types_per_col.get(col_idx) in (
                            "mixed", "unknown"):
                        # skipping col
                        continue
                    if np.isnan(offset):
                        continue
                    c = col_idx - 1
                    obs_image[:, s:e, c] += offset

    return obs_image


def fix_variable_columns(obs_image: np.ndarray, col_groups: dict):
    """
    Wrapper for finding blocks of lines where multiple columns are
    simultaneously higher or lower in DN across bands (caused by some kind
    of background electronic effect).

    Identify blocks of lines where pre-identified column groups are bad and
    applies an offset.

    Args:
        obs_image: Input image, in bands x lines x cols. Could be rdn / rfl or
            DN but need to change settings for anything not DN.
        col_groups: Dictionary with lists of bad columns.
    """

    block_results = find_variable_column_blocks(
        np.mean(obs_image[:15, :, :], axis=0),
        col_groups
    )

    if any(block_results.values()):
        # only run fix if there's something to fix
        obs_image = apply_variable_column_correction(
            obs_image,
            block_results,
            stat='median',
            method='column',
        )
    else:
        print('No bad column blocks identified.')

        # send image back in detector format
    return obs_image.transpose(1, 0, 2)
