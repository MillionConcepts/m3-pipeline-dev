import numpy as np
from pathlib import Path


def detector_array_tap_interpolation(obs_data: np.ndarray, cols: list):
    """
    Columns related to the read-out are linearly interpolated in the spatial
    direction (columns on either side). The columns are 161, 321, 481 for
    target and 81, 161, 241 for global mode.
    """
    cols = np.array(cols)
    if obs_data.ndim == 3:
        obs_data[:, :, cols] = (obs_data[:, :, cols - 1] +
                                obs_data[:, :, cols + 1]) / 2.0
    else:
        obs_data[:, cols] = (obs_data[:, cols - 1] +
                             obs_data[:, cols + 1]) / 2.0
    return obs_data


def filter_seam_interpolation(obs_data: np.ndarray, channels: list):
    """
    Order sorting filter seams show up on the detector, we linearly interpolate
    across them in the spectral direction (these are horizontal features). For
    target mode they are channels 41, 42, and 116 and for global mode they are
    13 and 50.
    """
    for channel in channels:
        # we have special cases for 41 & 42 because they are next to each other
        # this could be less specific if we anticipate expanding the number of
        # channels flagged for this, but I don't think we will. We also could
        # get rid of the check if the pipeline is never used for target mode
        # obs.
        if channel == 41:
            obs_data[:, 41, :] = (2 / 3) * obs_data[:, 40, :] + \
                                 (1 / 3) * obs_data[:, 43, :]
        elif channel == 42:
            obs_data[:, 42, :] = (1 / 3) * obs_data[:, 40, :] + \
                                 (2 / 3) * obs_data[:, 43, :]
        else:
            obs_data[:, channel, :] = (obs_data[:, channel - 1, :] +
                                       obs_data[:, channel + 1, :]) / 2.0
    return obs_data


def bad_detector_element_correction(obs_data: np.ndarray, bde_path: Path):
    """
    Elements are flagged 0-5 in these fits files. Values 1-4 seem to indicate
    things that are "flagged", 1 being the lowest level and 4 the highest. I
    think the mission interpolated everything flagged 1-4 in the spectral
    direction. This does seem a little problematic to me because the filter
    seams (channel features) are flagged, usually as 4, so right now we
    interpolate across them twice.

    Pixels with only one "good" pixel above or below it (this means they are
    likely near the edge) will adopt that good pixel's value. If there are NO
    good pixels in a column, right now we don't do anything. This could change,
    but should only matter for the tap columns we interpolate across in the
    spatial direction in a later step.

    Supports 3D arrays (frames, channels, cols) and 2D arrays (channels, cols).
    For 3D arrays, correction is applied per-frame using that frame's values.
    """
    from .loader import load_fits_into_frame

    bde_map = load_fits_into_frame(bde_path)

    bad_mask = bde_map != 0
    n_rows, n_cols = bad_mask.shape  # 86 x 320 for global mode

    # we only need to figure out the interpolation weights once for the obs
    # new_val =
    # val_at_top + (bad_row-top_row)/(bot_row-top_row)*(val_at_bot-val_at_top)
    # where the "weight" is (bad_row-top_row)/(bot_row-top_row)

    bad_rows, bad_cols = np.where(bad_mask)  # all bad pixels

    # find the closest top and bottom pixel
    top_rows = np.full(len(bad_rows), -1,
                       dtype=np.intp)  # np.intp is for indices
    bottom_rows = np.full(len(bad_rows), n_rows, dtype=np.intp)

    # "good" pixels per column
    good_in_col = []
    for c in range(n_cols):
        # get all good rows
        good_in_col.append(np.where(~bad_mask[:, c])[0])

    # find closest good pixels
    for i, (r, c) in enumerate(zip(bad_rows, bad_cols)):
        g = good_in_col[c]
        if g.size == 0:
            continue
            # TODO: IDK what to do if the whole column is bad?
            #  for readout cols we interpolate across the col (spatially)
            #  instead of across channels (spectrally)
        idx = np.searchsorted(g, r)
        if idx > 0:
            top_rows[i] = g[idx - 1]  # above
        if idx < len(g):
            bottom_rows[i] = g[idx]  # below

    has_top = top_rows >= 0
    has_bottom = bottom_rows < n_rows
    both = has_top & has_bottom
    top_only = has_top & ~has_bottom
    bottom_only = ~has_top & has_bottom
    # neither = ~has_top & ~has_bottom
    # we ignore this right now,
    # could interpolate across / spatially

    # weights for linear interpolation when there is a top and bottom pixel
    denominator = np.where(both, bottom_rows - top_rows, 1)
    t = np.where(both, (bad_rows - top_rows) / denominator, 0.0)

    is_3d = obs_data.ndim == 3

    if both.any():
        br = bad_rows[both]
        bc = bad_cols[both]
        tr = top_rows[both]
        btr = bottom_rows[both]
        w = t[both]

        if is_3d:
            n_frames = obs_data.shape[0]
            for frame_idx in range(n_frames):
                top_vals = obs_data[frame_idx, tr, bc]
                bot_vals = obs_data[frame_idx, btr, bc]
                interp = top_vals + w * (bot_vals - top_vals)
                obs_data[frame_idx, br, bc] = interp
        else:
            top_vals = obs_data[tr, bc]
            bot_vals = obs_data[btr, bc]
            interp = top_vals + w * (bot_vals - top_vals)
            obs_data[br, bc] = interp
    if top_only.any():
        br = bad_rows[top_only]
        bc = bad_cols[top_only]
        tr = top_rows[top_only]
        if is_3d:
            n_frames = obs_data.shape[0]
            for frame_idx in range(n_frames):
                obs_data[frame_idx, br, bc] = obs_data[frame_idx, tr, bc]
        else:
            obs_data[br, bc] = obs_data[tr, bc]

    if bottom_only.any():
        br = bad_rows[bottom_only]
        bc = bad_cols[bottom_only]
        btr = bottom_rows[bottom_only]
        if is_3d:
            n_frames = obs_data.shape[0]
            for frame_idx in range(n_frames):
                obs_data[frame_idx, br, bc] = obs_data[frame_idx, btr, bc]
        else:
            obs_data[br, bc] = obs_data[btr, bc]

    return obs_data
