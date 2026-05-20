import numpy as np


def electronic_panel_ghost_correction(
        obs_data: np.ndarray,
        l0_samples: int,
        ghost_correction: float,
        dark_cols: str = None,
):
    """
    During readout from the 6604a detector array, small 'ghosts' occur in the
    other three panels in the same row as the original signal. The signal of
    each ghost is .0048 a fraction of the original signal.

    From CRISM DPSIS (640 wide det):
       DN'(x,y) =  DN(x,y)
                   - GHOST_DN(x,y)
                   - GHOST_DN(x+160, y)
                   - GHOST_DN(x+320, y)
                   - GHOST_DN(x+480, y)
    So they subtracted the fractional ghost amount from every pixel using a
    all corresponding panel location pixels.

    But for M3 the ghosts are actually depressions in signal, so we should
    ADD the ghost offset. And I don't want to include the panel itself.
    So more like:
       DN'(x,y) =  DN(x,y)
                   + GHOST_DN(x+160, y)
                   + GHOST_DN(x+320, y)
                   + GHOST_DN(x+480, y)
    In the list of corrections in the DPSIS this comes after interpolation
    of 'bad' elements, so right now the ghosts of those are not accounted
    for. It is not even clear to me if they occur?
    """
    if dark_cols is None:
        dark_cols = []

    panel_width = l0_samples // 4  # 80 for 320 etc

    frame = obs_data.copy()

    # don't use dark vals if we pass dark cols
    # for correction bc they aren't dark subtracted
    # this is a bekah design decision we could change
    frame[:, :, dark_cols] = 0

    # split by 4 panels 80 or 160 wide
    frame_panels = frame.reshape(
        obs_data.shape[0],
        obs_data.shape[1],
        4,
        panel_width
    )
    obs_data_reshaped = obs_data.reshape(
        obs_data.shape[0], obs_data.shape[1],
        4,
        panel_width
    )
    # sum other 3 panels and multiply by correction factor
    for panel_idx in range(4):
        other_panels = np.concatenate(
            [frame_panels[:, :, :panel_idx, :],
             frame_panels[:, :, panel_idx+1:, :]],
            axis=2
        )
        ghost_signal = ghost_correction * other_panels.sum(axis=2)
        obs_data_reshaped[:, :, panel_idx, :] += ghost_signal

    return obs_data_reshaped.reshape(
        obs_data_reshaped.shape[0],
        obs_data_reshaped.shape[1], -1
    )
