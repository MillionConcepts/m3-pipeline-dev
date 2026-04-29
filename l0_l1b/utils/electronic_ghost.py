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
    So they subtracted the fractional ghost amount from every pixel using all
    corresponding panel location pixels.

    In the list of corrections in the DPSIS this comes after interpolation
    of 'bad' elements, so right now the ghosts of those are not
    """
    if dark_cols is None:
        dark_cols = []

    panel_width = l0_samples // 4  # 80 for 320 etc

    obs_data = obs_data.astype(np.float32)
    frame = obs_data.copy()

    # don't use dark vals for correction bc they aren't dark subtracted
    # this is a bekah design decision we could change
    frame[:, :, dark_cols] = 0

    # sum the 4 panels
    frame_panels = frame.reshape(obs_data.shape[0], obs_data.shape[1], 4,
                                 panel_width)
    ghost_signal = ghost_correction * frame_panels.sum(axis=2)

    # reshape obs data for subtraction
    obs_data = obs_data.reshape(obs_data.shape[0], obs_data.shape[1], 4,
                                panel_width)

    obs_data -= ghost_signal[:, :, np.newaxis, :]

    return obs_data.reshape(obs_data.shape[0], obs_data.shape[1], -1)
