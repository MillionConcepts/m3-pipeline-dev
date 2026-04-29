import numpy as np
from astropy.io import fits
from pathlib import Path
from typing import Literal


def make_dark_signal_image(
        dark_path: Path,
        dark_cols: list = None,
        dark_method: Literal['mean', 'median', 'std', 'max'] = 'median'
) -> np.ndarray:
    """
    The dark signal of an observation is estimated from a dark signal
    observation acquired prior to the real observation during a non-illuminated
    portion of the orbit. They were also used to generate the bad detector
    element image (BDE), but we are not currently doing that.

    Args:
        dark_path: Path to the dark signal obs.
        dark_cols: Columns used for estimating dark signal during an
        observation can be optionally set to 0 to preserve the obs dark signal
        after subtraction (in the DSS image).
        dark_method: Statistic to calculate (mean, med, std, max). Mean
        and med are both good for dark signal subtraction, std and max for
        detecting bad detector elements.
    """
    from .loader import load_fits_into_frame

    dark_obs_data = load_fits_into_frame(dark_path)
    # check everything looks normal (it should)
    # row = frames = detector pov etc
    bands, lines, cols = dark_obs_data.shape

    if lines <= 4:
        print("This dark signal obs is probably too short to be useful.")

    # exclude first and last two frames bc they can be funky. this number
    # could increase tbh but haven't done extensive investigation
    exc = 2

    if dark_method.lower() == 'mean':
        dark_signal = dark_obs_data[exc:-exc, :, :].mean(axis=0)
    elif dark_method.lower() == 'median':
        dark_signal = np.median(dark_obs_data[exc:-exc, :, :], axis=0)
    elif dark_method.lower() == 'std':
        dark_signal = np.std(dark_obs_data[exc:-exc, :, :], axis=0)
    elif dark_method.lower() == 'max':
        dark_signal = np.max(dark_obs_data[exc:-exc, :, :], axis=0)
    else:
        # IDK why we would use this yet
        return dark_obs_data

    if dark_cols is not None:
        # set avg for dark cols to 0 to preserve observation dark signal
        # values for later steps.
        dark_signal[:, dark_cols] = 0

    return dark_signal


def basic_dark_pedestal_correction(
        obs_image: np.ndarray,
        dark_cols: list = None,
) -> np.ndarray:
    """
    Simple dark pedestal estimation. In the HVM3 code they use median of
    dark cols per frame and subtract that from the same frame.

    For this to work, the dark cols must be dark subtracted in the DSS image.
    Otherwise, they will be huge numbers that get subtracted.

    If the median offset is negative, then it's additive to the image.

    This is adding a single scalar value to the whole image, so loses the
    'pattern' of the dark.
    """
    if dark_cols is None:
        # don't do this
        return obs_image

    pedestal = np.median(obs_image[:, :, dark_cols], axis=0)

    obs_image = obs_image - pedestal[np.newaxis, :, :]

    return obs_image


def experimental_dark_pedestal_correction(
        obs_image: np.ndarray,
        dark_signal: np.ndarray,
        dark_cols: list = None,
) -> np.ndarray:
    """
    Compute the ratio between the dark signal image and the dark cols of the
    observation DSS image per channel per frame. Multiply the dark signal
    channel by that ratio and add to the corresponding channel of the image.
    The idea is to preserve the 'pattern' of the dark signal image instead of
    applying a single scalar offset to a whole frame.

    For this to work, the dark cols can't be dark subtracted in the DSS image.
    Otherwise, they will be small negative numbers.

    Args:
        obs_image: DSS obs image array.
        dark_cols: Columns used for estimating dark signal.
        dark_signal: Median or mean dark signal observation.
    """
    if dark_cols is None:
        # don't do this
        return obs_image

    for frame in range(obs_image.shape[0]):
        for channel in range(obs_image.shape[1]):
            ratio = np.median(obs_image[frame, channel, dark_cols]) / \
                    np.median(dark_signal[channel, dark_cols])

            offset = (1 - ratio) * dark_signal[channel, :]

            obs_image[frame, channel, :] += offset

    return obs_image
