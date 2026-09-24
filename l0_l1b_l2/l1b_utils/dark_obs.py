import numpy as np
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


    # TODO: add variable column group flagging as we do for the whole obs &
    #   apply offset. may need to subtract min or median from dark or use std
    #   (but then wouldn't flag if whole dark obs is with cols on).

    # exclude first and last two frames bc they can be funky. this number
    # could increase tbh but haven't done extensive investigation
    exc = 1

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


# Ideas at the moment:
# Currently I think our best bet is to treat the dark pedestal effect as some
# percentage of the illuminated line / band median value. So this will vary
# appropriately with the terrain and band. Something between 3-5% seems
# appropriate based on looking at pedestal offsets divided by median per line.
# Possibly it would be a good idea to scale a dark signal image by 3-5% and
# add that back? I need to determine if we should use the last “cold” dark
# signal vs the closest in temp.
# Last cold dark signal would probably be around 147 K. Even in the cooler
# darks, they range by 10-30 DN across the array. So 3% of that would
# Could try testing mean neg / neg count divided by the 4 sectors of the array
# (read out columns etc), aka seeing if the 4 bright / dark areas in the dark
# are reflected in dark pedestal
# This method reduces the increase in noise in the few dark signal column
# pixels we have per band. They get very noisy / messy at high temperatures,
# and we don’t want to propagate that noise into the image.
# Alternatively, we could try to fit a smoother curve across the dark pedestal
# data per dark column set per band / line. However, at warmer temperatures
# the data gets so messy and jagged I don’t know if this is a good idea.

def basic_dark_pedestal_correction(
        obs_image: np.ndarray,
        dark_cols: list = None,
) -> np.ndarray:
    """
    Simple dark pedestal estimation. In the HVM3 code they use median of
    dark cols per frame and subtract that from the same frame. In the DPSIS
    they say they use illumination levels to determine the pedestal offset--
    we're not doing that here. Maybe we should in the future!

    For this to work, the dark cols must be dark subtracted in the DSS image.
    Otherwise, they will be huge numbers that get subtracted.

    We apply on a line by line (aka frame by frame) basis because that's what
    the DPSIS says. This is also what the HVM3 code does. I don't really
    understand this on a theoretical basis because brighter illuminated bands
    have a stronger dark pedestal effect while less illuminated areas have a
    lower dark pedestal effect.

    Also, the columns on the side of the detector seem to grow in value at a
    faster rate than the median of the columns in the middle (especially on
    the left side).

    Args:
        obs_image: Obs image data, dark subtracted.
        dark_cols: Columns used for estimating dark signal during an
            observation (they receive no light).
    """
    if dark_cols is None:
        # don't do this
        return obs_image
    # pedestal for each frame
    # TODO: add setting to switch between per frame and band vs one value per
    #    frame, they both work approx equally wrong vs og l1b
    pedestals = np.nanmedian(obs_image[:, :, dark_cols], axis=(2, 1))
    obs_image = obs_image - pedestals[:, np.newaxis, np.newaxis]

    return obs_image


def illumination_based_dark_pedestal_correction(
        obs_image: np.ndarray,
        left_cutoff_col: int,
        right_cutoff_col: int,
) -> np.ndarray:
    """
    Using the dark pedestal columns on the left and right side of the array
    to determine the dark pedestal effect is a messy science because
    1) they increase in noisiness as the detector warms up
    2) they increase at a different rate than other parts of the detector
    3) they could be a bad pixel from the starts
    4) the dark signal obs used could be too different in temperature

    By comparing the dark pedestal effect per band against the median
    illuminated signal per band per line, we see that the dark pedestal effect
    is a 3-5% effect at most bands (things get messy below channel ~20 in
    global, where the SNR is not as good).

    So here we take the median signal of a line per band and add 5% back
    to the image. In the future we could consider calculating the dark pedestal
    ratio per observation.

    Args:
        obs_image: Obs image data, dark subtracted.
        left_cutoff_col: Left side of illuminated area.
        right_cutoff_col: Right side of illuminated area.
    """
    # could squeeze in further to avoid weird illuminated edges at
    # low/high bands
    pedestals = np.nanmedian(
        obs_image[:, :, left_cutoff_col:right_cutoff_col],
        axis=2) * 0.03

    # abs value the pedestal for the infrequent scenario that it is a very dark
    # section of an observation that has been over dark signal subtracted
    obs_image = obs_image + np.abs(pedestals[:, :, np.newaxis])

    return obs_image


def experimental_dark_pedestal_correction(
        obs_image: np.ndarray,
        dark_path: Path,
        bde_path: Path,
        dark_cols: list = None,
        tap_cols: list = None,
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
        obs_image: DSS obs image array with dark cols not dark subtracted.
        dark_path: We reload dark signal obs without calculating dark signal
        image for DSS.
        dark_cols: Columns used for estimating dark signal.
    """
    from .loader import load_fits_into_frame
    from .mission_bde import bde_correction, \
        detector_array_tap_interpolation
    if dark_cols is None:
        # don't do this
        return obs_image

    # reload dark without setting dark cols to 0 & compress into one frame
    # (median of all but first 2 and last 2 frames)
    dark_signal = load_fits_into_frame(dark_path)
    exc = 2
    dark_signal = np.nanmedian(dark_signal[exc:-exc, :, :], axis=0)

    dark_signal = detector_array_tap_interpolation(dark_signal, tap_cols)

    dark_signal = bde_correction(dark_signal, bde_path)

    for frame in range(obs_image.shape[0]):
        for channel in range(obs_image.shape[1]):
            ratio = np.nanmedian(obs_image[frame, channel, dark_cols]) / \
                np.nanmedian(dark_signal[channel, dark_cols])

            if np.isfinite(ratio):
                offset = (1 - ratio) * dark_signal[channel, :]
            else:
                offset = dark_signal[channel, :]
            obs_image[frame, channel, :] += offset

    return obs_image
