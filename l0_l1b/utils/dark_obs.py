import numpy as np
from astropy.io import fits
from pathlib import Path


def make_dark_signal_mean_image(dark_path: Path, dark_cols: list = None):
    """
    The dark signal of an observation is estimated from a dark signal
    observation acquired prior to the real observation during a non-illuminated
    portion of the orbit. They were also used to generate the bad detector
    element image (BDE), but we are not currently doing that.

    The dark signal is averaged for all "lines" to get an average dark signal
    value for each cross-track sample and spectral channel.

    dark_cols: for when we don't want to subtract avg dark signal from the cols
    used for dark pedestal corrections later on (pixels with purportedly no
    light landing on them during an obs)
    """

    # TODO: use median instead of mean? could write a separate function so we
    #  switch them out as desired or make it an option in this function

    with fits.open(dark_path) as hdul:
        dark_obs_data = hdul[0].data

    # check everything looks normal (it should)
    # row = frames = detector pov etc
    bands, lines, cols = dark_obs_data.shape

    if lines <= 4:
        print("This dark signal obs is probably too short to be useful.")

    # this is not listed in the DPSIS or Green, but I think it makes sense to
    # exclude the first two frames and the last two when computing the mean
    # dark signal bc sometimes they exhibit odd and anomalous characteristics.
    # could even do more tbh.
    dark_signal_avg = dark_obs_data[:, 2:-2, :].mean(axis=1)

    if dark_cols is not None:
        # set avg for dark cols to 0 to preserve dark signal values for later
        # steps.
        dark_signal_avg[:, dark_cols] = 0

    # main obs data is in int16
    return dark_signal_avg


def make_dark_signal_std_image(dark_path: Path):
    """
    Not a pipeline step at the moment. Just for QA. Could be used for new BDE.
    """

    with fits.open(dark_path) as hdul:
        dark_obs_data = hdul[0].data

    # check everything looks normal (it should)
    bands, lines, cols = dark_obs_data.shape

    if lines <= 10:
        print("This dark signal obs is probably too short to be useful.")

    # this is not listed in the DPSIS or Green, but I think it makes sense to
    # exclude the first two frames and the last two when computing the std
    # dark signal bc sometimes they exhibit odd and anomalous characteristics.
    dark_signal_std = np.std(dark_obs_data.transpose(1, 0, 2)[2:-2, :, :],
                             axis=0)

    return dark_signal_std
