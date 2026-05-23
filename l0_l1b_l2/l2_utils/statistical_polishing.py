import numpy as np
from pathlib import Path


def apply_stat_polishing(image: np.ndarray, stat_pol_path: Path):
    """
    Multiply by statistical polishing coefficients. These are temperature
    dependent, there is a "warm" and "cold" version determined by the date.
    There used to be an additional statistical polishing offset but for the
    latest delivery I believe it was all 0.

    L2s2(λ) = L2s1(λ) * gSP(λ) + oSP(λ))
    """
    import pandas as pd

    statpol = pd.read_fwf(stat_pol_path, names=['channel', 'wavelength',
                                                'mult_factor', 'offset'])

    image *= statpol['mult_factor'].values[np.newaxis, :, np.newaxis]

    # this step is only adding 0s at the moment
    # image += statpol['offset'].values[np.newaxis, :, np.newaxis]

    return image
