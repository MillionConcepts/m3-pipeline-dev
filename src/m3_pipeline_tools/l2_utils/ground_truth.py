import numpy as np
from pathlib import Path


def ground_truth_correction(image: np.ndarray, ground_truth_path: Path):
    """
    Ground truth correction was not applied to L2 data in the PDS but they
    provided the cal files.

    L2s5(λ) = L2s4(λ) * gGT(λ) + oGT(λ)

    Here, oGT(λ) is 0.
    """
    import pandas as pd

    ground_truth = pd.read_fwf(ground_truth_path,
                               names=['channel', 'wavelength',
                                      'corr_factor', 'corr_offset'])

    image *= ground_truth['corr_factor'].values[np.newaxis, :, np.newaxis]

    # this step is only adding 0s at the moment
    # image += ground_truth['corr_offset'].values[np.newaxis, :, np.newaxis]

    return image
