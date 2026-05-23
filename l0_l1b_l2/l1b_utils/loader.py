from pathlib import Path
import numpy as np


def load_fits_into_frame(obs_path: Path) -> np.ndarray:
    """
    Load the obs data and return it in detector POV / frame view where
    axis 0 is frames / lines.
    """
    from astropy.io import fits

    filename = obs_path.name.lower()

    with fits.open(obs_path) as hdul:
        image = hdul[0].data

    if "ff" in filename or "flat" in filename:
        return image.astype(np.float32)

    if "bde" in filename:
        return image.astype(np.uint8)

    return image.transpose(1, 0, 2).astype(np.float32)
