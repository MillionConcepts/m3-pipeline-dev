from pathlib import Path


def load_fits_into_frame(obs_path: Path):
    """
    Load the obs data and return it in detector POV / frame view where
    axis 0 is frames / lines.
    """
    from astropy.io import fits
    import numpy as np

    with fits.open(obs_path) as hdul:
        image = hdul[0].data

    if "ff" in obs_path.name or "flat" in obs_path.name:
        return image.astype(np.float32)

    if "bde" in obs_path.name:
        return image.astype(np.uint8)

    # TODO: are the images ever flipped relative to the detector frame in
    #  the spectral direction?
    return image.transpose(1, 0, 2).astype(np.float32)
