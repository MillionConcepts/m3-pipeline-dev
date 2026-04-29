from pathlib import Path


def load_fits_into_frame(obs_path: Path):
    """
    Load the obs data and return it in detector POV / frame view where
    axis 0 is frames / lines.
    """
    from astropy.io import fits

    with fits.open(obs_path) as hdul:
        image = hdul[0].data

    return image.transpose(1, 0, 2)
