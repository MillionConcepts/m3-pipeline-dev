from pathlib import Path
import numpy as np


def load_flats(flat_path: Path):
    """
    Read in flat files.
    """
    from astropy.io import fits

    with fits.open(flat_path) as hdul:
        flat = hdul[0].data

    return flat


def fix_flagged_in_lab_flat(flat_path: Path, bde_path: Path):
    """
    Use mission interpolation method for BDE (flag) pixels on the DSS image to
    apply the same interpolation to the lab flat. The idea is that the lab
    flat doesn't work for pixels that are interpolated (they weren't really
    'measured' at that pixel).

    Using this does mean the mission observation-derived flats probably
    no longer work because those used the original lab flat, which I don't
    think they modified in this way.
    """
    from astropy.io import fits
    from mission_bde import bad_detector_element_correction

    with fits.open(flat_path) as hdul:
        flat = hdul[0].data

    mod_flat = bad_detector_element_correction(flat, bde_path)

    return mod_flat
