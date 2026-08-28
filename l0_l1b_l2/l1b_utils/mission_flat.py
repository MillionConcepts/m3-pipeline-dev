from pathlib import Path
import numpy as np
from typing import Optional


def load_flats(flat_path: Path):
    """
    Read in flat files.
    """
    from astropy.io import fits

    with fits.open(flat_path) as hdul:
        flat = hdul[0].data

    return flat


def apply_flat(
        obs_data: np.ndarray,
        flat_path: Path,
        flag_path: Optional[Path] = None
):
    """
    Load and multiple flat by image. Modify flat for flagged elements
    in the BDE if flag_path is given.
    """

    if flag_path is None:
        flat = load_flats(flat_path)
    else:
        flat = fix_flagged_in_lab_flat(
            flat_path,
            flag_path
        )
    if "lab" in str(flat_path):
        return obs_data * flat[:, np.newaxis, :]
    else:
        flat_masked = np.where(flat == 0, 1, flat)
        return obs_data / flat_masked[:, np.newaxis, :]


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
    from .mission_bde import bde_correction

    flat = load_flats(flat_path)

    mod_flat = bde_correction(flat, bde_path)

    return mod_flat
