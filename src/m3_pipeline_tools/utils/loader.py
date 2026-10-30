""" Loading functions for observational data and some calibration files."""

from pathlib import Path
import numpy as np


def load_fits_into_frame(obs_path: Path) -> np.ndarray:
    """
    Load the obs data and return it in detector POV / frame view where
    axis 0 is frames / lines.
    """
    from astropy.io import fits

    obs_path = Path(obs_path)

    if not obs_path.is_file():
        raise FileNotFoundError(f"File not found: {obs_path}")
    try:
        with fits.open(obs_path, memmap=False) as hdul:
            image = hdul[0].data
    except OSError as e:
        raise OSError(f"Could not read FITS file {obs_path}, {e}")
    if image is None:
        raise ValueError(f"No data in primary HDU of {obs_path}")

    filename = obs_path.name.lower()

    if "ff" in filename or "flat" in filename:
        return image.astype(np.float32)

    if "bde" in filename or "flag" in filename:
        return image.astype(np.uint8)

    if len(image.shape) > 2:
        # only transpose if it's an observation, not cal file
        # we want lines as axis 0
        return image.transpose(1, 0, 2).astype(np.float32)
    else:
        # just in case a flag map or flat isn't indicated as that in the name
        return image.astype(np.float32)


def load_bde_map(flag_path: Path,
                 extra_flag_path: Path = None,
                 ) -> np.ndarray:
    """
    Load the flag map as an int array, optionally adding a secondary flag map
    of the same shape.

    Some flashing (likely ROIC noise) pixels are not flagged in warmer maps or
    are flagged on and off throughout the mission. Because these are frequently
    bad in the observations, we can just add them to every flag map as a
    precaution.
    """
    from ..reference import check_shape

    try:
        bde_map = load_fits_into_frame(flag_path)
    except Exception as e:
        bde_map = None
        print(f"Error loading observation-level BDE map: {e}.")
    if extra_flag_path is not None:
        print(f"Adding standard flag map for flagging.")
        extra_bde_map = load_fits_into_frame(extra_flag_path)
        # the values in the map itself are counts of bde maps with that flag,
        # so we just set everything to 1 where flagged
        extra_bde_map = (extra_bde_map > 0).astype(np.uint8)
        if bde_map is None:
            # use standard flag map only
            return extra_bde_map
        check_shape(bde_map, extra_bde_map, "extra flag map loading")
        bde_map = bde_map + extra_bde_map
    return bde_map
