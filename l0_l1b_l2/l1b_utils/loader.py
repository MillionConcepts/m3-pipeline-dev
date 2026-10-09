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

    if len(image.shape) > 2:
        # only transpose if it's an observation, not cal file
        # we want lines as axis 0
        return image.transpose(1, 0, 2).astype(np.float32)
    else:
        # just in case a flag map or flat isn't indicated as that in the name
        return image.astype(np.float32)


def load_bde_map(flag_path: Path,
                 additional_flag_path: Path = None
                 ) -> np.ndarray:
    """
    Load the flag map as an int array, optionally adding a secondary flag map
    of the same shape.

    Some flashing (likely ROIC noise) pixels are not flagged in warmer maps or
    are flagged on and off throughout the mission. Because these are frequently
    bad in the observations, we can just add them to every flag map as a
    precaution.
    """
    from l0_l1b_l2.reference import check_shape

    bde_map = load_fits_into_frame(flag_path)

    if additional_flag_path is not None:
        #TODO: make an extra cold switching flag bde map for target mode
        extra_bde_map = load_fits_into_frame(additional_flag_path)
        # the values in the map itself are counts of bde maps with that flag,
        # so we just set everything to 1 where flagged
        extra_bde_map = (extra_bde_map > 0).astype(np.uint8)
        check_shape(bde_map, extra_bde_map, "extra flag map loading")
        bde_map = bde_map + extra_bde_map

    return bde_map
