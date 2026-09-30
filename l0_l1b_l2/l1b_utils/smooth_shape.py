from pathlib import Path
import numpy as np


def load_ssc_factors(ssc_path: Path):
    """
    Read in smooth shape correction factors, which are per obs based on obs
    temp. These were calculated by the mission.
    """
    import pandas as pd

    # could combine these tables to one parquet file maybe (they're really
    # small)
    ssc_table = pd.read_fwf(ssc_path,
                            names=["channel", "corr_factor"],
                            dtype={'corr_factor': np.float32},
                            )
    return ssc_table['corr_factor'].values


def apply_ssc_factors(obs_image: np.ndarray, ssc_path: Path):
    """
    Apply smooth shape correction gain factors developed by the mission team.
    These are temperature-specific curves.
    """
    from l0_l1b_l2.reference import check_shape

    ssc_factors = load_ssc_factors(ssc_path)
    check_shape(
        obs_image.shape,
        (len(ssc_factors), obs_image.shape[-1]),
        'smooth shape'
    )
    return obs_image * ssc_factors[np.newaxis, :, np.newaxis]
