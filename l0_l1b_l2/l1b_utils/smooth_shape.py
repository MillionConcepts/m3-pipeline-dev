from pathlib import Path
import numpy as np


def load_ssc_factors(ssc_path: Path):
    """
    Read in smooth shape correction factors, which are per obs based on obs
    temp. These were calculated by the mission.

    Deprecated because now we have a table of all SSC factors.
    """
    import pandas as pd

    # could combine these tables to one parquet file maybe (they're really
    # small)
    ssc_table = pd.read_fwf(ssc_path,
                            names=["channel", "corr_factor"],
                            dtype={'corr_factor': np.float32},
                            )
    return ssc_table['corr_factor'].values


def load_ssc_factors_from_table(obs_id: str, ssc_table_path: Path):
    """
    Load smooth shape correction factors from table of correction factors.
    A global and a target mode table of SSC factors were made by pulling all
    SSC files from the PDS.

    Will return 85 gain factors for global mode or 256 for target mode.
    """
    import pandas as pd

    df = pd.read_csv(ssc_table_path, dtype={0: str})

    obs_row = df[df.iloc[:, 0] == str(obs_id)]
    if obs_row.empty:
        raise KeyError(f"SSC corr factors for {obs_id} not found in"
                       f" {ssc_table_path}.")

    # get every col value after the obs id
    ssc_factors = obs_row.iloc[0, 1:].to_numpy(dtype=np.float32)
    return ssc_factors


def apply_ssc_factors(obs_image: np.ndarray, obs_id: str, ssc_path: Path):
    """
    Apply smooth shape correction gain factors developed by the mission team.
    These are temperature-specific curves.
    """
    from l0_l1b_l2.reference import check_shape

    ssc_factors = load_ssc_factors_from_table(obs_id, ssc_path)
    check_shape(
        obs_image.shape,
        (len(ssc_factors), obs_image.shape[-1]),
        'smooth shape'
    )
    return obs_image * ssc_factors[np.newaxis, :, np.newaxis]
