from pathlib import Path


def load_ssc_factors(ssc_path: Path):
    """
    Read in smooth shape correction factors, which are per obs based on obs
    temp. These were calculated by the mission.
    """
    import pandas as pd

    # could combine these tables to one parquet file maybe (they're really
    # small)
    ssc_table = pd.read_fwf(ssc_path, names=["channel", "corr_factor"])

    return ssc_table['corr_factor'].values
