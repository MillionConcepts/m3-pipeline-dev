# M3 Calibration Pipeline for L0 to L1B processing

# import os
# import sys
# from astropy.io import fits
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Optional, Sequence, Mapping, Literal

from l0_l1b.reference import PipeManager, check_observation


def run_half_pipe(
    obs_id: str,
    local_root: str = "data",
    save_steps: bool = False,
    verbose: bool = True,
):
    """
       Args:
           obs_id: Observation ID for the M3 observation, a string beginning
           with M3.
           local_root: Where is the data stored?
           save_steps: Save intermediate step data as fits files.
           #TODO: Could make more complicated to save only specific steps.
           verbose: Give me all the info or don't.
    """

    obs_warn, obs_error, metadata = check_observation(obs_id, local_root)
    if verbose and len(obs_warn) > 0:
        print("\n".join(obs_warn))
    if len(obs_error) > 0:
        print("\n".join(obs_error))
        print("Bailing out.")
        return f"return code: {';'.join(obs_error)}"

    moonager = PipeManager(
        obs_id=obs_id,
        metadata=metadata,
        local_root=local_root,
        save_steps=save_steps,
    )

    return
