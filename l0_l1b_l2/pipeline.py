
from typing import Literal

from l0_l1b_l2.reference import PipeManager, check_observation
from l0_l1b_l2.l1b_pipeline import run_l1b_new_pipeline, \
    run_l1b_mission_pipeline


def run_pipe(
    obs_id: str,
    pipe_version: Literal['mission', 'new'] = 'mission',
    local_root: str = "data",
    save_steps: bool = False,
    backplanes: bool = False,
    verbose: bool = True,
):
    """
       Args:
           obs_id: Observation ID for the M3 observation, a string beginning
           with M3.
           pipe_version: Do we want a mission-faithful processing of the L0
           data, or a more experimental processing?
           local_root: Where is the data stored?
           save_steps: Save intermediate step data as fits files. Eventually we
           could change this to designate specific steps to save.
           backplanes: Return backplanes (flag map, "error" from dark std)
           verbose: Give me all the info or don't.
    """

    obs_warn, obs_error, metadata = check_observation(obs_id)
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
        backplanes=backplanes,
        verbose=verbose,
    )

    if pipe_version == 'mission':
        return run_l1b_mission_pipeline(moonager)

    elif pipe_version == 'new':
        return run_l1b_new_pipeline(moonager)

