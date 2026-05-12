# M3 Calibration Pipeline for L0 to L1B processing

from typing import Literal

from l0_l1b.reference import PipeManager, check_observation


def run_pipe(
    obs_id: str,
    pipe_version: Literal['mission', 'new'] = 'mission',
    local_root: str = "data",
    save_steps: bool = False,
    verbose: bool = True,
):
    """
       Args:
           obs_id: Observation ID for the M3 observation, a string beginning
           with M3.
           pipe_version: Do we want a mission-faithful processing of the L0
           data, or a more experimental processing?
           local_root: Where is the data stored?
           save_steps: Save intermediate step data as fits files.
           #TODO: Could make more complicated to save only specific steps.
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
        verbose=verbose,
    )

    if pipe_version == 'mission':
        return run_mission_pipeline(moonager)

    elif pipe_version == 'new':
        return run_new_pipeline(moonager)


def run_mission_pipeline(moonager: PipeManager):
    """
    Pipeline based on an originalist reading of the DPSIS.
    """
    import numpy as np
    from l0_l1b.utils.loader import load_fits_into_frame
    from l0_l1b.utils.dark_obs import make_dark_signal_image, \
        basic_dark_pedestal_correction
    from l0_l1b.utils.electronic_ghost import electronic_panel_ghost_correction
    from l0_l1b.utils.mission_bde import bad_detector_element_correction, \
        detector_array_tap_interpolation, filter_seam_interpolation
    from l0_l1b.utils.mission_flat import apply_flat
    from l0_l1b.utils.smooth_shape import load_ssc_factors
    from l0_l1b.utils.radiometric_calibration import load_rdn_cal_factors

    obs_image = load_fits_into_frame(moonager.obs_path)
    # obs_image shape = (frames / lines, channels / bands, samples / columns)

    if moonager.verbose:
        print("Subtracting dark signal.")
    # (1) Dark Signal Subtraction
    obs_image -= make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_method='mean'
    )

    # (2) Bad Detector Element Correction (Flag)
    if moonager.verbose:
        print("Running flagged pixel correction.")
    obs_image = bad_detector_element_correction(
        obs_data=obs_image,
        bde_path=moonager.flag_path,
    )

    # (3) Detector Tap Interpolation
    if moonager.verbose:
        print("Interpolating tap cols.")
    obs_image = detector_array_tap_interpolation(
        obs_data=obs_image,
        cols=moonager.read_out_cols
    )

    # (4) Filter Seam Interpolation
    if moonager.verbose:
        print("Interpolating filter seams.")
    obs_image = filter_seam_interpolation(
        obs_data=obs_image,
        channels=moonager.filter_seam_rows
    )

    # (5) Electronic Ghost Correction
    if moonager.verbose:
        print("Running electronic ghost correction.")
    obs_image = electronic_panel_ghost_correction(
        obs_data=obs_image,
        l0_samples=moonager.l0_samples,
        ghost_correction=moonager.ghost_correction,
    )

    # (6) Dark Pedestal Shift Correction
    if moonager.verbose:
        print("Running dark pedestal shift correction.")
    obs_image = basic_dark_pedestal_correction(
        obs_image=obs_image,
        dark_cols=moonager.dark_cols,
    )

    # (7) Scattered Light Correction
    # not implemented

    # (8) Lab Flat Correction
    if moonager.verbose:
        print("Applying lab flat.")
    obs_image = apply_flat(
        obs_data=obs_image,
        flat_path=moonager.lab_flat_path,
        flag_path=moonager.flag_path
    )

    # (9) Imaging-based Flat Correction
    if moonager.verbose:
        print("Applying observation-level flat.")
    obs_image = apply_flat(
        obs_data=obs_image,
        flat_path=moonager.obs_flat_path,
        flag_path=moonager.flag_path
    )

    # (10) Radiometric Calibration
    rdn_cal = load_rdn_cal_factors(moonager.rdn_cal_path)
    obs_image = obs_image * rdn_cal[np.newaxis, :, np.newaxis]

    # Drop first channel(s) and trim vignetted and dark columns
    # obs_image shape = (frames / lines, channels / bands, samples / columns)
    if moonager.verbose:
        print("Trimming image samples and channels to L1B size.")
    obs_image = obs_image[
                :,
                np.max(moonager.omitted_channels) + 1:,
                moonager.left_col_cutoff:moonager.right_col_cutoff
                ]

    # (11) Smooth Shape Correction
    ssc_factors = load_ssc_factors(moonager.ssc_path)
    obs_image = obs_image * ssc_factors[np.newaxis, :, np.newaxis]

    # (12) Ray tracing / location
    # not implemented

    return obs_image


def run_new_pipeline(moonager: PipeManager):
    """
    DPSIS with creative liberties.
    """
    from l0_l1b.utils.loader import load_fits_into_frame
    from l0_l1b.utils.dark_obs import make_dark_signal_image, \
        experimental_dark_pedestal_correction
    from l0_l1b.utils.electronic_ghost import electronic_panel_ghost_correction
    from l0_l1b.utils.mission_bde import bad_detector_element_correction, \
        detector_array_tap_interpolation
    from l0_l1b.utils.mission_flat import apply_flat

    obs_image = load_fits_into_frame(moonager.obs_path)

    # Dark Signal Subtraction
    # leaving dark cols of obs image unaffected for pedestal
    # correction
    dark_signal_image = make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_cols=moonager.dark_cols,
        dark_method='median'
    )
    obs_image -= dark_signal_image

    # Dark Pedestal Shift Correction
    obs_image = experimental_dark_pedestal_correction(
        obs_image=obs_image,
        dark_path=moonager.dark_path,
        dark_cols=moonager.dark_cols,
    )

    # Electronic Ghost Correction
    # fix ghosts before flagged pixels
    obs_image = electronic_panel_ghost_correction(
        obs_data=obs_image,
        l0_samples=moonager.l0_samples,
        ghost_correction=moonager.ghost_correction,
    )

    # Bad Detector Element Correction (Flag)
    obs_image = bad_detector_element_correction(
        obs_data=obs_image,
        bde_path=moonager.flag_path,
    )

    # Detector Tap Interpolation
    obs_image = detector_array_tap_interpolation(
        obs_data=obs_image,
        cols=moonager.read_out_cols
    )

    # Scattered Light Correction
    # not implemented

    # # Lab Flat Correction
    obs_image = apply_flat(
        obs_data=obs_image,
        flat_path=moonager.lab_flat_path,
    )

    # No lab based flat because we modified the lab flat?

    # Radiometric Calibration
    # not implemented

    # Smooth Shape Correction
    # not implemented

    # Ray tracing / location
    # not implemented

    return obs_image
