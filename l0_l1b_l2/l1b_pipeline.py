# M3 Calibration Pipelines for L0 to L1B processing

from astropy.io import fits
from l0_l1b_l2.reference import PipeManager


def make_dark_std_backplane(moonager: PipeManager):
    """
    Optional, not in original pipeline: Make dark std backplane, processed to
    radiance units.
    We skip some steps: dark pedestal, scattered light, flats. These all deal
    with effects that don't happen in a dark signal image. The flats especially
    would introduce rippled light. I don't know if this will end up being very
    useful, but given how much dark signal images vary at higher temperatures,
    the idea is to get an estimate of the error that could be on a pixel
    (ie if the background varies by 10 DN during an obs, here is that same 10
    DN in radiance units).
    """
    import numpy as np
    from l0_l1b_l2.l1b_utils.dark_obs import make_dark_signal_image
    from l0_l1b_l2.l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation, filter_seam_interpolation
    from l0_l1b_l2.l1b_utils.smooth_shape import load_ssc_factors
    from l0_l1b_l2.l1b_utils.radiometric_calibration import load_rdn_cal

    # we do load a lot of stuff twice doing this separate from the
    # main pipeline. but they are all small files. so IDK.
    dark_std = make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_method='std'
    )
    dark_std = bde_correction(
        obs_data=dark_std,
        bde_path=moonager.flag_path,
    )
    dark_std = detector_array_tap_interpolation(
        obs_data=dark_std,
        cols=moonager.read_out_cols
    )

    # filter seams don't show up in darks, but bc the real values get removed
    # in the obs, we also interpolate.
    # dark_std = filter_seam_interpolation(
    #     obs_data=dark_std,
    #     channels=moonager.filter_seam_rows
    # )
    rdn_cal = load_rdn_cal(moonager.rdn_cal_path)
    dark_std = dark_std * rdn_cal[:, np.newaxis]
    dark_std = dark_std[
                np.max(moonager.omitted_channels) + 1:,
                moonager.left_col_cutoff:moonager.right_col_cutoff
                ]
    ssc_factors = load_ssc_factors(moonager.ssc_path)
    dark_std = dark_std * ssc_factors[:, np.newaxis]

    return dark_std


def run_l1b_mission_pipeline(moonager: PipeManager):
    """
    L0 to L1B Pipeline based on an originalist reading of the DPSIS.
    If moonager.backplanes = True, returns the obs image and the dark std
    processed to radiance. Else, just returns the obs in radiance.
    """
    import numpy as np
    from l0_l1b_l2.l1b_utils.loader import load_fits_into_frame
    from l0_l1b_l2.l1b_utils.dark_obs import make_dark_signal_image, \
        basic_dark_pedestal_correction
    from l0_l1b_l2.l1b_utils.electronic_ghost import ghost_correction
    from l0_l1b_l2.l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation, filter_seam_interpolation
    from l0_l1b_l2.l1b_utils.mission_flat import apply_flat
    from l0_l1b_l2.l1b_utils.smooth_shape import load_ssc_factors
    from l0_l1b_l2.l1b_utils.radiometric_calibration import load_rdn_cal
    from l0_l1b_l2.l1b_utils.scattered_light import basic_scattered_light_corr
    from l0_l1b_l2.reference import check_l1b_label

    obs_image = load_fits_into_frame(moonager.l0_obs_path)
    # obs_image shape = (frames / lines, channels / bands, samples / columns)

    # (1) Dark Signal Subtraction
    if moonager.verbose:
        print("Subtracting dark signal.")
    obs_image -= make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_method='mean'
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_dss.fits",
            np.flip(obs_image.transpose(1, 0, 2), axis=(1, 2)),
            overwrite=True
        )
    # (2) Bad Detector Element Correction (Flag)
    if moonager.verbose:
        print("Running flagged pixel correction.")
    obs_image = bde_correction(
        obs_data=obs_image,
        bde_path=moonager.flag_path,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_bde.fits",
            obs_image[100:200, :, :],
            overwrite=True
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
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_fs_tap.fits",
            obs_image[100:200, :, :],
            overwrite=True
        )
    # (5) Electronic Ghost Correction
    if moonager.verbose:
        print("Running electronic ghost correction.")
    obs_image = ghost_correction(
        obs_data=obs_image,
        l0_samples=moonager.l0_samples,
        correction_factor=moonager.ghost_corr_factor,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_ghost.fits",
            obs_image[100:200, :, :],
            overwrite=True
        )
    # (6) Dark Pedestal Shift Correction
    if moonager.verbose:
        print("Running dark pedestal shift correction.")
    obs_image = basic_dark_pedestal_correction(
        obs_image=obs_image,
        dark_cols=moonager.dark_cols,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_pedestal.fits",
            obs_image[100:200, :, :],
            overwrite=True,
        )
    # (7) Scattered Light Correction
    if moonager.verbose:
        print("Applying scattered light correction.")
    obs_image = basic_scattered_light_corr(
        obs_image=obs_image,
        left_cols=moonager.vignetted_cols_left,
        right_cols=moonager.vignetted_cols_right,
        all_cols=moonager.vignetted_cols,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_sl.fits",
            obs_image[100:200, :, :],
            overwrite=True
        )
    # (8) Lab Flat Correction
    if moonager.verbose:
        print("Applying lab flat.")
    obs_image = apply_flat(
        obs_data=obs_image,
        flat_path=moonager.lab_flat_path,
        # flag_path=moonager.flag_path
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_labflat.fits",
            obs_image[100:200, :, :],
            overwrite=True
        )
    # (9) Imaging-based Flat Correction
    if moonager.verbose:
        print("Applying observation-level flat.")
    obs_image = apply_flat(
        obs_data=obs_image,
        flat_path=moonager.obs_flat_path,
        # flag_path=moonager.flag_path
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_obsflat.fits",
            obs_image[100:200, :, :],
            overwrite=True
        )
    # (10) Radiometric Calibration
    rdn_cal = load_rdn_cal(moonager.rdn_cal_path)
    obs_image = obs_image * rdn_cal[np.newaxis, :, np.newaxis]
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_radcal.fits",
            obs_image[100:200, :, :],
            overwrite=True
        )
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
    # TODO: flip things around to the orientation used in level 2 etc.
    #   For now, we check for a relevant L1B label which gives orientation info
    #   and then flip around accordingly. if there is no L1B label, return as
    #   is.

    # load orientation info from L1B label
    reverse_lines, reverse_samples = check_l1b_label(moonager.l1b_label)

    if reverse_lines:
        obs_image = obs_image[::-1, :, :]
    if reverse_samples:
        obs_image = obs_image[:, :, ::-1]

    # make backplanes, if you want
    if moonager.backplanes:
        dark_std = make_dark_std_backplane(moonager)
        if reverse_samples:
            dark_std = dark_std[:, ::-1]

        return obs_image, dark_std

    return obs_image


def run_l1b_new_pipeline(moonager: PipeManager):
    """
    DPSIS with creative liberties.
    """
    from l0_l1b_l2.l1b_utils.loader import load_fits_into_frame
    from l0_l1b_l2.l1b_utils.dark_obs import make_dark_signal_image, \
        experimental_dark_pedestal_correction
    from l0_l1b_l2.l1b_utils.electronic_ghost import ghost_correction
    from l0_l1b_l2.l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation
    from l0_l1b_l2.l1b_utils.new_flat import get_relative_gain_flat

    obs_image = load_fits_into_frame(moonager.l0_obs_path)

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
        bde_path=moonager.flag_path,
    )

    # Electronic Ghost Correction
    # fix ghosts before flagged pixels
    obs_image = ghost_correction(
        obs_data=obs_image,
        l0_samples=moonager.l0_samples,
        correction_factor=moonager.ghost_corr_factor,
    )

    # Bad Detector Element Correction (Flag)
    obs_image = bde_correction(
        obs_data=obs_image,
        bde_path=moonager.flag_path,
    )

    # Detector Tap Interpolation
    obs_image = detector_array_tap_interpolation(
        obs_data=obs_image,
        cols=moonager.read_out_cols,
    )

    # Scattered Light Correction
    # not implemented

    # New flat
    flat = get_relative_gain_flat(
        obs_image=obs_image,
        moonager=moonager,
    )
    obs_image = obs_image * flat

    # Radiometric Calibration
    # not implemented

    # Smooth Shape Correction
    # not implemented

    # Ray tracing / location
    # not implemented

    return obs_image
