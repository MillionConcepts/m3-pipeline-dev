# M3 Calibration Pipelines for L0 to L1B processing

from astropy.io import fits
import numpy as np
from l0_l1b_l2.reference import PipeManager


def run_l1b_mission_pipeline(moonager: PipeManager):
    """
    L0 to L1B Pipeline based on an originalist reading of the DPSIS.
    If moonager.backplanes = True, returns the obs image and the dark std
    processed to radiance. Else, just returns the obs in radiance.
    """
    from l0_l1b_l2.l1b_utils.loader import load_fits_into_frame
    from l0_l1b_l2.l1b_utils.dark_obs import make_dark_signal_image, \
        basic_dark_pedestal_correction,\
        illumination_based_dark_pedestal_correction
    from l0_l1b_l2.l1b_utils.electronic_ghost import ghost_correction
    from l0_l1b_l2.l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation, filter_seam_interpolation
    from l0_l1b_l2.l1b_utils.mission_flat import apply_flat
    from l0_l1b_l2.l1b_utils.smooth_shape import load_ssc_factors
    from l0_l1b_l2.l1b_utils.radiometric_calibration import load_rdn_cal
    from l0_l1b_l2.l1b_utils.scattered_light import apply_scattered_light_corr
    from l0_l1b_l2.reference import check_l1b_label
    from l0_l1b_l2.l1b_utils.make_backplanes import make_dark_std_backplane, \
        make_flag_backplane
    from l0_l1b_l2.l1b_utils.new_flat import fix_variable_columns

    obs_image = load_fits_into_frame(moonager.l0_obs_path)
    # obs_image shape = (frames / lines, channels / bands, samples / columns)
    obs_image[obs_image < 0] = np.nan

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
            obs_image[:, :, :],
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
            obs_image[:, :, :],
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
            obs_image[:, :, :],
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
            obs_image[:, :, :],
            overwrite=True
        )

    # (6) Dark Pedestal Shift Correction
    if moonager.verbose:
        print("Running dark pedestal shift correction.")
    obs_image = basic_dark_pedestal_correction(
        obs_image=obs_image,
        dark_cols=moonager.dark_cols,
    )
    # obs_image = illumination_based_dark_pedestal_correction(
    #             obs_image=obs_image,
    #             left_cutoff_col=moonager.left_col_cutoff,
    #             right_cutoff_col=moonager.right_col_cutoff,
    # )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_pedestal.fits",
            obs_image[:, :, :],
            overwrite=True,
        )

    # Variable column group correction
    # (time-variable organized flashing of background columns
    # is not good for any kind of flat based on full obs length
    # averages)
    if moonager.verbose:
        print("Looking for variable columns.")
    obs_image, bad_col_group_map = fix_variable_columns(
        obs_image=obs_image.transpose(1, 0, 2),
        col_groups=moonager.bad_column_groups,
    )

    # (7) Scattered Light Correction
    if moonager.verbose:
        print("Applying scattered light correction.")
    obs_image = apply_scattered_light_corr(
        obs_image=obs_image.transpose(1, 0, 2),
        obs_type=moonager.mode,
        sl_ratio_corr=True,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_sl.fits",
            obs_image[:, :, :],
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
            obs_image[:, :, :],
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
            obs_image[:, :, :],
            overwrite=True
        )

    # (10) Radiometric Calibration
    rdn_cal = load_rdn_cal(moonager.rdn_cal_path)
    obs_image = obs_image * rdn_cal[:, np.newaxis, np.newaxis]
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_radcal.fits",
            obs_image[:, :, :],
            overwrite=True
        )

    # make overall flag map by combining other flag maps
    if moonager.backplanes:
        if moonager.verbose:
            print("Making flag backplane.")
        flag_backplane = make_flag_backplane(
            moonager.flag_path,
            bad_col_group_map,
            obs_image
        )
        del bad_col_group_map

    # Drop first channel(s) and trim vignetted and dark columns
    # obs_image shape = (frames / lines, channels / bands, samples / columns)
    if moonager.verbose:
        print("Trimming image samples and channels to L1B size.")
    obs_image = obs_image[
                np.max(moonager.omitted_channels) + 1:,
                :,
                moonager.left_col_cutoff:moonager.right_col_cutoff
                ]
    # # (11) Smooth Shape Correction
    ssc_factors = load_ssc_factors(moonager.ssc_path)
    obs_image = obs_image * ssc_factors[:, np.newaxis, np.newaxis]
    # (12) Ray tracing / location
    # TODO: flip things around to the orientation used in level 2 etc.
    #   For now, we check for a relevant L1B label which gives orientation info
    #   and then flip around accordingly. if there is no L1B label, return as
    #   is.

    # load orientation info from L1B label
    reverse_lines, reverse_samples = check_l1b_label(moonager.l1b_label)

    if reverse_lines:
        obs_image = obs_image[:, ::-1, :]
    if reverse_samples:
        obs_image = obs_image[:, :, ::-1]

    # make backplanes, if you want
    if moonager.backplanes:
        dark_std = make_dark_std_backplane(moonager)

        # trim flag map to L1B size
        flag_backplane = flag_backplane[
                np.max(moonager.omitted_channels) + 1:,
                :,
                moonager.left_col_cutoff:moonager.right_col_cutoff
                ]

        if reverse_lines:
            flag_backplane = flag_backplane[:, ::-1, :]
        if reverse_samples:
            dark_std = dark_std[:, ::-1]
            flag_backplane = flag_backplane[:, :, :-1]

        return obs_image, dark_std, flag_backplane

    return obs_image


def run_basic_cleanup_l0(moonager: PipeManager):
    """
    L0 pipeline just for cleaning up L0 data (bad elements, flatting,
    seam interpolation, ghosts, and DSS). No pedestal, scattered light,
    rad cal or smooth shape. Result is in DN.
    """
    from l0_l1b_l2.l1b_utils.loader import load_fits_into_frame
    from l0_l1b_l2.l1b_utils.dark_obs import make_dark_signal_image
    from l0_l1b_l2.l1b_utils.electronic_ghost import ghost_correction
    from l0_l1b_l2.l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation, filter_seam_interpolation
    from l0_l1b_l2.l1b_utils.mission_flat import apply_flat
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
    # (2) Bad Detector Element Correction (Flag)
    if moonager.verbose:
        print("Running flagged pixel correction.")
    obs_image = bde_correction(
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
    obs_image = ghost_correction(
        obs_data=obs_image,
        l0_samples=moonager.l0_samples,
        correction_factor=moonager.ghost_corr_factor,
    )
    obs_image = obs_image.transpose(1, 0, 2)

    # (8) Lab Flat Correction
    if moonager.verbose:
        print("Applying lab flat.")
    obs_image = apply_flat(
        obs_data=obs_image,
        flat_path=moonager.lab_flat_path,
        # flag_path=moonager.flag_path
    )
    # (9) Imaging-based Flat Correction
    if moonager.verbose:
        print("Applying observation-level flat.")
    obs_image = apply_flat(
        obs_data=obs_image,
        flat_path=moonager.obs_flat_path,
        # flag_path=moonager.flag_path
    )

    # Run BDE again because their lab and obs flat don't account for flagged
    # pixels well
    obs_image = bde_correction(
        obs_data=obs_image.transpose(1, 0, 2),
        bde_path=moonager.flag_path,
    )

    # Drop first channel(s) and trim vignetted and dark columns
    # obs_image shape = (frames / lines, channels / bands, samples / columns)
    if moonager.verbose:
        print("Trimming image samples and channels to L1B size.")
    obs_image = obs_image[
                np.max(moonager.omitted_channels) + 1:,
                :,
                moonager.left_col_cutoff:moonager.right_col_cutoff
                ]

    # load orientation info from L1B label
    reverse_lines, reverse_samples = check_l1b_label(moonager.l1b_label)

    if reverse_lines:
        obs_image = obs_image[:, ::-1, :]
    if reverse_samples:
        obs_image = obs_image[:, :, ::-1]

    return obs_image


def run_l1b_new_pipeline(moonager: PipeManager):
    """
    DPSIS with creative liberties.
    """
    from l0_l1b_l2.l1b_utils.loader import load_fits_into_frame
    from l0_l1b_l2.l1b_utils.dark_obs import make_dark_signal_image, \
        illumination_based_dark_pedestal_correction
    from l0_l1b_l2.l1b_utils.electronic_ghost import ghost_correction
    from l0_l1b_l2.l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation
    from l0_l1b_l2.l1b_utils.mission_flat import make_flat_field_from_obs
    from l0_l1b_l2.l1b_utils.new_flat import get_relative_gain_flat, \
        fix_variable_columns
    from l0_l1b_l2.l1b_utils.radiometric_calibration import load_rdn_cal
    from l0_l1b_l2.l1b_utils.scattered_light import apply_scattered_light_corr
    from l0_l1b_l2.reference import check_l1b_label
    from l0_l1b_l2.l1b_utils.make_backplanes import make_dark_std_backplane, \
        make_flag_backplane

    obs_image = load_fits_into_frame(moonager.l0_obs_path)
    obs_image[obs_image < 0] = np.nan

    # Dark Signal Subtraction
    # leaving dark cols of obs image unaffected for pedestal
    # correction
    if moonager.verbose:
        print("Subtracting dark signal.")
    dark_signal_image = make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_cols=moonager.dark_cols,
        dark_method='mean'
    )
    obs_image -= dark_signal_image
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.local_root}/{moonager.obs_id}_dss.fits",
            obs_image[:, :, :],
            overwrite=True
        )
        fits.writeto(
            f"{moonager.local_root}/{moonager.obs_id}_dark_signal.fits",
            dark_signal_image,
            overwrite=True
        )

    # Dark Pedestal Shift Correction
    if moonager.verbose:
        print("Applying dark pedestal shift correction.")
    obs_image = illumination_based_dark_pedestal_correction(
        obs_image=obs_image,
        left_cutoff_col=moonager.left_col_cutoff,
        right_cutoff_col=moonager.right_col_cutoff,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.local_root}/{moonager.obs_id}_pedestal.fits",
            obs_image[:, :, :],
            overwrite=True,
        )

    # Electronic Ghost Correction
    # fix ghosts before flagged pixels
    if moonager.verbose:
        print("Running ghost correction. Spooky!")
    obs_image = ghost_correction(
        obs_data=obs_image,
        l0_samples=moonager.l0_samples,
        correction_factor=moonager.ghost_corr_factor,
    )

    # Bad Detector Element Correction (Flag)
    if moonager.verbose:
        print("Running bad detector element correction.")
    obs_image = bde_correction(
        obs_data=obs_image,
        bde_path=moonager.flag_path,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.local_root}/{moonager.obs_id}_bde.fits",
            obs_image[:, :, :],
            overwrite=True
        )
    # Detector Tap Interpolation
    if moonager.verbose:
        print("Running TAP array interpolation.")
    obs_image = detector_array_tap_interpolation(
        obs_data=obs_image,
        cols=moonager.read_out_cols,
    )

    # Variable column group correction
    # (time-variable organized flashing of background columns
    # is not good for any kind of flat based on full obs length
    # averages)
    if moonager.verbose:
        print("Looking for variable columns.")
    obs_image, bad_col_group_map = fix_variable_columns(
        obs_image=obs_image.transpose(1, 0, 2),
        col_groups=moonager.bad_column_groups,
    )

    # Scattered Light Correction
    if moonager.verbose:
        print("Applying scattered light correction.")
    obs_image = apply_scattered_light_corr(
        obs_image=obs_image.transpose(1, 0, 2),
        obs_type=moonager.mode,
        sl_ratio_corr=True,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.local_root}/{moonager.obs_id}_sl.fits",
            obs_image[:, :, :],
            overwrite=True
        )

    # New flat
    if moonager.verbose:
        print("Making new flat.")
    # flat = get_relative_gain_flat(
    #     obs_image=obs_image.transpose(1, 0, 2),
    #     moonager=moonager,
    # )
    flat = make_flat_field_from_obs(obs_image)
    obs_image = obs_image * flat[:, np.newaxis, :]
    if moonager.verbose:
        print(f"Writing new flat to: {moonager.obs_id}_new_flat.fits.")
    fits.writeto(
        f"{moonager.local_root}/{moonager.obs_id}_new_flat.fits",
        flat,
        overwrite=True
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.local_root}/{moonager.obs_id}_flatted.fits",
            obs_image[:, :, :],
            overwrite=True
        )

    # Radiometric Calibration
    if moonager.verbose:
        print("Converting DN to radiance.")
    rdn_cal = load_rdn_cal(moonager.rdn_cal_path)
    obs_image = obs_image * rdn_cal[:, np.newaxis, np.newaxis]

    # Smooth Shape Correction
    # not implemented

    # Ray tracing / location
    # not implemented

    # make overall flag map by combining other flag maps
    if moonager.backplanes:
        if moonager.verbose:
            print("Making flag backplane.")
        flag_backplane = make_flag_backplane(
            moonager.flag_path,
            bad_col_group_map,
            obs_image
        )
        del bad_col_group_map

    # load orientation info from L1B label
    reverse_lines, reverse_samples = check_l1b_label(moonager.l1b_label)

    if reverse_lines:
        obs_image = obs_image[:, ::-1, :]
    if reverse_samples:
        obs_image = obs_image[:, :, ::-1]

    # trim to L1B size
    if moonager.verbose:
        print("Trimming image samples and channels to L1B size.")
    obs_image = obs_image[
                np.max(moonager.omitted_channels) + 1:,
                :,
                moonager.left_col_cutoff:moonager.right_col_cutoff
                ]

    # make backplanes, if you want
    if moonager.backplanes:
        # make dark backplane
        dark_std = make_dark_std_backplane(moonager)

        # trim flag map to L1B size
        flag_backplane = flag_backplane[
                np.max(moonager.omitted_channels) + 1:,
                moonager.left_col_cutoff:moonager.right_col_cutoff
                ]

        # fix to L1B orientations
        if reverse_lines:
            flag_backplane = flag_backplane[:, ::-1, :]
        if reverse_samples:
            dark_std = dark_std[:, ::-1]
            flag_backplane = flag_backplane[:, :, :-1]

        return obs_image, dark_std, flag_backplane

    return obs_image
