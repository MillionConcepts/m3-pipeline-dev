# M3 Calibration Pipelines for L0 to L1B processing

from astropy.io import fits
import numpy as np
from .reference import PipeManager, check_shape


def run_l1b_mission_pipeline(moonager: PipeManager):
    """
    L0 to L1B Pipeline based on an originalist reading of the DPSIS.
    If moonager.backplanes = True, returns the obs image and the dark std
    processed to radiance. Else, just returns the obs in radiance.
    """
    from .utils.loader import load_fits_into_frame, orient_and_trim_to_l1b
    from .l1b_utils.dark_obs import make_dark_signal_image, \
        basic_dark_pedestal_correction
    from .l1b_utils.electronic_ghost import ghost_correction
    from .l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation, filter_seam_interpolation
    from .l1b_utils.mission_flat import apply_flat
    from .l1b_utils.smooth_shape import apply_ssc_factors
    from .l1b_utils.radiometric_calibration import apply_rdn_cal
    from .l1b_utils.scattered_light import apply_scattered_light_corr
    from .l1b_utils.dark_backplane import make_dark_std_backplane
    from .flagging.make_flag_backplanes import make_flag_backplane
    from .flagging.flicker_flagging import build_flicker_maps

    # standard obs_image shape for this software will be axis 0 = lines aka
    # frames aka temporal direction, so:
    # obs_image = [lines, bands, samples]

    obs_image = load_fits_into_frame(moonager.l0_path)
    check_shape(
        obs_image.shape,
        (moonager.l0_channels, moonager.l0_samples),
        'loading l0 obs data'
    )
    obs_image[obs_image < 0] = np.nan

    # (1) Dark Signal Subtraction
    if moonager.verbose:
        print("Subtracting dark signal.")
    obs_image -= make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_method='median'
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
        obs_image=obs_image,
        bde_path=moonager.flag_path,
        ignore_bands=moonager.omitted_channels,
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
        obs_image=obs_image,
        cols=moonager.read_out_cols
    )

    # (4) Filter Seam Interpolation
    if moonager.verbose:
        print("Interpolating filter seams.")
    obs_image = filter_seam_interpolation(
        obs_image=obs_image,
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
        obs_image=obs_image,
        l0_samples=moonager.l0_samples,
        correction_factor=moonager.ghost_corr_factor,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_ghost.fits",
            obs_image[:, :, :],
            overwrite=True
        )

    # Variable column group correction
    # (time-variable organized flashing of background columns
    # is not good for any kind of flat based on full obs length
    # averages)
    if moonager.verbose:
        print("Looking for variable columns.")
    # Variable column group correction
    # (time-variable organized flashing of background columns
    # is not good for any kind of flat based on full obs length
    # averages)
    if moonager.verbose:
        print("Looking for variable columns.")
    flicker_map, _ = build_flicker_maps(
        obs_image=obs_image,
        bad_col_groups=moonager.flicker_col_groups,
        offset=False,
        bands=moonager.l0_flicker_bands,
        l1b_samples=moonager.l1b_samples,
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
            obs_image[:, :, :],
            overwrite=True,
        )

    # (7) Scattered Light Correction
    if moonager.verbose:
        print("Applying scattered light correction.")
    obs_image = apply_scattered_light_corr(
        obs_image=obs_image,
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
        obs_image=obs_image,
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
        obs_image=obs_image,
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
    if moonager.verbose:
        print("Applying radiometric calibration coefficients.")
    obs_image = apply_rdn_cal(obs_image, moonager.rdn_cal_path)
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_radcal.fits",
            obs_image[:, :, :],
            overwrite=True
        )

    # Make overall flag map by combining other flag maps
    if moonager.backplanes:
        if moonager.verbose:
            print("Making flag backplane.")
        flag_backplane = make_flag_backplane(
            flags=flicker_map,
            obs_image=obs_image,
            flag_path=moonager.flag_path,
            deprecated_channels=moonager.degraded_channels,
            omitted_channels=moonager.omitted_channels,
        )
        del flicker_map

    # Drop first channel(s) and trim vignetted and dark columns
    # obs_image shape = (frames / lines, channels / bands, samples / columns)
    if moonager.verbose:
        print("Trimming image samples and channels to L1B size.")
    obs_image = orient_and_trim_to_l1b(
        input_array=obs_image,
        left_col_cutoff=moonager.left_col_cutoff,
        right_col_cutoff=moonager.right_col_cutoff,
        reverse_lines=moonager.reverse_lines,
        reverse_samples=moonager.reverse_samples,
        omitted_channels=moonager.omitted_channels,
    )

    # # (11) Smooth Shape Correction
    if moonager.smooth_shape:
        if moonager.verbose:
            print("Applying smooth shape correction.")
        obs_image = apply_ssc_factors(
            obs_image,
            moonager.obs_id,
            moonager.ssc_table_path
        )

    # (12) Ray tracing / location
    # TODO: flip things around to the orientation used in level 2 etc.
    #   For now, we check for a relevant L1B label which gives orientation info
    #   and then flip around accordingly. if there is no L1B label, return as
    #   is.

    # use orientation info from L1B label
    if moonager.reverse_lines:
        obs_image = obs_image[::-1, :, :]
    if moonager.reverse_samples:
        obs_image = obs_image[:, :, ::-1]

    # make backplanes, if you want
    if moonager.backplanes:
        dark_std = make_dark_std_backplane(moonager)

        # trim flag map to L1B size
        flag_backplane = flag_backplane[
                :,
                np.max(moonager.omitted_channels) + 1:,
                moonager.left_col_cutoff:moonager.right_col_cutoff
                ]

        if moonager.reverse_lines:
            flag_backplane = flag_backplane[::-1, :, :]
        if moonager.reverse_samples:
            dark_std = dark_std[:, ::-1]
            flag_backplane = flag_backplane[:, :, ::-1]

        return obs_image, dark_std, flag_backplane

    return obs_image


def run_l1b_new_pipeline(moonager: PipeManager):
    """
    DPSIS with creative liberties.
    """
    from .utils.loader import load_fits_into_frame, orient_and_trim_to_l1b
    from .l1b_utils.dark_obs import make_dark_signal_image, \
        per_band_dark_pedestal_correction
    from .l1b_utils.electronic_ghost import ghost_correction
    from .l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation, filter_seam_interpolation
    from .l1b_utils.mission_flat import \
        get_shadowing_from_lab_flat, make_flat_field_from_obs_with_masking
    from .l1b_utils.radiometric_calibration import apply_rdn_cal
    from .l1b_utils.smooth_shape import apply_ssc_factors
    from .l1b_utils.dark_backplane import make_dark_std_backplane
    from .l1b_utils.scattered_light import apply_scattered_light_corr
    from .flagging.make_flag_backplanes import make_flag_backplane
    from .flagging.flicker_flagging import build_flicker_maps

    # standard obs_image shape for this software will be axis 0 = lines aka
    # frames aka temporal direction, so:
    # obs_image = [lines, bands, samples]

    obs_image = load_fits_into_frame(moonager.l0_path)
    check_shape(
        obs_image.shape,
        (moonager.l0_channels, moonager.l0_samples),
        'loading l0 obs data'
    )
    obs_image[obs_image < 0] = np.nan

    # (1) Dark Signal Subtraction
    if moonager.verbose:
        print("Subtracting dark signal.")
    obs_image -= make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_method='median'
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
        obs_image=obs_image,
        bde_path=moonager.flag_path,
        extra_bde_path=moonager.additional_flag_path,
        ignore_bands=moonager.omitted_channels,
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
        obs_image=obs_image,
        cols=moonager.read_out_cols
    )

    # (4) Filter Seam Interpolation
    if moonager.verbose:
        print("Interpolating filter seams.")
    obs_image = filter_seam_interpolation(
        obs_image=obs_image,
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
        obs_image=obs_image,
        l0_samples=moonager.l0_samples,
        correction_factor=moonager.ghost_corr_factor,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_ghost.fits",
            obs_image[:, :, :],
            overwrite=True
        )

    # Variable column group correction
    # (time-variable organized flashing of background columns
    # is not good for any kind of flat based on full obs length
    # averages)
    if moonager.verbose:
        print("Looking for variable columns.")
    flicker_map, offset_map = build_flicker_maps(
        obs_image=obs_image,
        bad_col_groups=moonager.flicker_col_groups,
        offset=True,
        bands=moonager.l0_flicker_bands,
        l1b_samples=moonager.l1b_samples,
    )
    # add offsets where one offset map defined for all bands.
    # this might change.
    obs_image = obs_image + offset_map[:, np.newaxis, :]
    del offset_map

    # (6) Dark Pedestal Shift Correction
    if moonager.verbose:
        print("Running dark pedestal shift correction.")
    obs_image = per_band_dark_pedestal_correction(
        obs_image=obs_image,
        dark_cols=moonager.dark_cols,
    )

    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_pedestal.fits",
            obs_image[:, :, :],
            overwrite=True,
        )

    # if moonager.verbose:
    #     print("Adding back band dependence in dark signal.")
    # obs_image = remove_band_dependence_in_dark_signal(
    #     obs_image=obs_image,
    #     dark_path=moonager.dark_path
    # )

    # (7) Scattered Light Correction
    if moonager.verbose:
        print("Applying scattered light correction.")
    obs_image = apply_scattered_light_corr(
        obs_image=obs_image,
        obs_type=moonager.mode,
        sl_ratio_corr=True,
    )
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_sl.fits",
            obs_image[:, :, :],
            overwrite=True
        )

    # New flat
    if moonager.verbose:
        print("Making new flat.")
    shadow_surface = get_shadowing_from_lab_flat(
        flat_path=moonager.lab_flat_path,
        left_cutoff=moonager.left_col_cutoff,
        right_cutoff=moonager.right_col_cutoff,
        mode=moonager.mode,
    )
    obs_image[:, :, moonager.left_col_cutoff:moonager.right_col_cutoff] *=\
        shadow_surface[np.newaxis, :, :]
    flat = make_flat_field_from_obs_with_masking(
            obs_image,
            left_col_cutoff=moonager.left_col_cutoff,
            right_col_cutoff=moonager.right_col_cutoff,
            flag_path=moonager.flag_path,
            extra_flag_path=moonager.additional_flag_path,
            ignored_flag_cols=moonager.read_out_cols,
            ignored_flag_rows=moonager.filter_seam_rows,
            n_center=40
    )
    obs_image[:, :, moonager.left_col_cutoff:moonager.right_col_cutoff] /= \
        flat[np.newaxis, :, :]
    if moonager.verbose:
        print(f"Writing new flat to: {moonager.obs_id}_new_flat.fits.")
        #todo: save cal outputs as separate option for moonager?
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

    # (10) Radiometric Calibration
    if moonager.verbose:
        print("Applying radiometric calibration coefficients.")
    obs_image = apply_rdn_cal(obs_image, moonager.rdn_cal_path)
    if moonager.save_steps:
        fits.writeto(
            f"{moonager.obs_id}_radcal.fits",
            obs_image[:, :, :],
            overwrite=True
        )

    # Make overall flag map by combining other flag maps
    if moonager.backplanes:
        if moonager.verbose:
            print("Making flag backplane.")
        flag_backplane = make_flag_backplane(
            flag_path=moonager.flag_path,
            flags=flicker_map,
            obs_image=obs_image,
            deprecated_channels=moonager.degraded_channels,
            omitted_channels=moonager.omitted_channels,
            extra_flag_path=moonager.additional_flag_path,
        )
        flag_backplane = orient_and_trim_to_l1b(
            input_array=flag_backplane,
            left_col_cutoff=moonager.left_col_cutoff,
            right_col_cutoff=moonager.right_col_cutoff,
            reverse_lines=moonager.reverse_lines,
            reverse_samples=moonager.reverse_samples,
            omitted_channels=moonager.omitted_channels,
        )
        del flicker_map

    # Drop first channel(s) and trim vignetted and dark columns
    # obs_image shape = (frames / lines, channels / bands, samples / columns)
    if moonager.verbose:
        print("Trimming image samples and channels to L1B size.")
    obs_image = orient_and_trim_to_l1b(
        input_array=obs_image,
        left_col_cutoff=moonager.left_col_cutoff,
        right_col_cutoff=moonager.right_col_cutoff,
        reverse_lines=moonager.reverse_lines,
        reverse_samples=moonager.reverse_samples,
        omitted_channels=moonager.omitted_channels,
    )

    # # (11) Smooth Shape Correction
    if moonager.smooth_shape:
        if moonager.verbose:
            print("Applying smooth shape correction.")
        obs_image = apply_ssc_factors(
            obs_image,
            moonager.obs_id,
            moonager.ssc_table_path
        )

    # (12) Ray tracing / location
    # TODO: PUT GEOMETRY STUFF HERE PROBABLY

    # make backplanes, if you want
    if moonager.backplanes:
        dark_std = make_dark_std_backplane(moonager)
        return obs_image, dark_std, flag_backplane

    return obs_image
