

from ..reference import PipeManager


def make_dark_std_backplane(moonager: PipeManager):
    """
    Optional, not in original pipeline: Make dark std backplane, processed to
    radiance units.
    We skip some steps: dark pedestal, scattered light, flats. These all deal
    with effects that don't happen in a dark signal image.

    The flats especially would introduce rippled light. Therefore, the relative
    magnitudes are not 100% correct.

    I don't know if this will end up being very useful, but given how much
    dark signal images vary at higher temperatures, the idea is to get an
    estimate of the error that could be on a pixel (ie if the background
    varies by 10 DN during an obs, here is that same 10 DN in radiance units).
    """
    import numpy as np
    from .dark_obs import make_dark_signal_image
    from .mission_bde import bde_correction, \
        detector_array_tap_interpolation
    from .smooth_shape import load_ssc_factors_from_table
    from .radiometric_calibration import load_rdn_cal

    # we do load a lot of stuff twice doing this separate from the
    # main pipeline. but they are all small files. so IDK.
    dark_std = make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_method='std'
    )
    dark_std = bde_correction(
        obs_image=dark_std,
        bde_path=moonager.flag_path,
        extra_bde_path=moonager.additional_flag_path
    )
    dark_std = detector_array_tap_interpolation(
        obs_image=dark_std,
        cols=moonager.read_out_cols
    )

    rdn_cal = load_rdn_cal(moonager.rdn_cal_path)
    dark_std = dark_std * rdn_cal[:, np.newaxis]
    dark_std = dark_std[
               np.max(moonager.omitted_channels) + 1:,
               moonager.left_col_cutoff:moonager.right_col_cutoff
               ]
    ssc_factors = load_ssc_factors_from_table(moonager.obs_id,
                                              moonager.ssc_table_path)
    dark_std = dark_std * ssc_factors[:, np.newaxis]

    return dark_std
