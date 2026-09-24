import numpy as np
from l0_l1b_l2.reference import PipeManager
from pathlib import Path


def make_dark_std_backplane(moonager: PipeManager):
    """
    Optional, not in original pipeline: Make dark std backplane, processed to
    radiance units.
    We skip some steps: dark pedestal, scattered light, flats. These all deal
    with effects that don't happen in a dark signal image.

    The flats especially would introduce rippled light. Therefore, the relative
    magnitudes are not 100% correct.
    TODO: Need to think about if light effects would be an issue for
        meaning of dark std backplane?

    I don't know if this will end up being very useful, but given how much
    dark signal images vary at higher temperatures, the idea is to get an
    estimate of the error that could be on a pixel (ie if the background
    varies by 10 DN during an obs, here is that same 10 DN in radiance units).
    """
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


def _bit(mask, bit):
    return mask.astype(np.uint8) * np.uint8(bit)


def make_flag_backplane(
        flag_path: str,
        bad_col_group_map: np.ndarray,
        obs_image: np.ndarray,
):
    """
    Combine all flags into backplane of L1B observation size.

    0 = OK data
    1 = negative L0 values, rollover high DNs (usually -2 or -3 in DN)
    2 = flag map, interpolated values, all lines per sample
    3 = bad columns, all lines per sample (secondary flagging)
    4 = bad column groups, not all lines (secondary flagging)
    5 = very high std, flashing (secondary flagging)
    6 = negative L1B values (secondary flagging)

    """
    from .loader import load_fits_into_frame
    from l0_l1b_l2.l1b_utils.secondary_flagging import build_bad_col_map

    nan = 1
    bde = 2
    block_col = 4
    bad_col = 8

    # bit 0: NaN in obs, set to NaN because of negative values in L0
    # flags shape bands, lines, samples
    flags = _bit(np.isnan(obs_image), nan)

    # bit 1: interpolated pixel in BDE map
    # bde map shape bands, samples
    bde_map = np.asarray(load_fits_into_frame(Path(flag_path)))
    flags |= _bit(bde_map > 0, bde)[:, np.newaxis, :]

    # bit 2: variable column block
    # bad col group map shape lines, samples
    flags |= _bit(bad_col_group_map > 0, block_col)[np.newaxis, :, :]
    # bit 3: bad columns (ie bad flat or missed bright / dark pixel)
    # bad cols shape lines, samples
    bad_cols = np.asarray(build_bad_col_map(obs_image))
    flags |= _bit(bad_cols > 1, bad_col)[:, np.newaxis, :]

    return flags
