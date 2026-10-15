import numpy as np
from l0_l1b_l2.reference import PipeManager
from pathlib import Path

# FLAGS FOR FLAG BACKPLANE
NAN = 1           # pixels set to NaN for being negative at L0 level (rollover)
BDE = 2           # pixels spectrally interpolated because flagged in the BDE
BLOCK_COL = 4     # pixels likely part of flashing ROIC background
BAD_COL = 8       # pixels part of a column offset from its neighbors in value
L2_DEPRECATED = 16  # bands flagged by the mission in L2 as deprecated

# Description mostly for writing to FITS header
FLAG_DICT = [
    ("FLG_NAN",  NAN,            "NaN, negative at L0 (rollover)"),
    ("FLG_BDE",  BDE,            "spectrally interpolated (BDE)"),
    ("FLG_FC",   BLOCK_COL,      "flashing col groups"),
    ("FLG_BADC", BAD_COL,        "column offset from neighbors"),
    ("FLG_DEP",  L2_DEPRECATED,  "Deprecated by mission in L2"),
]


def _bit(mask, bit):
    return mask.astype(np.uint8) * np.uint8(bit)


def make_flag_backplane(
        flag_path: Path,
        bad_col_group_map: np.ndarray,
        obs_image: np.ndarray,
        deprecated_channels: list[int] = None,
        extra_flag_path: Path = None,
):
    """
    Combine all flags into backplane of L1B observation size. Meant to be run
    as part of L0 to L1B processing, not afterwards. See
    make_mission_l1b_flag_backplane for processing a pre-existing L1B obs.
    """
    from .loader import load_bde_map
    from l0_l1b_l2.l1b_utils.secondary_flagging import build_bad_col_map

    # bit 0: NaN in obs, set to NaN because of negative values in L0
    # flags shape bands, lines, samples
    flags = _bit(np.isnan(obs_image), NAN)

    # bit 1: interpolated pixel in BDE map
    # the bde map has shape bands, samples
    bde_map = np.asarray(load_bde_map(Path(flag_path), extra_flag_path))
    flags |= _bit(bde_map > 0, BDE)[np.newaxis, :, :]

    # bit 2: variable column block
    # bad col group map shape lines, samples
    flags |= _bit(bad_col_group_map > 0, BLOCK_COL)[:, np.newaxis, :]

    # bit 3: bad columns (ie bad flat or missed bright / dark pixel)
    # bad cols shape band, samples
    bad_cols = np.asarray(build_bad_col_map(obs_image))
    flags |= _bit(bad_cols > 1, BAD_COL)[np.newaxis, :, :]

    # bit 4: flag deprecated bands in l2 that were typically set to -999 by the
    # mission in l2. they look normal in l1b and l0.
    if deprecated_channels is not None and len(deprecated_channels) > 0:
        # add one to each deprecated channel bc this is pre-trim to L1B length
        flags[:, [c + 1 for c in deprecated_channels], :] |= L2_DEPRECATED
    return flags


def flag_l1b(
        paths: dict,
        omitted_channels: list[int],
        left_col_cutoff: int,
        right_col_cutoff: int,
        reverse_lines: bool,
        reverse_samples: bool,
        deprecated_channels: list[int] = None,
):
    """
    From L1B data, and optionally L0 data and mission BDE files, build an
    L1B-shaped flag map with in the shape lines x bands x samples.

    dict paths keys: "l0_path", "l1b_path",  "flag_path"
    """
    from astropy.io import fits
    from .secondary_flagging import build_bad_col_map, get_l0_negative_flags
    from .loader import load_fits_into_frame

    # If the L0 data are not present, we make flag map arr based
    # on L1B shape. We check here to avoid having L0 data and L1B data loaded
    # at the same time later on. Otherwise, flags will get made when we check
    # L0 for negative pixels.
    if not paths['l0_path']:
        hdr = fits.getheader(paths['l1b_path'])
        flags = np.zeros(
            (hdr['NAXIS3'], hdr['NAXIS2'], hdr['NAXIS1']),
            dtype=np.uint8,
        )

    # bit 0: NaN in obs, set to NaN because of negative values in L0
    if paths['l0_path']:
        # Must load L0, flag negative, and then trim and orient correctly
        flags = get_l0_negative_flags(l0_path=paths['l0_path'])
        flags = flags[
                :,
                np.max(omitted_channels) + 1:,
                left_col_cutoff:right_col_cutoff
                ]
        if reverse_lines:
            flags = flags[::-1, :, :]
        if reverse_samples:
            flags = flags[:, :, ::-1]
        flags = _bit(flags == True, NAN)

    # bit 1: interpolated pixel in BDE map
    if paths['flag_path']:
        # Need to load and trim / orient to L1B s
        bde_map = np.asarray(load_fits_into_frame(paths['flag_path']))
        bde_map = bde_map[
                  np.max(omitted_channels) + 1:,
                  left_col_cutoff:right_col_cutoff
                  ]
        if reverse_samples:
            bde_map = bde_map[:, ::-1]
        flags |= _bit(bde_map > 0, BDE)[np.newaxis, :, :]

    # flagging for the following bits requires the L1B data
    obs_image = load_fits_into_frame(paths['l1b_path'])

    # bit 2: variable column block
    # bad col group map shape lines, samples
    # flags |= _bit(bad_col_group_map > 0, block_col)[:, np.newaxis, :]

    # bit 3: bad columns (ie bad flat or missed bright / dark pixel)
    # bad cols shape band, samples
    bad_cols = build_bad_col_map(obs_image)
    flags |= _bit(bad_cols > 1, BAD_COL)[np.newaxis, :, :]

    # bit 4: flag deprecated bands in l2 that were typically set to -999
    # by the mission in l2. they look normal in l1b and l0.
    if deprecated_channels is not None and len(deprecated_channels) > 0:
        flags[:, deprecated_channels, :] |= L2_DEPRECATED

    # return in original view of l1b
    return flags.transpose(1, 0, 2)


def make_mission_l1b_flag_backplane(
        obs_id: str,
        local_root: str = "data",
        verbose: bool = True,
):
    """
    From L1B data, and optionally L0 data and mission BDE files, build an
    L1B-shaped flag map with in the shape lines x bands x samples.

    Checks for necessary files in given dir (l1b data) and optional files
    (l0 data, l1b label, bde/flag map) and then uses what is available to
    make the flag map.
    """
    from l0_l1b_l2.reference import PipeManager, check_observation

    # Set up paths and check observation for issues with flag maps,
    # data availability, etc
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
        verbose=verbose,
    )

    # Check files we need exist. We can run this if only L1B exists, but not
    # without it (could modify to run and produce L0 shaped mask when no L1B?)
    paths = {
        "l0_path": moonager.l0_path,
        "l1b_path": moonager.l1b_rdn_path,
        "flag_path": moonager.flag_path,
    }
    for name, p in paths.items():
        if not p.is_file():
            if name == "l1b_path":
                print(f"{name} not found at {p}. Cannot run L1B flagging "
                      f"without the observation file. Bailing out.")
                return
            if moonager.verbose:
                if name == "l0_path":
                    print(f"{name} not found: {p}. Without L0 data we cannot "
                          f"flag negative rollover values.")
                if name == "flag_path":
                    # technically not true, we could use the dark, but then
                    # it's not 100% necessarily true to the source material
                    print(f"{name} not found: {p}. We cannot flag spectrally "
                          f"interpolated values.")
            # false if path doesn't exist
            paths[name] = False

    return flag_l1b(
        paths=paths,
        omitted_channels=moonager.omitted_channels,
        left_col_cutoff=moonager.left_col_cutoff,
        right_col_cutoff=moonager.right_col_cutoff,
        reverse_lines=moonager.reverse_lines,
        reverse_samples=moonager.reverse_samples,
        deprecated_channels=moonager.degraded_channels,
    )


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
    from l0_l1b_l2.l1b_utils.dark_obs import make_dark_signal_image
    from l0_l1b_l2.l1b_utils.mission_bde import bde_correction, \
        detector_array_tap_interpolation
    from l0_l1b_l2.l1b_utils.smooth_shape import load_ssc_factors
    from l0_l1b_l2.l1b_utils.radiometric_calibration import load_rdn_cal

    # we do load a lot of stuff twice doing this separate from the
    # main pipeline. but they are all small files. so IDK.
    dark_std = make_dark_signal_image(
        dark_path=moonager.dark_path,
        dark_method='std'
    )
    dark_std = bde_correction(
        obs_image=dark_std,
        bde_path=moonager.flag_path,
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
    ssc_factors = load_ssc_factors(moonager.ssc_path)
    dark_std = dark_std * ssc_factors[:, np.newaxis]

    return dark_std
