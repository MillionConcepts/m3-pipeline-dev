from datetime import datetime
import pandas as pd
from pathlib import Path
import re
from typing import Optional
from . import CAL_DIR

L1B_CAL_DIR = Path(CAL_DIR) / "l1b"
L2_CAL_DIR = Path(CAL_DIR) / "l1b"

OBS_INFO_FILE = "l1b_processing_info.csv"
OBS_ID_PATTERN = re.compile(r'^m3[gt]\d{8}t\d{6}$', re.IGNORECASE)

# temperature difference between dark and obs to give a warning at
MAX_TEMP_DIFF_K = 3


def pull_metadata(obs_id: str, cal_dir: Optional[str] = None) \
        -> Optional[dict]:
    """
    Read metadata scraped from L1B RDN HDRs for the eclipse + some quality
    checks done by hand on flag maps and darks.

    The flat field temp is looked up from the flat obs row and added as
    'flat_temp'. NaN if the flat obs isn't present (unlikely).

    Returns None if the observation isn't in the table (ie likely a dark or
    never processed to L1B due to temperature).
    """
    cal_dir = Path(CAL_DIR if cal_dir is None else cal_dir)
    info = pd.read_csv(cal_dir / OBS_INFO_FILE)
    id_cols = ['obs_id', 'bde_id', 'flat_id', 'dark_id', 'closest_dark',
               'replacement_dark_id']

    # should all be lowercase but make sure
    info[id_cols] = info[id_cols].apply(lambda col: col.str.lower())

    matches = info[info['obs_id'] == obs_id.lower()]

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        # The table currently has one row per obs; flag it if that changes.
        print(f"Warning: {len(matches)} rows found for {obs_id}, "
              f"using the first.")

    metadata = matches.iloc[0].to_dict()

    # get temp of obs used for flat field
    flat_row = info.loc[info['obs_id'] == metadata['flat_id'],
                        'obs_temperature']
    metadata['flat_temp'] = (flat_row.iloc[0] if len(flat_row)
                             else float('nan'))

    return metadata


def check_observation(obs_id: str, local_root: Optional[str] = None):
    """
    Check that the input observation string matches the expected format & load
    info we have on it from the latest version of its L1B files
    (ie V03 not V01). Most of this info was scraped from L1B RDN HDRs in the
    JPL PDS files as well as L1B V03 LBL files.

    Some files were never processed to L1B for whatever reason: they are dark
    obs and therefore should not be processed in this way, or they had issues
    (extremely hot etc.). These will return an error and empty dict.

    Basic obs format: M3MYYYYMMDDTHHMMSS, ex: m3g20090720t214000

    Returns (warnings, errors, metadata).
    """
    obs_warn: list[str] = []
    obs_error: list[str] = []

    # check the obs ID looks like an obs ID
    if not OBS_ID_PATTERN.match(obs_id):
        obs_error.append(f"This is not a valid obs ID: {obs_id}")
        return obs_warn, obs_error, {}

    metadata = pull_metadata(obs_id, local_root)

    if metadata is None:
        obs_error.append(f"No info found for observation {obs_id}. "
                         f"It may be a dark signal observation or a real "
                         f"observation that was never processed to L1B.")
        return obs_warn, obs_error, {}

    # check that this isn't a bad BDE / dark / flat, usually from during the
    # broken star tracker period
    if metadata['bad_dark']:
        obs_warn.append(f"This has been flagged as a bad dark obs, "
                        f"{metadata['dark_id']}. Replacing with "
                        f"{metadata['replacement_dark_id']}.")
        metadata['dark_id'] = metadata['replacement_dark_id']
    if metadata['bad_flat']:
        obs_warn.append(f"This has been flagged as a bad flat field obs, "
                        f"{metadata['flat_id']}.")
    if metadata['bad_bde']:
        obs_warn.append(f"This has been flagged as a bad BDE map, "
                        f"{metadata['bde_id']}. Replacing with "
                        f"{metadata['replacement_dark_id']}.")
        metadata['bde_id'] = metadata['replacement_dark_id']
    # if the dark used for L1B processing isn't the closest, warn the user
    if not metadata['dark_matches_closest']:
        obs_warn.append(f"The dark used for this obs, {metadata['dark_id']}, "
                        f"is not the closest dark temporally, "
                        f"{metadata['closest_dark']} "
                        f"({metadata['closest_dark_gap_min']:.1f} min away, "
                        f" and {metadata['closest_dark_temp']:.2f} K).")

    # temperature differences between the obs and its flat and dark
    # this temp is from the L1B V03 LBLs. Other temps are recorded in various
    # HDRS / L0 LBLS, but we go with the latest temp value here.
    obs_temp = metadata['obs_temperature']

    if pd.isna(metadata['flat_temp']):
        # as far as I know I don't think this should ever occur
        obs_warn.append(f"No temperature found for the flat field obs, "
                        f"{metadata['flat_id']}.")
    else:
        flat_diff = abs(obs_temp - metadata['flat_temp'])
        if flat_diff > MAX_TEMP_DIFF_K:
            obs_warn.append(f"This obs and its flat field obs have a temp "
                            f"difference greater than {MAX_TEMP_DIFF_K} "
                            f"Kelvin, {flat_diff:.2f} K.")

    if pd.isna(metadata['dark_temp']):
        # this does occur
        obs_warn.append(f"No temperature recorded for the dark obs, "
                        f"{metadata['dark_id']}.")
    else:
        dark_diff = abs(obs_temp - metadata['dark_temp'])
        if dark_diff > MAX_TEMP_DIFF_K:
            obs_warn.append(f"This obs and its dark have a temp difference "
                            f"greater than {MAX_TEMP_DIFF_K} Kelvin, "
                            f"{dark_diff:.2f} K.")

    return obs_warn, obs_error, metadata


class PipeManager:

    """
    Information about an observation necessary to process it, including
    cal file paths, image orientation and size, as well as flagging specs.
    """

    def __init__(self,
                 obs_id: str,
                 local_root: str,
                 metadata: dict,
                 save_steps: bool = False,
                 backplanes: bool = False,
                 smooth_shape: bool = False,
                 verbose: bool = True,
                 ):
        # pipeline config
        self.save_steps = save_steps
        self.backplanes = backplanes
        self.verbose = verbose
        self.smooth_shape = smooth_shape
        self.local_root = Path(local_root)

        # for intermediate image cropping (if you want to quality check
        # only a section etc).
        # TODO: could offer this as an option for the whole pipeline tbh
        self.line_start = 100
        self.line_stop = 200

        # obs metadata
        self.obs_id = obs_id
        self.mode = metadata['obs_type']
        # self.obs_period = metadata['obs_period']
        self.date = metadata['date']
        self.time = metadata['time']
        self.obs_temp = metadata['obs_temperature']
        self.obs_beta_angle = metadata['beta_angle']

        # obs l1b image size and orientation
        self.lines = metadata['lines']
        self.limb_dir = metadata['limb_dir']
        self.yaw_dir = metadata['yaw_dir']
        self.reverse_lines = (self.limb_dir == "ascending")
        self.reverse_samples = \
            (self.limb_dir == "descending" and self.yaw_dir == "reverse") or \
            (self.limb_dir == "ascending" and self.yaw_dir == "forward")

        # temperature as hot or cold based on date, this designation
        # comes from the L2 section of the DPSIS
        self.date_time = datetime.fromisoformat(f"{self.date}T{self.time}")
        self.temp_des = (
            "cold" if any(
                datetime.fromisoformat(
                    s) <= self.date_time < datetime.fromisoformat(e)
                for s, e in [
                    ("2009-01-19T00:00:00", "2009-02-15T00:00:00"),
                    ("2009-04-15T00:00:00", "2009-04-28T00:00:00"),
                    ("2009-07-12T00:00:00", "2009-08-17T00:00:00"),
                ]
            ) else "warm" if any(
                datetime.fromisoformat(
                    s) <= self.date_time < datetime.fromisoformat(e)
                for s, e in [
                    ("2008-11-18T00:00:00", "2009-01-19T00:00:00"),
                    ("2009-05-13T00:00:00", "2009-05-17T00:00:00"),
                    ("2009-05-20T00:00:00", "2009-07-10T00:00:00"),
                ]
            ) else None
        )

        # L0 -> L1B related calibration file IDs
        self.dark_id = metadata['dark_id'].lower()
        self.dark_temp = metadata['dark_temp']
        self.closest_dark_id = metadata['closest_dark'].lower()
        self.closest_dark_temp = metadata['closest_dark_temp']
        self.flag_id = metadata['bde_id'].lower()
        self.obs_flat_id = metadata['flat_id'].lower()

        # Cal file paths
        self.l0_path = self.local_root / f'{self.obs_id}_l0.fits'
        self.dark_path = self.local_root / f'{self.dark_id}_l0.fits'
        self.l1b_label = self.local_root / f'{self.obs_id}_l1b.xml'
        self.obs_flat_path = self.local_root / f'{self.obs_flat_id}_ff.fits'
        self.flag_path = self.local_root / f'{self.flag_id}_bde.fits'
        self.ssc_path = self.local_root / f'{self.obs_id}_ssc.txt'
        if self.mode.upper() == 'T':
            self.lab_flat_path = Path(L1B_CAL_DIR) / 'lab_flat_field_target.fits'
            self.rdn_gain_path = Path(L1B_CAL_DIR) / 'm3t20070912_rdn_gain.tab'
            self.rdn_spc_path = Path(L1B_CAL_DIR) / 'm3t20070912_rdn_spc.tab'
            self.rdn_cal_path = Path(L1B_CAL_DIR) / 'm3t20081118_rdn_cal.tab'
            self.additional_flag_path = \
                Path(L1B_CAL_DIR) / 'flag_map_target_combo.fits'
            self.ssc_table_path = Path(L1B_CAL_DIR) / 'target_ssc.csv'
        elif self.mode.upper() == 'G':
            self.lab_flat_path = Path(L1B_CAL_DIR) / 'lab_flat_field_global.fits'
            self.rdn_gain_path = Path(L1B_CAL_DIR) / 'm3g20081211_rdn_gain.tab'
            self.rdn_spc_path = Path(L1B_CAL_DIR) / 'm3g20081211_rdn_spc.tab'
            self.rdn_cal_path = Path(L1B_CAL_DIR) / 'm3g20081118_rdn_cal.tab'
            self.additional_flag_path = \
                Path(L1B_CAL_DIR) / 'flag_map_global_combo.fits'
            self.ssc_table_path = Path(L1B_CAL_DIR) / 'global_ssc.csv'

        # other L0 -> L1B calibration things
        self.ghost_corr_factor = .0048

        # specific channel and samples for target and global
        # mostly for L0 -> L1B but some L1B -> L2
        # TODO: looking at the flats for target makes it seem like they
        #  interpolated across more rows than indicated in the DPSIS, so
        #  this is something we should investigate more thoroughly and
        #  possibly expand the list of rows here
        if self.mode.upper() == 'T':
            self.l0_samples = 640
            self.l0_channels = 260
            self.l1b_samples = 608
            self.l1b_channels = 256
            self.omitted_channels = [0, 1, 2, 3]
            # you might argue 0 belongs in dark cols, but it is also a readout
            # column so behaves strangely
            self.dark_cols = [1, 2, 3, 4, 5, 6, 7, 636, 637, 638, 639]
            # 9 - 15 in DPSIS
            self.vignetted_cols_left = [8, 9, 10, 11, 12, 13, 14]
            # 628 - 636 in DPSIS
            self.vignetted_cols_right = [627, 628, 629, 630, 631,
                                         632, 633, 634, 635]
            # 608 samples after trimming cols
            self.left_col_cutoff = 18  # inclusive
            self.right_col_cutoff = 624  # exclusive
            # read out / tap cols are 1, 161, 321, 481 in DPSIS
            self.read_out_cols = [0, 160, 320, 480]
            # filter channels are 41, 42, 116 in DPSIS
            self.filter_seam_rows = [40, 41, 39, 115, 116, 114]
            # for L2
            self.degraded_channels = [0, 1, 2, 3, 4, 5, 6, 7, 255]
            # bad /variable column groups (1 indexed)
            self.bad_column_groups = {
                'group1': [74, 138, 170, 234, 298, 330, 394, 458, 490, 554,
                           618],
                'group2': [42, 106, 202, 266, 362, 426, 522, 586],
            }

        elif self.mode.upper() == 'G':
            self.l0_samples = 320
            self.l0_channels = 86
            self.l1b_samples = 304
            self.l1b_channels = 85
            self.omitted_channels = [0]
            self.dark_cols = [1, 2, 318, 319]
            # the vignetted cols for global are not listed explicitly
            # in the DPSIS
            self.vignetted_cols_left = [5, 6, 7]  # cut 8 and 7
            self.vignetted_cols_right = [314, 315, 316]  # 313 cut?
            self.vignetted_cols = [5, 6, 7, 314, 315, 316]
            # 304 samples after trimming cols
            self.left_col_cutoff = 9  # inclusive
            self.right_col_cutoff = 313  # exclusive
            self.read_out_cols = [0, 80, 160, 240]
            # 13 and 50 in DPSIS
            self.filter_seam_rows = [12, 49]
            # for L2
            self.degraded_channels = [0, 1]
            # bad /variable column groups (1 indexed)
            self.bad_column_groups = {
                'group1': [310, 278, 246, 230, 198, 166, 150, 118, 86, 70, 38,
                           6],
                'group2': [294, 262, 214, 182, 134, 102, 54, 22],
            }

        # L1B -> L2 related calibration files
        # some of these are both mode and time / date dependant
        self.l1b_rdn_path = self.local_root / f'{self.obs_id}_l1b_rdn.fits'
        self.l1b_obs_path = self.local_root / f'{self.obs_id}_l1b_obs.fits'
        self.l1b_tim_path = self.local_root / f'{self.obs_id}_l1b_tim.tab'
        self.l1b_loc_path = self.local_root / f'{self.obs_id}_loc.fits'

        if self.mode.upper() == 'T':
            self.sol_spec_path = Path(
                L2_CAL_DIR) / 'm3t20110224_rfl_solar_spec.tab'
            self.old_f_alpha_hil_path = Path(
                L2_CAL_DIR) / 'm3t20111109_rfl_f_alpha_hil.tab'
            self.new_f_alpha_hil_path = Path(
                L2_CAL_DIR) / 'm3t20120120_rfl_f_alpha_hil.tab'
            if self.temp_des == 'warm':
                self.grnd_truth_path = Path(
                    L2_CAL_DIR) / 'm3t20111117_rfl_grnd_tru_2.tab'
                self.stat_pol_path = Path(
                    L2_CAL_DIR) / 'm3t20111020_rfl_stat_pol_2.tab'
            elif self.temp_des == 'cold':
                self.grnd_truth_path = Path(
                    L2_CAL_DIR) / 'm3t20111117_rfl_grnd_tru_1.tab'
                self.stat_pol_path = Path(
                    L2_CAL_DIR) / 'm3t20111020_rfl_stat_pol_1.tab'
        elif self.mode.upper() == 'G':
            self.sol_spec_path = Path(
                L2_CAL_DIR) / 'm3g20110224_rfl_solar_spec.tab'
            self.old_f_alpha_hil_path = Path(
                L2_CAL_DIR) / 'm3g20111109_rfl_f_alpha_hil.tab'
            self.new_f_alpha_hil_path = Path(
                L2_CAL_DIR) / 'm3g20120120_rfl_f_alpha_hil.tab'
            if self.temp_des == 'warm':
                self.grnd_truth_path = Path(
                    L2_CAL_DIR) / 'm3g20111117_rfl_grnd_tru_2.tab'
                self.stat_pol_path = Path(
                    L2_CAL_DIR) / 'm3g20110830_rfl_stat_pol_2.tab'
            elif self.temp_des == 'cold':
                self.grnd_truth_path = Path(
                    L2_CAL_DIR) / 'm3g20111117_rfl_grnd_tru_1.tab'
                self.stat_pol_path = Path(
                    L2_CAL_DIR) / 'm3g20110830_rfl_stat_pol_1.tab'


class MismatchShapeError(ValueError):
    pass


def check_shape(shape, ref, cal_name="cal arr"):
    """
    Check that cal file and obs image array axis match in shape / orientation.
    Raise MismatchShapeError if not.

    Can pass arrays themselves or their shapes.
    """
    shape = tuple(getattr(shape, "shape", shape))
    ref = tuple(getattr(ref, "shape", ref))
    for s, label in ((shape, "obs_image"), (ref, cal_name)):
        if len(s) not in (2, 3):
            raise MismatchShapeError(f"For {label}: expected 2D or 3D, "
                                     f"got {len(s)}D {s}")
    if shape[-2:] != ref[-2:]:
        msg = (f"During {cal_name}: obs image axes {shape[-2:]} "
               f"don't match cal array {ref[-2:]} "
               f"(full shapes {shape} vs {ref})")
        if shape[-2:] == ref[-2:][::-1]:
            msg += " — it looks transposed, check the axes orientation"
        raise MismatchShapeError(msg)
