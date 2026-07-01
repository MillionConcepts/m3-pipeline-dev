from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Optional
from l0_l1b_l2 import CAL_DIR


def pull_metadata(obs_id: str, cal_dir: Optional[str] = None) -> dict:
    """
    Read metadata scraped from L1B RDN HDRs for the eclipse + some quality
    checks done by hand on flag maps and darks.
    """
    if cal_dir is None:
        cal_dir = CAL_DIR
    if not isinstance(cal_dir, Path):
        cal_dir = Path(cal_dir)

    metadata = pd.read_csv(cal_dir / "obs_cal_info.csv")
    metadata = metadata[metadata['obs_id'] == obs_id.upper()]

    if len(metadata) == 0:
        print("This is not a valid observation ID, although it could be a dark"
              " signal observation.")

    # sometimes there are multiple versions per obs, we want the latest one
    # ie V03 instead of V01
    if len(metadata) > 1:
        metadata = metadata.loc[
            metadata['version'].str.extract(r'(\d+)')[0].astype(int).idxmax()]
        return metadata.to_dict()
    return metadata.iloc[0].to_dict()


def check_observation(obs_id: str, local_root: Optional[str] = None):
    """
    Check that the input observation string matches the expected format & load
    info we have on it from the latest version of its L1B files
    (ie V03 not V01). Most of this info was scraped from L1B RDN HDRs in the
    JPL PDS files.

    Some files were never processed to L1B for whatever reason: they are dark
    obs and therefore should not be processed in this way, or they had issues
    (extremely hot etc.). These will return an error and empty dataframe at
    this time.

    Basic obs format: M3MYYYYMMDDTHHMMSS, ex: m3g20090720t214000

    """
    import re

    obs_warn: list[str] = []
    obs_error: list[str] = []

    # check the obs ID looks like an obs ID
    obs_id_pattern = r'^m3[gt]\d{8}t\d{6}$'
    if not bool(re.match(obs_id_pattern, obs_id, re.IGNORECASE)):
        obs_error.append(f"This is not a valid obs ID: {obs_id}")
        return obs_warn, obs_error, {}

    metadata = pull_metadata(obs_id, local_root)

    if len(metadata) == 0:
        obs_error.append(f"No metadata found for observation {obs_id}.")
        return obs_warn, obs_error

    # check that this isn't a bad BDE or dark, usually from during the broken
    # star tracker period
    if metadata['bad_dark']:
        obs_warn.append(f"This has been flagged as a bad dark obs, "
                        f"{metadata['dark_signal_id']}.")
    if metadata['bad_bde']:
        obs_warn.append(f"This has been flagged as a bad BDE map, "
                        f"{metadata['bad_detector_map_id']}.")

    # these could be warnings or errors depending on if they are used in the
    # pipeline version being run
    if not metadata['flat_in_usgs']:
        obs_warn.append(f"The obs-based flat for this obs is not in the PDS, "
                        f"{metadata['flat_field_id']}.")
    if not metadata['bde_in_usgs']:
        obs_warn.append(f"The flag map for this obs is not in the PDS, "
                        f"{metadata['bad_detector_map_id']}.")
    if not metadata['dark_in_usgs']:
        obs_warn.append(f"The dark obs for this obs is not in the PDS, "
                        f"{metadata['dark_signal_id']}.")

    # somewhat arbitrary warning at 10% flagged
    if metadata['bde_flag_coverage'] * 100 > 10:
        obs_warn.append(f"{metadata['bde_flag_coverage'] * 100}% of the "
                        f"mission-derived bad element map is flagged for this "
                        f"observation.")

    # the 3 K here is somewhat arbitrary. Lower may be more appropriate for
    # a warning, or eventually unnecessary if we use synthetic darks or do
    # away with obs-based flats.
    flat_diff = abs(metadata['obs_temperature'] - metadata['flat_field_temp'])
    if flat_diff > 3:
        obs_warn.append(f"This obs and its flat field obs have a temp "
                        f"difference greater than 3 Kelvin, "
                        f"{flat_diff} "
                        f"K.")

    dark_diff = abs(metadata['obs_temperature'] - metadata['dark_signal_temp'])
    if dark_diff > 3:
        obs_warn.append(f"This obs and its dark have a temp difference greater"
                        f" than 3 Kelvin, "
                        f"{dark_diff} "
                        f"K.")

    return obs_warn, obs_error, metadata


class PipeManager:
    # this is kind of hellishly long so idk maybe we could structure all this
    # info differently

    def __init__(self,
                 obs_id: str,
                 local_root: str,
                 metadata: dict,
                 save_steps: bool = False,
                 backplanes: bool = False,
                 verbose: bool = True,
                 ):
        # pipeline config
        self.save_steps = save_steps
        self.backplanes = backplanes
        self.verbose = verbose
        self.local_root = Path(local_root)

        # obs metadata
        self.obs_id = obs_id
        self.mode = metadata['obs_type']
        self.obs_period = metadata['obs_period']
        self.date = metadata['obs_date']
        self.time = metadata['obs_time']
        self.obs_temp = metadata['obs_temperature']
        self.obs_beta_angle = metadata['obs_beta_angle']

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

        # L0 -> L1B related calibration files
        self.dark_temp = metadata['dark_signal_temp']
        self.dark_id = metadata['dark_signal_id'].lower()
        self.flag_id = metadata['bad_detector_map_id'].lower()
        self.obs_flat_id = metadata['flat_field_id'].lower()
        self.l0_obs_path = self.local_root / f'{self.obs_id}_l0.fits'
        self.dark_path = self.local_root / f'{self.dark_id}_l0.fits'
        self.l1b_label = str(self.local_root / f'{self.obs_id}_l1b.xml')
        self.obs_flat_path = self.local_root / f'{self.obs_flat_id}_ff.fits'
        self.flag_path = self.local_root / f'{self.flag_id}_bde.fits'
        self.ssc_path = self.local_root / f'{self.obs_id}_ssc.txt'
        if self.mode.upper() == 'T':
            self.lab_flat_path = Path(CAL_DIR) / 'lab_flat_field_target.fits'
            self.rdn_gain_path = Path(CAL_DIR) / 'm3t20070912_rdn_gain.tab'
            self.rdn_spc_path = Path(CAL_DIR) / 'm3t20070912_rdn_spc.tab'
            self.rdn_cal_path = Path(CAL_DIR) / 'm3t20081118_rdn_cal.tab'
        elif self.mode.upper() == 'G':
            self.lab_flat_path = Path(CAL_DIR) / 'lab_flat_field_global.fits'
            self.rdn_gain_path = Path(CAL_DIR) / 'm3g20081211_rdn_gain.tab'
            self.rdn_spc_path = Path(CAL_DIR) / 'm3g20081211_rdn_spc.tab'
            self.rdn_cal_path = Path(CAL_DIR) / 'm3g20081118_rdn_cal.tab'

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
            self.left_col_cutoff = 17  # inclusive
            self.right_col_cutoff = 625  # exclusive
            # read out / tap cols are 1, 161, 321, 481 in DPSIS
            self.read_out_cols = [0, 160, 320, 480]
            # filter channels are 41, 42, 116 in DPSIS
            self.filter_seam_rows = [40, 41, 39, 115, 116, 114]
            # for L2
            self.degraded_channels = [0, 1, 2, 3, 4, 5, 6, 7]

        elif self.mode.upper() == 'G':
            self.l0_samples = 320
            self.l0_channels = 86
            self.l1b_samples = 304
            self.l1b_channels = 85
            self.omitted_channels = [0]
            self.dark_cols = [1, 2, 3, 318, 319]
            # the vignetted cols for global are not listed explicitly
            # in the DPSIS
            self.vignetted_cols_left = [4, 5, 6]  # cut 8 and 7
            self.vignetted_cols_right = [314, 315, 316, 317]  # 313 cut?
            self.vignetted_cols = [4, 5, 6, 314, 315, 316, 317]
            # 304 samples after trimming cols
            self.left_col_cutoff = 9  # inclusive
            self.right_col_cutoff = 313  # exclusive
            self.read_out_cols = [0, 80, 160, 240]
            # 13 and 50 in DPSIS
            self.filter_seam_rows = [12, 49]
            # for L2
            self.degraded_channels = [0, 1]

        # L1B -> L2 related calibration files
        # some of these are both mode and time / date dependant
        self.l1b_rdn_path = self.local_root / f'{self.obs_id}_l1b_rdn.fits'
        self.l1b_obs_path = self.local_root / f'{self.obs_id}_l1b_obs.fits'
        self.l1b_tim_path = self.local_root / f'{self.obs_id}_l1b_tim.tab'
        self.l1b_loc_path = self.local_root / f'{self.obs_id}_loc.fits'

        if self.mode.upper() == 'T':
            self.sol_spec_path = Path(
                CAL_DIR) / 'm3t20110224_rfl_solar_spec.tab'
            self.old_f_alpha_hil_path = Path(
                CAL_DIR) / 'm3t20111109_rfl_f_alpha_hil.tab'
            self.new_f_alpha_hil_path = Path(
                CAL_DIR) / 'm3t20120120_rfl_f_alpha_hil.tab'
            if self.temp_des == 'warm':
                self.grnd_truth_path = Path(
                    CAL_DIR) / 'm3t20111117_rfl_grnd_tru_2.tab'
                self.stat_pol_path = Path(
                    CAL_DIR) / 'm3t20111020_rfl_stat_pol_2.tab'
            elif self.temp_des == 'cold':
                self.grnd_truth_path = Path(
                    CAL_DIR) / 'm3t20111117_rfl_grnd_tru_1.tab'
                self.stat_pol_path = Path(
                    CAL_DIR) / 'm3t20111020_rfl_stat_pol_1.tab'
        elif self.mode.upper() == 'G':
            self.sol_spec_path = Path(
                CAL_DIR) / 'm3g20110224_rfl_solar_spec.tab'
            self.old_f_alpha_hil_path = Path(
                CAL_DIR) / 'm3g20111109_rfl_f_alpha_hil.tab'
            self.new_f_alpha_hil_path = Path(
                CAL_DIR) / 'm3g20120120_rfl_f_alpha_hil.tab'
            if self.temp_des == 'warm':
                self.grnd_truth_path = Path(
                    CAL_DIR) / 'm3g20111117_rfl_grnd_tru_2.tab'
                self.stat_pol_path = Path(
                    CAL_DIR) / 'm3g20110830_rfl_stat_pol_2.tab'
            elif self.temp_des == 'cold':
                self.grnd_truth_path = Path(
                    CAL_DIR) / 'm3g20111117_rfl_grnd_tru_1.tab'
                self.stat_pol_path = Path(
                    CAL_DIR) / 'm3g20110830_rfl_stat_pol_1.tab'


def check_l1b_label(l1b_path: str):
    """
    Read L1B label for yaw / limb direction to undo until we have our own
    orientation info.
    """
    import pds4_tools

    try:
        label = pds4_tools.pds4_read(
            l1b_path, lazy_load=True, quiet=True
        ).label
    except:
        # if we don't have the label
        print("Original L1B label missing, needed for orientation info.")
        return False, False

    params = label.to_dict()['Product_Observational']['Observation_Area'][
        'Mission_Area']['chan1:Chandrayaan-1_Parameters']

    yaw = params['chan1:spacecraft_yaw_direction'].lower()
    limb = params['chan1:orbit_limb_direction'].lower()
    print(yaw)
    print(limb)
    reverse_lines = (limb == "ascending")
    reverse_samples = (limb == "descending" and yaw == "reverse") or \
                      (limb == "ascending" and yaw == "forward")

    return reverse_lines, reverse_samples
