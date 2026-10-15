

def pull_metadata(obs_id: str, cal_dir: Optional[str] = None) -> dict:
    """
    DEPRECATED because this info is now pulled from l1b_processing_info.csv,
    instead of obs_cal_info.csv.

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
        return None
    # sometimes there are multiple versions per obs, we want the latest one
    # ie V03 instead of V01
    elif len(metadata) > 1:
        best_idx = metadata['version'].str.extract(r'(\d+)')[0].astype(
            int).idxmax()
        metadata = metadata.loc[[best_idx]]

    return metadata.iloc[0].to_dict()


def check_observation(obs_id: str, local_root: Optional[str] = None):
    """
    DEPRECATED because this info is now pulled from l1b_processing_info.csv,
    instead of obs_cal_info.csv.

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


def check_l1b_label(l1b_path: Path):
    """
    DEPRECATED because this info is now all in l1b_processing_info.csv.
    So no need to have each label downloaded.

    Read L1B label for yaw / limb direction to undo until we have our own
    orientation info.
    """
    import pds4_tools

    try:
        label = pds4_tools.pds4_read(
            str(l1b_path), lazy_load=True, quiet=True
        ).label
    except Exception as e:
        print(f"Issue opening original L1B label, information inside needed "
              f"for orientation info: {e}")
        return False, False

    params = label.to_dict()['Product_Observational']['Observation_Area'][
        'Mission_Area']['chan1:Chandrayaan-1_Parameters']

    yaw = params['chan1:spacecraft_yaw_direction'].lower()
    limb = params['chan1:orbit_limb_direction'].lower()
    reverse_lines = (limb == "ascending")
    reverse_samples = (limb == "descending" and yaw == "reverse") or \
                      (limb == "ascending" and yaw == "forward")

    return reverse_lines, reverse_samples


