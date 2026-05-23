# M3 Calibration Pipelines for L1B to L2 processing

from l0_l1b_l2.reference import PipeManager


def run_l2_mission_pipeline(moonager: PipeManager):
    """
    Pipeline for L1B (radiance) to L2 (reflectance) based on an originalist
    reading of the DPSIS. This version of the pipeline uses multiple products
    from L1B level processing, including topo data etc. You could (at some
    point) use these products from our version of the L1B pipeline, or use the
    L1B products stored by the USGS PDS node (fits files).
    """
    from l0_l1b_l2.l1b_utils.loader import load_fits_into_frame
    from l0_l1b_l2.l2_utils.radiance_factor import calculate_radiance_factor
    from l0_l1b_l2.l2_utils.statistical_polishing import apply_stat_polishing
    from l0_l1b_l2.l2_utils.photometric_corr import photometric_correction
    from l0_l1b_l2.l2_utils.ground_truth import ground_truth_correction

    image = load_fits_into_frame(moonager.l1b_rdn_path)

    # (1) I/F Correction
    # temporarily using 1 AU for sun distance
    if moonager.verbose:
        print("Converting radiance to radiance factor (I/F).")
    image = calculate_radiance_factor(
        image=image,
        sol_spec_path=moonager.sol_spec_path
    )

    # (2) Statistical Polishing
    if moonager.verbose:
        print("Applying statistical polishing factors.")
    image = apply_stat_polishing(
        image=image,
        stat_pol_path=moonager.stat_pol_path
    )

    # (3) Iterative Thermal Removal
    # TODO: not done at all

    # if moonager.verbose:
    #     print("Running iterative thermal removal.")

    # (4) Photometric Correction
    # TODO: not done yet
    if moonager.verbose:
        print("Running photometric correction of entire cube.")
    image = photometric_correction(
        image=image,
        f_alpha_path=moonager.new_f_alpha_hil_path,
        obs_path=moonager.l1b_obs_path
    )

    # (4b) Photometric Correction of only 1489 nm relative to a sphere
    # not implemented yet, this outputs an extra image

    # (5) Ground Truth Correction (Optional)
    # Option to turn this on or off? Not currently applied to L2 PDS products.
    if moonager.verbose:
        print("Applying ground truth correction.")
    image = ground_truth_correction(
        image=image,
        ground_truth_path=moonager.grnd_truth_path
    )

    # (6) Flag Degraded Channels
    if moonager.verbose:
        print("Flagging degraded channels.")
    image[:, moonager.degraded_channels, :] = -999.0

    return image
