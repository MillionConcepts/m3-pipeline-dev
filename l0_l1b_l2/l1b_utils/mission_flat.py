from astropy.io import fits
from pathlib import Path
import numpy as np
from typing import Optional
from l0_l1b_l2.reference import check_shape


def load_flats(flat_path: Path):
    """
    Read in flat files.
    """
    with fits.open(flat_path) as hdul:
        flat = hdul[0].data

    return flat


def apply_flat(
        obs_image: np.ndarray,
        flat_path: Path,
        flag_path: Optional[Path] = None
):
    """
    Load and multiply or divide flat by image. Modify flat for flagged elements
    in the BDE if flag_path is given.
    """
    if flag_path is None:
        flat = load_flats(flat_path)
    else:
        flat = fix_flagged_in_lab_flat(
            flat_path,
            flag_path
        )

    check_shape(obs_image.shape, flat.shape, 'flat fielding')

    if "lab" in flat_path.name:
        return obs_image * flat[np.newaxis, :, :]
    else:
        flat_masked = flat.copy()
        flat_masked[flat_masked == 0] = 1
        return obs_image / flat_masked[np.newaxis, :, :]


def get_shadowing_from_lab_flat(
        flat_path: Path,
        left_cutoff: int,
        right_cutoff: int
):
    """
    Get the shadowing from the lab flat, return as 2d surface.

    The idea is to divide this out of an image before making a new flat.
    That way we don't accidentally keep the shadowing in the flat
    in an effort to remove observational illumination/shadowing.
    """
    flat = load_flats(flat_path)

    flat_surface = fit_surface(
        flat[:, left_cutoff:right_cutoff],
        band_degree=3,
        sample_degree=3,
        n_iterations=2,
    )

    return flat_surface


def fix_flagged_in_lab_flat(flat_path: Path, bde_path: Path):
    """
    Use mission interpolation method for BDE (flag) pixels on the DSS image to
    apply the same interpolation to the lab flat. The idea is that the lab
    flat doesn't work for pixels that are interpolated (they weren't really
    'measured' at that pixel).

    Using this does mean the mission observation-derived flats probably
    no longer work because those used the original lab flat, which I don't
    think they modified in this way.
    """
    from .mission_bde import bde_correction

    flat = load_flats(flat_path)

    mod_flat = bde_correction(flat, bde_path)

    return mod_flat


def average_over_lines(paths):
    """
    Average one (or multiple) observations across lines, keeping sample / band
    structure.
    """
    image_sum = None

    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            # TODO: update this depending on how we decide to save outputs?
            # or just add a check on the shape. ideally our new fits files
            # indicate line / band axes etc
            cube = hdul[0].data
            n_bands, n_lines, n_samples = cube.shape
            if image_sum is None:
                image_sum = np.zeros((n_bands, n_samples))
            elif image_sum.shape != (n_bands, n_samples):
                raise ValueError(f"{path} has a different shape :(")
            for band in range(n_bands):
                image_sum[band] += np.sum(cube[band], axis=0)
    return image_sum / n_lines


def normalize_to_center(flat, n_center=40):
    """
    Divide each band by central 40 samples (could change #).
    """
    n_samples = flat.shape[1]
    start = n_samples // 2 - n_center // 2
    center_mean = np.nanmean(flat[:, start:start + n_center], axis=1)
    flat = flat / center_mean[:, np.newaxis]
    return flat.astype(np.float32)


def fit_surface(flat,
                band_degree=3,
                sample_degree=4,
                n_iterations=5,
                clip=3.0
                ):
    """
    Fit 2-D polynomial surface to the flat.
    TODO: explore fitting just across band instead of 2d? could destroy ripples
    """
    from numpy.polynomial import legendre

    # not passing moonager band / sample counts bc I think sometimes it will be
    # useful to make on L1B dims or L0 dims
    n_bands, n_samples = flat.shape

    band_coords, sample_coords = np.meshgrid(np.linspace(-1, 1, n_bands),
                                             np.linspace(-1, 1, n_samples),
                                             indexing="ij"  # keep dims / shape
                                             )
    design = legendre.legvander2d(band_coords.ravel(),
                                  sample_coords.ravel(),
                                  [band_degree, sample_degree]
                                  )
    values = flat.ravel()
    # possible to have inf / NaN if other flats were used first
    # or because of masking bad pixels
    good = np.isfinite(values)

    for _ in range(n_iterations):
        # select solution of least squares
        coeffs = np.linalg.lstsq(design[good], values[good], rcond=None)[0]
        # fit model to whole shape of field
        model = design @ coeffs
        # compare model with flat
        residual = values - model

        # don't keep going if nothing changes / if it's a perfect fit (if flat
        # is all 0 that might happen etc)
        limit = clip * np.nanstd(residual[good])
        new_good = np.isfinite(values) & (np.abs(residual) <= limit)
        if np.isclose(limit, 0) or np.array_equal(new_good, good):
            break
        good = new_good
    return model.reshape(n_bands, n_samples)


def make_flat_field_from_paths(paths, n_center=40) -> np.ndarray:
    """
    Make an image-based flat field as described in Green 2011.

    Rough steps from Green 2011:
        1) averaging the longest on orbit data
        sets and then dividing by the average of the central 40 cross-track
        sample values.
        2) a two‐dimensional surface is fit to the image based flat field and
         removed from the flat field correction factor.
            * don't think factor is supposed to be single value,
                but who knows...
        3) to suppress the impact of major features in the image‐based flat
        field on the resulting illuminated lunar surface images, a smoothed
        spectral average is divided out in a final flat field.

    This function can take in multiple observations to use for making flats,
    instead of just one as described by Green.
    """
    if isinstance(paths, str):
        paths = [paths]

    # 1) average data & normalize
    line_average = average_over_lines(paths)
    flat = normalize_to_center(line_average, n_center)

    # 2) 2d surface removal
    flat = flat / fit_surface(flat)

    # 3) divide out spectral average
    flat = flat / np.nanmedian(flat, axis=0, keepdims=True)

    # repeat 1b
    return normalize_to_center(flat, n_center)


def make_flat_field_from_obs(
        obs_image: np.ndarray,
        left_col_cutoff: int,
        right_col_cutoff: int,
        n_center=40,
) -> np.ndarray:
    """
    Make an image-based flat field as described in Green 2011.

    Rough steps from Green 2011:
        1) averaging the longest on orbit data
        sets and then dividing by the average of the central 40 cross-track
        sample values.
        2) a two‐dimensional surface is fit to the image based flat field and
         removed from the flat field correction factor.
            * don't think factor is supposed to be single value,
                but who knows...
        3) to suppress the impact of major features in the image‐based flat
        field on the resulting illuminated lunar surface images, a smoothed
        spectral average is divided out in a final flat field.

    This function takes in one observation.
    """

    # 1) average data & normalize
    # obs image has shape band, line, sample
    # TODO: consider using only beginning or end of warm, long obs?
    # TODO: fit 2d plane to first lab flat and divide that out of the obs prior
    #    to this version of flatting
    line_average = np.nanmedian(
        obs_image[:, :, left_col_cutoff:right_col_cutoff],
        axis=0
    )
    flat = normalize_to_center(line_average, n_center)

    # 2) 2d surface removal
    flat = flat / fit_surface(flat)

    # 3) divide out spectral average
    flat = flat / np.nanmedian(flat, axis=0, keepdims=True)

    # repeat 1b
    return normalize_to_center(flat, n_center)


def get_masked_flat_values(
        obs_image: np.ndarray,
        flat: np.ndarray,
        mask: np.ndarray,
        left_col_cutoff: int,
        right_col_cutoff: int,
) -> np.ndarray:
    """
    Fill flat values for masked pixels using ratio of existing value to the
    predicted value from interpolating spatially with adjacent cols in the
    same band. Evaluated across all lines of observation to get one flat value.
    """
    # TODO: there are a lot of places where we interpolate spatially or
    #   spectrally (BDE, filter seam, tap, here, etc) with masked/bad pixels.
    #   so would be a good idea to break down the below logic & bde logic into
    #   shared helper functions instead of rewriting in various places.
    flat = flat.copy()
    # go band by band and get neighboring good columns for bands with flagged
    # pixels
    neighbors_by_band = {}
    for band in np.flatnonzero(mask.any(axis=1)):
        good_columns = np.flatnonzero(~mask[band] & np.isfinite(flat[band]))
        flagged_entries = []

        for flagged_column in np.flatnonzero(mask[band]):
            # location of flagged pixel in cols tells us if there are left and
            # right cols to interpolate with
            insert_position = np.searchsorted(good_columns, flagged_column)
            has_left = insert_position > 0
            has_right = insert_position < len(good_columns)

            # linearly interpolate if un-flagged cols on either side
            if has_left and has_right:
                left_column = good_columns[insert_position - 1]
                right_column = good_columns[insert_position]
                right_weight = (
                        (flagged_column - left_column) /
                        (right_column - left_column)
                )
                neighbor_weights = [(left_column, 1.0 - right_weight),
                                    (right_column, right_weight)]
            elif has_left:
                neighbor_weights = [(good_columns[insert_position - 1], 1.0)]
            elif has_right:
                neighbor_weights = [(good_columns[insert_position], 1.0)]
            else:
                print(f"Band {band} has no good pixels for flat building.")
                continue
            flagged_entries.append((flagged_column, neighbor_weights))
        neighbors_by_band[band] = flagged_entries

    # divide the interpolated but not flatted pixel value for every flag by
    # the value predicted by interpolating its flatted neighbors
    ratios_by_pixel = {
        (band, flagged_column): []
        for band, flagged_entries in neighbors_by_band.items()
        for flagged_column, _ in flagged_entries
    }

    # ignore divide by zeros (although unlikely except in super dark areas)
    with np.errstate(divide="ignore", invalid="ignore"):
        for band, flagged_entries in neighbors_by_band.items():
            band_image = obs_image[:, band, left_col_cutoff:right_col_cutoff]
            for flagged_column, neighbor_weights in flagged_entries:
                raw_flagged_values = band_image[:, flagged_column]
                # we want to flat the neighbor with the flat we have
                # thus far
                predicted_values = sum(
                        weight
                        * band_image[:, neighbor_column]
                        / flat[band, neighbor_column]
                        for neighbor_column, weight in neighbor_weights
                    )
                # ratio of raw flagged pixel to interpolated pixel value
                # for all lines
                ratios_by_pixel[(band, flagged_column)].append(
                        raw_flagged_values / predicted_values)
    # set to 1 initially
    flat[mask] = 1.0
    for (band, flagged_column), ratio_list in ratios_by_pixel.items():
        all_ratios = np.concatenate(ratio_list)
        finite_ratios = all_ratios[np.isfinite(all_ratios)]
        if finite_ratios.size:
            # use median value as the new flat weight for that flagged pixel.
            # I could see a world where mean may be better?
            # or perhaps these values could change with portion of the
            # observation we're in? I've noticed a value might be good for a
            # bright portion but bad for a dark portion etc
            flat[band, flagged_column] = np.median(finite_ratios)
    return flat


def make_flat_field_from_obs_with_masking(
        obs_image: np.ndarray,
        left_col_cutoff: int,
        right_col_cutoff: int,
        flag_path: Path,
        ignored_flag_cols: list[int],
        ignored_flag_rows: list[int],
        n_center=40,
) -> np.ndarray:
    """
    Make an image-based flat field loosely based on Green 2011.

    Rough steps from Green 2011:
        1) averaging the longest on orbit data
        sets and then dividing by the average of the central 40 cross-track
        sample values.
        2) a two‐dimensional surface is fit to the image based flat field and
         removed from the flat field correction factor.
            * don't think factor is supposed to be single value,
                but who knows...
        3) to suppress the impact of major features in the image‐based flat
        field on the resulting illuminated lunar surface images, a smoothed
        spectral average is divided out in a final flat field.

    This function takes in one observation.

    Ideally run after removing the 2d surface pulled from the lab flat with
    edge shadowing etc.

    We also apply the BDE / flag mask so that flagged & interpolated samples
    are not included in the first masking pass, to avoid heavily masked areas
    skewing the results. Then, the masked pixels have their own flat values
    derived based on ratios with neighboring, unmasked, flatted band values.
    Works less well for bands heavily masked. For things like the filter seams
    it is more fitting to just treat like a normal band since the whole thing
    is interpolated.
    """
    from .loader import load_fits_into_frame

    # 1) average data & normalize
    # obs image has shape band, line, sample
    line_average = np.nanmedian(
        obs_image[:, :, left_col_cutoff:right_col_cutoff],
        axis=0
    )

    # load mask and un-mask readout cols and filter seam rows,
    # trim to L1B width
    mask = load_fits_into_frame(flag_path)
    mask = mask != 0
    mask[ignored_flag_rows, :] = False
    mask[:, ignored_flag_cols] = False
    mask = mask[:, left_col_cutoff:right_col_cutoff]

    # TODO: add a warning or check on the number of masked pixels per band.
    #  like we wouldn't want to do this where there are more than 50% flagged
    #  probably. enough evenly distributed to make a robust-ish mask?
    flat = np.where(mask, np.nan, line_average)

    # 1b) normalize to center 40 samples like Green did
    flat = normalize_to_center(flat, n_center)

    # 2) 2d surface removal
    flat = flat / fit_surface(flat)

    # 3) divide out spectral average.
    # average from across the whole flat or the middle?
    flat = flat / np.nanmedian(flat, axis=0, keepdims=True)

    # repeat 1b
    flat = normalize_to_center(flat, n_center)

    # fill masked flat values using ratios with neighbors in the same band
    flat = get_masked_flat_values(
        obs_image=obs_image,
        flat=flat,
        mask=mask,
        left_col_cutoff=left_col_cutoff,
        right_col_cutoff=right_col_cutoff,
    )
    return flat
