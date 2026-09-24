from astropy.io import fits
from pathlib import Path
import numpy as np
from typing import Optional


def load_flats(flat_path: Path):
    """
    Read in flat files.
    """
    with fits.open(flat_path) as hdul:
        flat = hdul[0].data

    return flat


def apply_flat(
        obs_data: np.ndarray,
        flat_path: Path,
        flag_path: Optional[Path] = None
):
    """
    Load and multiple flat by image. Modify flat for flagged elements
    in the BDE if flag_path is given.
    """
    if flag_path is None:
        flat = load_flats(flat_path)
    else:
        flat = fix_flagged_in_lab_flat(
            flat_path,
            flag_path
        )

    if "lab" in flat_path.name:
        return obs_data * flat[:, np.newaxis, :]
    else:
        flat_masked = flat.copy()
        flat_masked[flat_masked == 0] = 1
        return obs_data / flat_masked[:, np.newaxis, :]


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


def make_flat_field_from_obs(obs_image: np.ndarray, n_center=40) -> np.ndarray:
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
    #TODO: consider using only beginning or end of warm, long obs?
    line_average = np.nanmean(obs_image, axis=1)
    flat = normalize_to_center(line_average, n_center)

    # 2) 2d surface removal
    flat = flat / fit_surface(flat)

    # 3) divide out spectral average
    flat = flat / np.nanmedian(flat, axis=0, keepdims=True)

    # repeat 1b
    return normalize_to_center(flat, n_center)