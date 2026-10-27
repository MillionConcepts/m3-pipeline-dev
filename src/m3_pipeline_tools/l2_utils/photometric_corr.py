import numpy as np
from pathlib import Path


def load_f_alpha_factors(f_alpha_path: Path, phase_angle: np.ndarray):
    """
    The actual Falpha_norm(α, λ) factor to be applied, is linearly interpreted
    in from a look-up table Falpha(α, λ) of correction factors dependent on
    α and λ.

    Falpha_norm(α, λ) is normalized to 30º phase,
    so Falpha(30, λ) / Falpha(α, λ)
    """
    import pandas as pd

    # Load Falpha_norm(α, λ)
    # unfortunately all the factors for one phase angle are in the same column
    falpha = pd.read_fwf(f_alpha_path, names=['wavelength',
                                              'f_alpha_corr_factors'])
    # TODO: we need to choose the angle from above and below the given phase
    #  angle and then linearly interpolate them and then normalize to 30 phase
    #  angle? and the phase angle can change throughout the array so idk there
    #  needs to be an efficient way to do this

    return falpha


def photometric_correction(
        image: np.ndarray,
        f_alpha_path: Path,
        obs_path: Path
) -> np.ndarray:
    """
    Photometric correction using local topography.

    L2s4(λ) = L2s3(λ) * { XL_norm(i_topo,e_topo,α) * Falpha_norm(α, λ) }

    We use geometry info from the l1b obs image to calculate the photometric
    limb darkening factor, Xl.
    The ten 'bands' (axis 1) of the obs file are the following:
    1. to-sun azimuth angle (decimal degrees, clockwise from local north)
    2. to-sun zenith angle (decimal degrees, zero at zenith)
    3. to-sensor azimuth angle (decimal degrees, clockwise from local north)
    4. to-sensor zenith angle (decimal degrees, zero at zenith)
    5. observation phase angle (decimal degrees, in plane of to-sun and
    to-sensor rays)
    6. to-sun path length (decimal au with scene mean subtracted)
    7. to-sensor path length (decimal meters)
    8. surface slope from DEM (decimal degrees, zero at horizontal)
    9. surface aspect from DEM (decimal degrees, clockwise from local north)
    10. local cosine i (unitless, cosine of angle between to-sun and local
    DEM facet normal vectors)
    """
    from astropy.io import fits

    obs = fits.open(obs_path)

    f_alpha_factors = load_f_alpha_factors(f_alpha_path, obs[4].data)

    # i_topo= incidence angle in degrees as supplied by the equation:
    # i= acos(cos(obs[1])*cos(obs[7])+sin(obs[1])*sin(obs[7])*cos((obs[0]-
    # obs[8]))). If the resulting incidence angle is greater than or equal to
    # 85.0 degrees then set i_topo to 85.0 before proceeding with the
    # photometry correction.

    i_topo = np.acos(np.cos(obs[1].data) * np.cos(obs[7].data) +
                     np.sin(obs[1].data) * np.sin(obs[7].data) *
                     np.cos((obs[0].data - obs[8].data)))

    i_topo[i_topo > 85.0] = 85.0

    # e_topo= emission angle in degrees as supplied by the equation:
    # e= acos(cos(obs[3])*cos(obs[7])+sin(obs[3])*sin(obs[7])*cos((obs[2]-
    # obs[8]))). If the resulting emission angle is greater than or equal to
    # 85.0 degrees then set e_topo to 85.0 before proceeding with the
    # photometry correction.

    e_topo = np.acos(np.cos(obs[3].data) * np.cos(obs[7].data) +
                     np.sin(obs[3].data) * np.sin(obs[7].data) *
                     np.cos((obs[2].data - obs[8].data)))

    e_topo[e_topo > 85.0] = 85.0

    # XL_norm(i_topo,e_topo,α) is normalized to 30º phase; equal to
    # XL(30,0,30) / XL(i_topo,e_topo,α) where:
    # XL(i_topo,e_topo,α) = (cos(i_topo) / (cos(e_topo) + cos(i_topo))),
    # a simple Lommel-Seeliger model
    # IDK why phase angle is lasted in the inputs bc we don't seem to use it
    x_l_topo = np.cos(i_topo) / (np.cos(e_topo) + np.cos(i_topo))
    x_l_30_0_30 = np.cos(30) / (np.cos(0) + np.cos(30))
    x_l_norm = x_l_30_0_30 / x_l_topo

    # too much memory? we could always do this by frame or band
    # probably band by band bc why load the obs fits file 1000 or so times...
    del e_topo
    del i_topo
    del obs
    del x_l_topo
    del x_l_30_0_30

    phot_correction = x_l_norm * f_alpha_factors
    # at the moment phot_correction would be in downtrack format (ie bands)
    # and the image is in frame POV
    image = image * phot_correction.transpose(1, 0, 2)

    return image
