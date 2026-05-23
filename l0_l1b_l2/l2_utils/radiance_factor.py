import numpy as np
from pathlib import Path


def calculate_radiance_factor(image: np.ndarray, sol_spec_path: Path):
    """
    Get I/F (radiance factor) using solar distance and solar spectrum file.
    L2s1(λ) = L1b(λ) * π / (SolarIrrad(λ) / d )

    SolarIrrad(λ) is a Global or Target file providing the exo-atmospheric
    solar spectrum at 1 Astronomical Unit as determined with MODTRAN.
    See Anderson et al., [2000] and Kurucz, [1995] (DPSIS 2011)
    """
    import pandas as pd

    # L1b(λ) * π
    image *= np.pi

    # then / (SolarIrrad(λ) / d)
    d = 1.0  # 1 AU placeholder until I pull them from l1b headers
    solarrad = pd.read_fwf(sol_spec_path, names=['wavelength', 'sol_irrad'])
    # fairly pointless atm
    solarrad['sol_irrad'] = solarrad['sol_irrad'] / d

    image *= solarrad['sol_irrad'].values[np.newaxis, :, np.newaxis]

    return image
