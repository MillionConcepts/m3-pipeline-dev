from pathlib import Path
import numpy as np
from astropy.io import fits


def write_flag_backplane(
        flags: np.ndarray,
        flag_out_path: Path,
        overwrite: bool = True
):
    from ..flagging.make_flag_backplanes import FLAG_DICT

    hdu = fits.PrimaryHDU(flags.astype(np.uint8))
    hdr = hdu.header
    hdr["FLAGTYPE"] = ("BITMASK")

    for key, val, comment in FLAG_DICT:
        hdr[key] = (val, comment)

    hdu.writeto(flag_out_path, overwrite=overwrite)
