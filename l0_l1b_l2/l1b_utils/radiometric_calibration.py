from pathlib import Path


"""
From Green 2011: 

Basic M3 radiometric calibration algorithm that converts the measured raw 12 
bit digitized numbers to units of sepctral radiance for each line and sample 
in the image is given in eqn (2): 

(2) L = RCC(C(DN-DS)) 
L = calibration radiance values for all lines / samples / wavelengths 
RCC = lab radiometric cal coefficients as a function of wavelength 
C = correction algorithms and factors (bad elements, dark pedestal, ghost,
non linearity, scattered light, flat fields) 
DN = 12-bit DN 
DS = average dark signal image values 

So this section is supposed to be the "RCC" component. I think the main value
is in the DATE_rdn_cal.tab files, which are a single multiplier cal value for 
channels 1-86. There is also a gain table (DATE_rd_gain.tab) but all the values
are 1... so if that is a mistake or not, there's no use loading them. There
are two other tables with wavelength values per channel and their centers. 
I don't think I need to use those here? 

so RCC = rdn_cal.tab values, we multiply by channels 

"""


def load_rdn_cal(rdn_cal_path: Path):
    """
    Read in radiometric calibration shape correction factors per channel.
    """
    import pandas as pd

    rdn_cal = pd.read_fwf(
        rdn_cal_path,
        names=['channel', 'rdn_cal_coeff'])
    return rdn_cal['rdn_cal_coeff'].values
