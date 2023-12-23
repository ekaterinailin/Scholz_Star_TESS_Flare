"""
UTF-8
Python 3.10

Flares in TESS Sectors 7 and 33 of Scholz's star, an M9+T5 nearby binary

Observed as TIC 318801864, TESS Sectors 7 and 33 @ 2 min cadence

This script fetches the light curves, detrends them, and calculates the flare energy.

"""

import numpy as np
import pandas as pd

from astropy.modeling import models
from astropy import units as u
from astropy.constants import sigma_sb

import matplotlib.pyplot as plt

from altaipony.flarelc import FlareLightCurve
from altaipony.lcio import from_mast
from altaipony.customdetrend import custom_detrending


# ignore FutureWarning and RuntimeWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def flare_factor(teff, radius, wav, resp,  tflare=10000):
    """Calculate the flare energy factor in ergs, following Shibaya et al. (2013)

    Parameters
    ----------
    teff : float
        Stellar effective temperature in Kelvin.
    radius : float
        Stellar radius in solar radii.
    wav : array
        Array of wavelengths in nanometers.
    resp : array
        Array of bandpass responses.
     tflare : float
        Flare temperature in Kelvin.
    
    Returns
    -------
    factor : float
        Flare energy factor in ergs/s.
    """

    # blackbody
    bb = models.BlackBody(temperature=teff * u.K)

    # blackbody flux in TESS band
    bbwavs = bb(wav * u.nm)  * resp

    fluxs = np.trapz(bbwavs.value, wav)

    # blackbody
    bb = models.BlackBody(temperature=tflare * u.K)

    # blackbody flux in TESS band
    bbwavf = bb(wav * u.nm)  * resp

    fluxf = np.trapz(bbwavf.value, wav)

    ratio = fluxs / fluxf

    factor = ratio * np.pi * (radius * u.R_sun) ** 2 * sigma_sb * (tflare * u.K)**4

    return factor.to("erg/s")

    
if __name__ == "__main__":
    

    # read in the TESS light curves
    lc1 = from_mast("TIC 318801864", c=7, mission="TESS", mode="LC", cadence="short")
    lc2 = from_mast("TIC 318801864", c=33, mission="TESS", mode="LC", cadence="short")

    # detrend both light curves (there is probably only little variability anyways)
    lc1d = lc1.detrend(mode="custom", func=custom_detrending)
    lc2d = lc2.detrend(mode="custom", func=custom_detrending)

    # find flares in both light curves
    f2 = lc2d.find_flares().flares
    print("\n Flares in sector 33:\n")
    print(f2)

    f1 = lc1d.find_flares().flares
    print("\n Flares in sector 7:\n")
    print(f1)

    # plot the flare in question
    plt.plot(lc1.time.value, lc1.flux.value, "k.", label="PDCSAP_FLUX")
    plt.plot(lc1d.time.value, lc1d.detrended_flux.value+100, "r", label="detrended light curve")
    plt.xlabel("Time [BJD - 2457000]")
    plt.ylabel(r"Flux [e$^{-}$/s]")
    plt.xlim(1493.5, 1494)
    plt.legend(frameon=False)

    plt.savefig("flare.png", dpi=300)

    # calculate the total observing time in days
    total_t = (lc1.detrended_flux.value.shape[0] + lc2.detrended_flux.value.shape[0]) / 30 / 24


    # read TESS response function
    tess_resp = pd.read_csv("tess-response-function-v2.0.csv", skiprows=7, names=["wav", "resp"], header=None)
    wav, resp = tess_resp.wav.values, tess_resp.resp.values

    # effective temperature of Scholz's star in K from Pecaut and Mamajek 2013
    teff = 2350

    # radius of Scholz's star in Solar radii from Pecaut and Mamajek 2013
    radius = 0.102

    print("\n Use effective temperature and radius from Pecaut and Mamajek (2013) for an M9 dwarf:\n")
    print(f"Effective temperature: {teff:.0f} K")
    print(fr"Radius: {radius:.3f} solar radii")

    # calculate bolometric flare energy
    print("\n We use an 8000K flare temperature, accounting for decreasing flare temperature with later spectral type.")
    print("See Maas et al. (2022)")
    bol_energy = flare_factor(teff, radius, wav, resp,  tflare=8000) * f1.ed_rec.iloc[0] * u.s


    print("\nBolometric flare energy in ergs:")
    print(f"{bol_energy:.2e}")


    # now assume that this flare energy is representative of a cumulative power law of flares with slope -1.8
    # and calculate the flare frequency above 10^35 erg
    slope = 1.8

    print(f"\n Assume a power law of flares with slope -{slope:.1f} and calculate the flare frequency above X erg:\n")

    beta = (slope - 1) / total_t / (bol_energy.value)**(-slope+ 1) 

    for exp in [32, 33, 34, 35, 36]:

        freq_e3x = beta / (slope - 1) * (10**exp)**(-slope + 1) * 365.25

        print(f"\nFlare frequency above 10^{exp} erg:")
        print(f"{freq_e3x:.2e} flares per year")


