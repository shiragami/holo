#!/usr/bin/python

import scipy.misc as sm
import numpy as np

# Function to generate HDR holo
# imginp : dictionary containing image with different exposures
# hdrset : exposure setting, key must match img
def generate_HDR(imginp,hdrset):
    response = np.loadtxt("response_flovel_red.dat",usecols=(1,))
    imgHDR = np.zeros([1024,1024]) # Final HDR image
    imgsum = np.zeros([1024,1024]) # Weight sum

    # Weighting function (hat)
    def w(z):
        if z<= 127:
            return z
        else:
            return 255-z

    # Camera response function
    def g(z):
        return response[z]

    # Vectorize functions
    vw = np.vectorize(w)
    vg = np.vectorize(g)

    for n in imginp:
        imgw = vw(imginp[n])
        imgsum += imgw
        imgHDR += imgw*(vg(imginp[n])-np.log(hdrset[n]))

    # Apply weight to nonzero element
    mask = imgsum > 0 
    imgHDR[mask] = imgHDR[mask]/imgsum[mask]

    return imgHDR
