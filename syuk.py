#!/usr/bin/python

import numpy as np
from skimage.feature import match_template
import scipy.ndimage as sn
import os
from subprocess import call
from scipy.ndimage.filters import gaussian_filter

# Function to compensate brightfield image.
# INPUTS
#  img    : Input image
#  imgbgd : Background image (No sample)
# OUTPUTS : Image with compensated background 
# Please apply PS:autocontrast for better comam image
def PseudoWhiteBalance(img,imgbgd):
    assert img.shape == imgbgd.shape
    # If grayscale image
    if len(img.shape) == 2:
#        bmean = np.mean(imgbgd)
        img = img/(imgbgd+1.)*128.
    else:
        for channel in range(img.shape[2]):
#            bmean = np.mean(imgbgd[...,channel])
            img[...,channel] = img[...,channel]/(imgbgd[...,channel]+1.)*128.
    return img



# Function to calculate the shifted distance sx,sy 
# between two images using image correlation
# INPUTS
#  imgref: Reference image. Ex. holo.amp
#  img   : Shifted image. Ex. brightfield.red
def CalculateImageShift(imgref,img):
    result = match_template(imgref,img,pad_input=True)
    ij = np.unravel_index(np.argmax(result),result.shape)
    sx,sy = ij[::-1]
    sx,sy = 512-sx,512-sy
    return sx,sy

# Circular shift image
def ShiftImage(img,sx,sy):
    img = np.roll(img,-sx,axis=1)
    img = np.roll(img,-sy,axis=0)
    return img

# Function to detect glass region in HE stained biopsy speciment
# Return binary image, True if glass
def detectglass(img,stain='HE'):
    if stain=='HE':
        # Get the green channel
        imgB = img[:,:,1]

        # Apply blur
        imgB = gaussian_filter(imgB, 1.5)

        # Apply threshold
        mu,sig = np.mean(imgB),np.std(imgB) 
        imgglass = imgB > mu + 1.1*sig

    return imgglass

# Shift phase so that background (glass) equals close to zero
# imggnd: binary image for background(glass). Glass= True
def NormalizePhase(imgphase,imggnd=None):

    # If no ground image is specified, use center region as reference
    if imggnd is None:
        imgcenter = imgphase[256:256+512,256:256+512]
        mu,sig = np.mean(imgcenter),np.std(imgcenter)
        imgphase = imgphase - (mu-2.0*sig)
    else:
        # Create masked array. None glass area is masked (ignored)
        imgglass = np.ma.masked_array(imgphase,mask=np.invert(imggnd))
        mu = imgglass.mean()
        print mu
        imgphase = imgphase - mu
    return imgphase

# Map phase image to 0~255 grayscale
def maptogray(imgphase,tmin,tmax):
    imgphase = (imgphase-tmin)/(tmax-tmin)*255
    imgphase[imgphase<0] = 0
    imgphase[imgphase>255] = 255
    imgphase = imgphase.astype(np.uint8,copy=False)
    return imgphase


# Save numpy array as binary float32
def savefloat(filename,img):
    x = np.array(img,'float32')
    fo = open(filename,'wb')
    x.tofile(fo)
    fo.close()
    return

# Hologram extraction using Takeda's method
# imgHolo : Image of interference patterns
# channel : RGB channel
# crop    : tuple for image cropping region (py,px)
# point   : tuple for coordinate of obj wave in freq space (py,px)
# width   : window size for obj wave

def extractHolo(imgHolo,paramExtract):
    channel = paramExtract['channel']
    crop    = paramExtract['crop']
    point   = paramExtract['point']
    width   = paramExtract['width']
    mask    = paramExtract['mask']

    assert len(imgHolo.shape) == 3
    assert len(crop) == 2
    assert len(point) == 2

    c = {'R':0,'G':1,'B':2}

    # Get single channel and crop image
    imgHolo = imgHolo[...,c[channel]]
    imgHolo = imgHolo[crop[0]:crop[0]+1024,crop[1]:crop[1]+1024]

    # FFT image
    spec = np.fft.fft2(imgHolo)
    spec = np.fft.fftshift(spec)

    # Crop obj wave
    objspec = spec[point[0]:point[0]+width,point[1]:point[1]+width]

    # Apply mask
    if mask:
        # Create mask
#        imgmask = np.fromfunction(lambda i,j:(i-width/2)**2 + (j-width/2)**2,(width,width),dtype=np.float32)
#        imgmask = imgmask < (width/2)**2
#        objspec = objspec*imgmask

        # Butterworth filter
        cutoff = 45
        order = 3
        imgmask = np.fromfunction(lambda i,j:(i-width/2)**2 + (j-width/2)**2,(width,width),dtype=np.float32)
        imgmask = np.sqrt(imgmask)
        imgmask = np.sqrt(1./(1. + np.power(imgmask/(float(cutoff)),2*order)))
        objspec = objspec*imgmask


    # Shift and IFFT
    objwave = np.zeros([1024,1024],dtype=np.cfloat)
    objwave[512-width/2:512+width/2,512-width/2:512+width/2] = objspec

    # IFFT
    objwave = np.fft.fftshift(objwave)
    objwave = np.fft.ifft2(objwave)

    return objwave


# Fresnel propagate
# Assuming the buffer size is always 1024x1024
# lambd : laser wavelength
# dd    : pixel size
# dz    : propagation distance
def Fresnelpropagate(objwave,lambd,dd,dz):
    print ("Calculating Fresnel propagation at %s m") % (dz)
    # Generate wave, optimized
    imgtheta = np.fromfunction(lambda i,j:np.square(i-512) + np.square(j-512),(1024,1024),dtype=np.float32)
    imgtheta = imgtheta*np.pi*np.square(dd)/(lambd*dz)

    # Create the propagation wave buffer
    imgwave = np.zeros([1024*2,1024*2],np.cfloat)
    imgwave[512:512+1024,512:512+1024] = np.cos(imgtheta) + 1.j*np.sin(imgtheta)

    # FFT & shift the propagation wave
    Wspec = np.fft.fft2(imgwave)
    Wspec = np.fft.ifftshift(Wspec)

    # Create buffer for object wave
    imgHolo = np.zeros([1024*2,1024*2],np.cfloat)
    imgHolo[512:512+1024,512:512+1024] = objwave

    # FFT & shift the object wave
    Hspec = np.fft.fft2(imgHolo)
    Hspec = np.fft.ifftshift(Hspec)

    # Convolution
    Rspec = Hspec*Wspec
    Rspec = np.fft.ifftshift(Rspec)

    imgRecon = np.fft.ifft2(Rspec)
    imgRecon = np.fft.fftshift(imgRecon)

    return imgRecon[512:512+1024,512:512+1024]

# Phase unwrap using Miguel unwrapper
# Mask: bool array. Element with value False will be ignored
def unwrap(imgphase,mask=None):

    savefloat("inptmpphi.dat",imgphase)
    if mask is None:
        call(["/home/tk2/lib/Miguel_2D_unwrapper_with_mask","inptmpphi.dat","outtmpphi.dat"])
    else:
        assert mask.dtype == bool
        # Save mask as uint8
        x = np.array(mask*255,'uint8')
        fo = open("inptmpmask.raw",'wb')
        x.tofile(fo)
        fo.close()
        call(["/home/tk2/lib/Miguel_2D_unwrapper_with_mask","inptmpphi.dat","outtmpphi.dat","inptmpmask.raw"])
        os.remove("inptmpmask.raw")
    # Reload phase
    imgphase = np.memmap("outtmpphi.dat",dtype=np.float32,shape=(1024,1024),mode='r')
    # Remove temporary file
    os.remove("inptmpphi.dat")
    os.remove("outtmpphi.dat")
    return imgphase

# Remove background phase
def rmvbgdphase(objwave,zerowave):
    objwave = objwave/np.exp(1.j*np.angle(zerowave))
    return objwave

# Cos and Sin average filter
# Input: Noisy phase image
# step: Iteration step
# wsize: Window size for spatial averaging 
def cosavgfilter(imgphase,step=20,wsize=5):
#    imgp = np.zeros([1024,1024],dtype=np.float32)
#    imgphase = np.memmap(phase_path,dtype=np.float32,shape=(1024,1024),mode='r')
    for k in range(step):
        # Convert to sin and cosine
        imgcos = np.cos(imgphase)
        imgsin = np.sin(imgphase)

        # Apply average filter
        imgsin = sn.filters.uniform_filter(imgsin,size=wsize)
        imgcos = sn.filters.uniform_filter(imgcos,size=wsize)

        # Reconvert sin & cosine to phase
        imgphase = np.arctan2(imgsin,imgcos)
    return imgphase
