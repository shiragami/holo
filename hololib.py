#!/usr/bin/python

import numpy as np
import scipy.misc as sm
from skimage.restoration import unwrap_phase

# Class for a single channel hologram
class HoloSingle:
    def __init__(self,channel=None):
        self.amp = None
        self.phi = None
        self.holo = None
        self.spec = None
        self.channel = channel

    # TODO:change dis to external holo.function. Nah
    def read(self,filename,crop=None):
        filetype = filename[-3:].lower()
        if filetype == "bmp":
            channel = {'R':0,'G':1,'B':2}
            self.holo = sm.imread(filename)[:,:,channel[self.channel]]
            if crop != None:
                self.holo = self.holo[crop[0]:crop[0]+1024,crop[1]:crop[1]+1024]
            
        elif filetype == "dat":
        # TODO:check this
            self.holo = np.memmap(filename,dtype=np.float32,shape=(1024,1300),mode='r')
            self.holo = self.holo[crop[0]:crop[0]+1024,crop[1]:crop[1]+1024]
        
    # Extract wavefront from hologram
    def extract(self,scrop):
        py,px,width = scrop

        # FFT image
        spec = np.fft.fft2(self.holo)
        spec = np.fft.fftshift(spec)
        self.spec = np.log(np.abs(spec))

        # Crop spectral
        objspec = spec[py:py+width,px:px+width]

        #TODO:add mask support

        # Shift and IFFT
        objwave = np.zeros([1024,1024],dtype=np.cfloat)
        objwave[512-width/2:512+width/2,512-width/2:512+width/2] = objspec
        self.holo = objspec

        # IFFT
        objwave = np.fft.fftshift(objwave)
        objwave = np.fft.ifft2(objwave)
 
        self.amp = np.abs(objwave)
        self.phi = np.angle(objwave)
       
    # Remove background phase
    def minusphase(self,phase):
        objwave = np.exp(1.j*self.phi)/np.exp(1.j*phase)
        self.phi = np.angle(objwave)
#        self.phi = np.sin(self.phi-phase)

    # Phase unwrapping
    def unwrap(self):
        self.phi = unwrap_phase(self.phi)
        #self.phi = np.unwrap(self.phi)
        pass

# Class for 3 channel hologram
class Holo:
    def __init__(self,imgfile=None,crop=None,high=None):
        self.R = HoloSingle('R')
        self.G = HoloSingle('G')
        self.B = HoloSingle('B')

        if imgfile is not None:
            imgholo = sm.imread(imgfile)

            if crop is not None:
                imgholo = imgholo[crop[0]:crop[0]+1024,crop[1]:crop[1]+1024,:]
                self.R.holo = imgholo[:,:,0]
                self.G.holo = imgholo[:,:,1]
                self.B.holo = imgholo[:,:,2]

#function to save
def holosave(filename,data):
    filetype = filename[-3:].lower()
    if filetype == "png" or filetype == "bmp":
        sm.imsave(filename,data)
    elif filetype == "dat":
        x = np.array(data,'float32')
        fo = open(filename,'wb')
        x.tofile(fo)
        fo.close()
    elif filetype == "raw":
        x = np.array(data,'uint8')
        fo = open(filename,'wb')
        x.tofile(fo)
        fo.close()
    else:
        print "Unsupported file type:%s"%filetype
    return

def holoload(ampfilename,phifilename):
    # Both must be float32, 1024x1024
    holo = HoloSingle()

    holo.phi = np.memmap(phifilename,dtype=np.float32,shape=(1024,1024),mode='r')
    holo.amp = np.memmap(ampfilename,dtype=np.float32,shape=(1024,1024),mode='r')

    return holo

def fresnel(holo,lambd,dd,dz):
    print ("Calculating Fresnel propagation at %s m") % (dz)
    # Generate wave, optimized
    imgtheta = np.fromfunction(lambda i,j:np.square(i-512) + np.square(j-512),
                               (1024,1024),dtype=np.float32)
    imgtheta = imgtheta*np.pi*np.square(dd)/(lambd*dz)

    # Create the propagation wave buffer
    imgwave = np.zeros([1024*2,1024*2],np.cfloat)
    imgwave[512:512+1024,512:512+1024] = np.cos(imgtheta) + 1.j*np.sin(imgtheta)
#    imgwave = np.cos(imgtheta) + 1.j*np.sin(imgtheta)

    # FFT & shift the propagation wave
    Wspec = np.fft.fft2(imgwave)
    Wspec = np.fft.ifftshift(Wspec)

    # Create buffer for object wave
    objwave = holo.amp*np.exp(1j*holo.phi)
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
    imgRecon = imgRecon[512:512+1024,512:512+1024]

    holoRecon = HoloSingle()
    holoRecon.amp = np.abs(imgRecon)
    holoRecon.phi = np.angle(imgRecon)
    holoRecon.holo = holo.holo
    holoRecon.spec = holo.spec

    return holoRecon

def isholo(img):
    img = img[:1024,:1024]
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    
    img = np.log(np.abs(img))
    mR = np.mean(img[128:384,128:385])
    mL = np.mean(img[128:384,640:896])

    if np.abs(mR-mL) > 1.:
        return True
    else:
        return False 
