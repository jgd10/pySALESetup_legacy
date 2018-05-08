import pySALESetup as pss
import numpy as np
import glob
import random

def DD_(x):
    return 2.**(-x)

allfiles = glob.glob('../grain_library/grain_area*.txt')
# create 5 different particles
N = 4
# Generate N phi values and equiv radii (in cells)
minphi  = -np.log2(2*4*2.5e-3)
maxphi  = -np.log2(2*200*2.5e-3)
phi = np.linspace(minphi,maxphi,N)

Rs = ((DD_(phi)*.5*1.e-3)/2.5e-6)
ctr = 0
for r,p in zip(Rs,phi):
    # generate grain object with radius r
    fff = random.choice(allfiles)
    g = pss.Grain(eqr=int(r),shape='file',File=fff)
    g.view(save=True,fname='grain_{}.png'.format(ctr),show=False)
    ctr += 1

