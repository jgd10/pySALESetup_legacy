import pySALESetup as pss
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def reverse_phi(p):
    return (2**(-p))*.5*1.e-3

m = pss.Mesh(X=5000,Y=5000)
G = pss.Ensemble(m)
SD = pss.SizeDistribution(func='lognormal',mu=3.,sigma=.5)
phi = np.linspace(0,6,20)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.hist(phi,bins=20)
#print len(Counter(phi).keys())
#print len(np.unique(phi))
dp = abs(phi[0]-phi[1])

Area = np.float64(.5*500.**2.)
#Nparts = 1000.
for p in phi:
    freq = SD.frequency(p,dp)*Area
    r = reverse_phi(p)/m.cellsize
    #print SD.frequency(p,dp)*Area,np.pi*r**2.
    freq = int(freq/(np.pi*r**2.))
    print freq
    for f in range(freq):
        g = pss.Grain(r)
        G.add(g,x=0,y=0)

#print G.details()
#print G.PSDdetails()
G.plotPSD()

