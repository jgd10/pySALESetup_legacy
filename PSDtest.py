import pySALESetup as pss
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def reverse_phi(p):
    return (2**(-p))*.5*1.e-3

m = pss.Mesh()
G = pss.Ensemble(m)
phi = np.random.normal(loc=1,scale=.5,size=100)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(phi,bins=20)
#print len(Counter(phi).keys())
#print len(np.unique(phi))
for p in phi:
    r = reverse_phi(p)
    r /= m.cellsize
    g = pss.Grain(r)
    G.add(g,x=0,y=0)

print G.details()
G.calcPSD()

