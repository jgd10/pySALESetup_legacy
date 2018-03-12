import pySALESetup as pss
import numpy as np
from collections import Counter

def reverse_phi(p):
    return (2**(-p))*.5


m = pss.Mesh()
G = pss.Ensemble(m)
phi = np.random.normal(loc=-2,scale=1,size=100)

print len(Counter(phi).keys())
print len(np.unique(phi))
for p in phi:
    r = reverse_phi(p)
    print r
    g = pss.Grain(r)
    G.add(g,x=0,y=0)

G.calcPSD()

