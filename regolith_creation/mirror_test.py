import numpy as np
import pySALESetup as pss
import matplotlib.pyplot as plt
from math import ceil

def CDF(x):
    # The CDF
    A = 2.908
    B = 0.028
    C = 0.320
    L = 0.643
    a = 99.4
    return (1./a)*A/(B+C*np.exp(-L*x))

def PDF(x):
    # The PDF
    A = 2.908
    B = 0.028
    C = 0.320
    L = 0.643
    a = 99.4
    pdf  = (1./a)*A*L*C*np.exp(L*x)/(B+C*np.exp(L*x))**2.
    return pdf

def lunar_pdf(x,LB_tol,UB_tol):                                                   
    # Integrate PDF at a value of x
    P     = abs(CDF(x+UB_tol) - CDF(x-LB_tol))
    return P

def PHI_(x):
    """
    x must be in SI (metres) for this function to work
    PHI is only calculated correctly when the arg is mm
    """
    return -1.*np.log2(x*1000.)

def DD_(x):
    return 2.**(-x)

# Top and bottom mesh created separately
meshA = pss.Mesh(X=500,Y=1200,cellsize=2.5e-6)
meshB = pss.Mesh(X=500,Y=1200,cellsize=2.5e-6)
meshA.label='A'
meshB.label='B'

# target volume (area) fraction
vfrac = 0.5

# Store grain objects in list, 'grains'
grains = []

# Minimum krubeim phi = min resolution (4 cppr)
# Max ... '' '' '' '' = max resolution (200 cppr) 
# Max res is one which still fits in the domain
minphi  = -np.log2(2*4*2.5e-3)
maxphi  = -np.log2(2*200*2.5e-3)

# create 10 different particles
N = 10
# Generate N phi values and equiv radii (in cells)
phi     = np.linspace(minphi,maxphi,N)

Rs = ((DD_(phi)*.5*1.e-3)/meshA.cellsize)

# interval over which to calculate number from pdf
# No. = |CDF(x+h) - CDF(x-h)| * no. of areas
h = abs(phi[1]-phi[0])*.5

# target area that ALL particles should take up at end
target_area = float(meshA.x*meshA.y*vfrac)
for r,p in zip(Rs,phi):
    # generate grain object with radius r
    g = pss.Grain(eqr=int(r))
    # calculate the target number of grains from CDF (see above)
    prob = abs(CDF(p+h) - CDF(p-h))
    g.targetFreq = int(round(prob * (target_area/float(g.area))))
    grains.append(g)

# library of grains has been generated, now place them into the mesh! 
# Just meshA for now

# order grains from largest to smallest
grains = [g for _,g in sorted(zip(phi,grains))]

groupA = pss.Ensemble(meshA,name='mirror_test_ens')
try:
    i = 0
    for g in grains:
        for f in range(g.targetFreq):
            g.insertRandomly(meshA, m=1)
            groupA.add(g,g.x,g.y)
except KeyboardInterrupt:
    pass



groupA.optimise_materials(np.array([1,2,3,4,5,6,7,8]))

groupA.save()


meshA.fillAll(-1)

for xA,yA,gA,mA in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats):
    gA.place(xA,yA,mA,meshA)

meshA.fillAll(9)
import copy
meshB = copy.deepcopy(meshA)
meshB.flipMesh()

meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)

meshC = pss.combine_meshes(meshA,meshB,axis=1)
meshC.top_and_tail()
meshC.viewMats()
meshC.viewVels()
meshC.save(fname='regolith_mirror_v3000.iSALE',compress=True)
meshC.multiplyVels()
meshC.save(fname='regolith_mirror_v1500.iSALE',compress=True)
meshC.multiplyVels()
meshC.save(fname='regolith_mirror_v750.iSALE',compress=True)

