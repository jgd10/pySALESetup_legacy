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
meshC = pss.Mesh(X=500,Y=1200,cellsize=2.5e-6)
meshD = pss.Mesh(X=500,Y=1200,cellsize=2.5e-6)
meshA.label='A'
meshB.label='B'
meshC.label='C'
meshD.label='D'

# target volume (area) fraction
vfrac = 0.5

# Store grain objects in list, 'grains'
grainsA = []
grainsB = []
grainsC = []
grainsD = []

# Minimum krubeim phi = min resolution (4 cppr)
# Max ... '' '' '' '' = max resolution (200 cppr) 
# Max res is one which still fits in the domain
minphi  = -np.log2(2*4*2.5e-3)
maxphi  = -np.log2(2*200*2.5e-3)

NA = 3
NB = 6
NC = 9
ND = 12
# Generate N phi values and equiv radii (in cells)
phiA = np.linspace(minphi,maxphi,NA)
phiB = np.linspace(minphi,maxphi,NB)
phiC = np.linspace(minphi,maxphi,NC)
phiD = np.linspace(minphi,maxphi,ND)

RsA = ((DD_(phiA)*.5*1.e-3)/meshA.cellsize)
RsB = ((DD_(phiB)*.5*1.e-3)/meshB.cellsize)
RsC = ((DD_(phiC)*.5*1.e-3)/meshC.cellsize)
RsD = ((DD_(phiD)*.5*1.e-3)/meshD.cellsize)

# interval over which to calculate number from pdf
# No. = |CDF(x+h) - CDF(x-h)| * no. of areas
hA = abs(phiA[1]-phiA[0])*.5
hB = abs(phiB[1]-phiB[0])*.5
hC = abs(phiC[1]-phiC[0])*.5
hD = abs(phiD[1]-phiD[0])*.5

# target area that ALL particles should take up at end
target_area = float(meshA.x*meshA.y*vfrac)
for rA,pA in zip(RsA,phiA):
    # generate grain object with radius r
    gA = pss.Grain(eqr=int(rA))
    # calculate the target number of grains from CDF (see above)
    prob = abs(CDF(pA+hA) - CDF(pA-hA))
    gA.targetFreq = int(round(prob * (target_area/float(gA.area))))
    grainsA.append(gA)
for rB,pB in zip(RsB,phiB):
    # generate grain object with radius r
    gB = pss.Grain(eqr=int(rB))
    # calculate the target number of grains from CDF (see above)
    prob = abs(CDF(pB+hB) - CDF(pB-hB))
    gB.targetFreq = int(round(prob * (target_area/float(gB.area))))
    grainsB.append(gB)
for rC,pC in zip(RsC,phiC):
    # generate grain object with radius r
    gC = pss.Grain(eqr=int(rC))
    # calculate the target number of grains from CDF (see above)
    prob = abs(CDF(pC+hC) - CDF(pC-hC))
    gC.targetFreq = int(round(prob * (target_area/float(gC.area))))
    grainsC.append(gC)
for rD,pD in zip(RsD,phiD):
    # generate grain object with radius r
    gD = pss.Grain(eqr=int(rD))
    # calculate the target number of grains from CDF (see above)
    prob = abs(CDF(pD+hD) - CDF(pD-hD))
    gD.targetFreq = int(round(prob * (target_area/float(gD.area))))
    grainsD.append(gD)
# library of grains has been generated, now place them into the mesh! 
# Just meshA for now

# order grains from largest to smallest
grainsA = [g for _,g in sorted(zip(phiA,grainsA))]
grainsB = [g for _,g in sorted(zip(phiB,grainsB))]
grainsC = [g for _,g in sorted(zip(phiC,grainsC))]
grainsD = [g for _,g in sorted(zip(phiD,grainsD))]

groupA = pss.Ensemble(meshA)
groupB = pss.Ensemble(meshB)
groupC = pss.Ensemble(meshC)
groupD = pss.Ensemble(meshD)
try:
    i = 0
    for gA in grainsA:
        for f in range(gA.targetFreq):
            gA.insertRandomly(meshA, m=1)
            groupA.add(gA,gA.x,gA.y)
    for gB in grainsB:
        for f in range(gB.targetFreq):
            gB.insertRandomly(meshB, m=1)
            groupB.add(gB,gB.x,gB.y)
    for gC in grainsC:
        for f in range(gC.targetFreq):
            gC.insertRandomly(meshC, m=1)
            groupC.add(gC,gC.x,gC.y)
    for gD in grainsD:
        for f in range(gD.targetFreq):
            gD.insertRandomly(meshD, m=1)
            groupD.add(gD,gD.x,gD.y)
except KeyboardInterrupt:
    pass


groupA.calcPSD()
groupB.calcPSD()
groupC.calcPSD()
groupD.calcPSD()

groupA.optimise_materials(np.array([1,2,3,4,5,6,7,8]))
groupB.optimise_materials(np.array([1,2,3,4,5,6,7,8]))
groupC.optimise_materials(np.array([1,2,3,4,5,6,7,8]))
groupD.optimise_materials(np.array([1,2,3,4,5,6,7,8]))


meshA.fillAll(-1)
meshB.fillAll(-1)
meshC.fillAll(-1)
meshD.fillAll(-1)

for xA,yA,gA,mA in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats):
    gA.place(xA,yA,mA,meshA)
for xB,yB,gB,mB in zip(groupB.xc,groupB.yc,groupB.grains,groupB.mats):
    gB.place(xB,yB,mB,meshB)
for xC,yC,gC,mC in zip(groupC.xc,groupC.yc,groupC.grains,groupC.mats):
    gC.place(xC,yC,mC,meshC)
for xD,yD,gD,mD in zip(groupD.xc,groupD.yc,groupD.grains,groupD.mats):
    gD.place(xD,yD,mD,meshD)

meshA.fillAll(9)
meshB.fillAll(9)
meshC.fillAll(9)
meshD.fillAll(9)

meshA.blanketVel(+1500.,axis=1)
meshB.blanketVel(-1500.,axis=1)
meshC.blanketVel(+1500.,axis=1)
meshD.blanketVel(-1500.,axis=1)

meshAB = pss.combine_meshes(meshA,meshB,axis=1)
meshCD = pss.combine_meshes(meshC,meshD,axis=1)
meshAB.top_and_tail()
meshCD.top_and_tail()
meshAB.viewMats()
meshCD.viewMats()
#meshC.save(fname='regolith_circles_v3000.iSALE',compress=True)
#meshC.multiplyVels()
#meshC.save(fname='regolith_circles_v1500.iSALE',compress=True)
#meshC.multiplyVels()
#meshC.save(fname='regolith_circles_v750.iSALE',compress=True)

