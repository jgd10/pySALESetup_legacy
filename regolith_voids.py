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

groupA = pss.Ensemble(meshA,name='shapevariationENS_A')
groupB = pss.Ensemble(meshB,name='shapevariationENS_B')
try:
    i = 0
    for g in grains:
        for f in range(g.targetFreq):
            g.insertRandomly(meshA, m=1)
            groupA.add(g,g.x,g.y)
            g.insertRandomly(meshB, m=1)
            groupB.add(g,g.x,g.y)
except KeyboardInterrupt:
    pass


#print groupA.fabricTensor_discs()
groupA.calcPSD()

groupA.optimise_materials(np.array([1,2,3,4,5,6,7,8]))
groupB.optimise_materials(np.array([1,2,3,4,5,6,7,8]))

groupA.save()
groupB.save()


meshA.fillAll(-1)
meshB.fillAll(-1)

for xA,yA,gA,mA,xB,yB,gB,mB in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats,groupB.xc,groupB.yc,groupB.grains,groupB.mats):
    gA.place(xA,yA,mA,meshA)
    gB.place(xB,yB,mB,meshB)

meshA.fillAll(9)
meshB.fillAll(9)

meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)

meshC = pss.combine_meshes(meshA,meshB,axis=1)
meshC.top_and_tail()
meshC.viewMats()
meshC.save(fname='regolith_circles_v3000.iSALE',compress=True)
meshC.multiplyVels()
meshC.save(fname='regolith_circles_v1500.iSALE',compress=True)
meshC.multiplyVels()
meshC.save(fname='regolith_circles_v750.iSALE',compress=True)

meshA.fillAll(-1)
meshB.fillAll(-1)

r = 0.5

r *= np.pi

off = .8

R = [[-1.,-1.],
     [-1.,1.],
     [1.,-1.],
     [1.,-1.]]

R[2][1] += off

for xA,yA,gA,mA in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats):
    grain1 = pss.Grain(eqr=gA.radius,shape='polygon',poly_params=R,rot=r)
    grain1.place(xA,yA,mA,meshA)
for xB,yB,gB,mB in zip(groupB.xc,groupB.yc,groupB.grains,groupB.mats):
    grain2 = pss.Grain(eqr=gB.radius,shape='polygon',poly_params=R,rot=r*3)
    grain2.place(xB,yB,mB,meshB)

meshA.fillAll(9)
meshB.fillAll(9)
meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)
meshD = pss.combine_meshes(meshA,meshB,axis=1)
meshD.top_and_tail()
meshD.viewMats()
meshD.save(fname='regolith_tetrahedracritangle_v3000.iSALE',compress=True)
meshD.multiplyVels()
meshD.save(fname='regolith_tetrahedracritangle_v1500.iSALE',compress=True)
meshD.multiplyVels()
meshD.save(fname='regolith_tetrahedracritangle_v750.iSALE',compress=True)
meshA.fillAll(-1)
meshB.fillAll(-1)

r = 0.5

r *= np.pi

off = 2.

R = [[-1.,-1.],
     [-1.,1.],
     [1.,-1.],
     [1.,-1.]]

R[2][1] += off

for xA,yA,gA,mA in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats):
    grain1 = pss.Grain(eqr=gA.radius,shape='polygon',poly_params=R,rot=r*3)
    grain1.place(xA,yA,mA,meshA)
for xB,yB,gB,mB in zip(groupB.xc,groupB.yc,groupB.grains,groupB.mats):
    grain2 = pss.Grain(eqr=gB.radius,shape='polygon',poly_params=R,rot=r)
    grain2.place(xB,yB,mB,meshB)

meshA.fillAll(9)
meshB.fillAll(9)
meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)
meshE = pss.combine_meshes(meshA,meshB,axis=1)
meshE.top_and_tail()
meshE.viewMats()
meshE.save(fname='regolith_square_v3000.iSALE',compress=True)
meshE.multiplyVels()
meshE.save(fname='regolith_square_v1500.iSALE',compress=True)
meshE.multiplyVels()
meshE.save(fname='regolith_square_v750.iSALE',compress=True)

meshA.fillAll(-1)
meshB.fillAll(-1)

r = 0.5

r *= np.pi


R = [[-1.,-1.],
     [-1.,1.],
     [1.,-1.],
     [1.,-1.]]


for xA,yA,gA,mA in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats):
    grain1 = pss.Grain(eqr=gA.radius,shape='polygon',poly_params=R,rot=r)
    grain1.place(xA,yA,mA,meshA)
for xB,yB,gB,mB in zip(groupB.xc,groupB.yc,groupB.grains,groupB.mats):
    grain2 = pss.Grain(eqr=gB.radius,shape='polygon',poly_params=R,rot=r*3)
    grain2.place(xB,yB,mB,meshB)

meshA.fillAll(9)
meshB.fillAll(9)
meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)
meshF = pss.combine_meshes(meshA,meshB,axis=1)
meshF.top_and_tail()
meshF.viewMats()
meshF.save(fname='regolith_triangle_v3000.iSALE',compress=True)
meshF.multiplyVels()
meshF.save(fname='regolith_triangle_v1500.iSALE',compress=True)
meshF.multiplyVels()
meshF.save(fname='regolith_triangle_v750.iSALE',compress=True)

meshA.fillAll(-1)
meshB.fillAll(-1)

import glob
import random

allfiles = glob.glob('./grain_library/grain_area*.txt')

for xA,yA,gA,mA in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats):
    fff = random.choice(allfiles)
    grain1 = pss.Grain(eqr=gA.radius,shape='file',File=fff)
    grain1.place(xA,yA,mA,meshA)

for xB,yB,gB,mB in zip(groupB.xc,groupB.yc,groupB.grains,groupB.mats):
    fff = random.choice(allfiles)
    grain2 = pss.Grain(eqr=gB.radius,shape='file',File=fff)
    grain2.place(xB,yB,mB,meshB)

meshA.fillAll(9)
meshB.fillAll(9)
meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)
meshG = pss.combine_meshes(meshA,meshB,axis=1)
meshG.top_and_tail()
meshG.viewMats()
meshG.save(fname='regolith_realistic_v3000.iSALE',compress=True)
meshG.multiplyVels()
meshG.save(fname='regolith_realistic_v1500.iSALE',compress=True)
meshG.multiplyVels()
meshG.save(fname='regolith_realistic_v750.iSALE',compress=True)
