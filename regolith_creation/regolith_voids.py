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
#meshA = pss.Mesh(X=50,Y=120,cellsize=2.5e-5,label='A')
#meshB = pss.Mesh(X=50,Y=120,cellsize=2.5e-5,label='B')
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
#Rs = np.ones((N))*4

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


vf_V = np.array([0.,0.125,0.25])
vf_V = vf_V[::-1]
# matrix porosity = 0.5
#vf_G = 5.*vf_V
# matrix porosity = 0.6
vf_G = 2.*vf_V/3. + 1./6.
# matrix porosity = 0.7
#vf_G = (2. + 3.*vf_V)/7.
# matrix porosity = 0.8
#vf_G = (.25*vf_V+3./8.)
print vf_V,vf_G
first = True
for v_void,v_grain in zip(vf_V,vf_G):
    if first:
        v_grain0 = v_grain
        v_void0 = v_void
        groupAG = pss.Ensemble(meshA,name='voidvariation_vfG-{:2.2f}_vfV-{:2.2f}_GRAINS_A'.format(v_grain0,v_void0))
        groupBG = pss.Ensemble(meshB,name='voidvariation_vfG-{:2.2f}_vfV-{:2.2f}_GRAINS_B'.format(v_grain0,v_void0))
        groupAV = pss.Ensemble(meshA,name='voidvariation_vfG-{:2.2f}_vfV-{:2.2f}_VOIDS_A'.format(v_grain0,v_void0))
        groupBV = pss.Ensemble(meshB,name='voidvariation_vfG-{:2.2f}_vfV-{:2.2f}_VOIDS_B'.format(v_grain0,v_void0))
    else:
        groupAG = pss.Ensemble(meshA,name='voidvariation_vfG-{:2.2f}_vfV-{:2.2f}_GRAINS_A'.format(v_grain0,v_void0),Reload=True)
        groupBG = pss.Ensemble(meshB,name='voidvariation_vfG-{:2.2f}_vfV-{:2.2f}_GRAINS_B'.format(v_grain0,v_void0),Reload=True)
        groupAV = pss.Ensemble(meshA,name='voidvariation_vfG-{:2.2f}_vfV-{:2.2f}_VOIDS_A'.format(v_grain0,v_void0),Reload=True)
        groupBV = pss.Ensemble(meshB,name='voidvariation_vfG-{:2.2f}_vfV-{:2.2f}_VOIDS_B'.format(v_grain0,v_void0),Reload=True)
    if first:
        i = 0
        vfA = 0.
        vfB = 0.
        while vfA <= v_grain:
            g = grains[i]
            for f in range(g.targetFreq):
                g.insertRandomly(meshA, m=1)
                groupAG.add(g,g.x,g.y)
                vfA = meshA.calcVol(frac=True)
                if vfA > v_grain: 
                    break
            i += 1
        j = 0
        while vfB <= v_grain:
            g = grains[j]
            for f in range(g.targetFreq):
                g.insertRandomly(meshB, m=1)
                groupBG.add(g,g.x,g.y)
                vfB = meshB.calcVol(frac=True)
                if vfB > v_grain: 
                    break
            j+= 1

        meshA.fillAll(2)
        meshB.fillAll(2)
    
        vfA = 0.
        vfB = 0.
        g = grains[-1]
        while vfA < v_void:
            g.insertRandomly(meshA, m=0,mattargets=[2])
            groupAV.add(g,g.x,g.y)
            vfA = 1.-meshA.calcVol(frac=True)
            if vfA > v_void: break
        while vfB < v_void:
            g.insertRandomly(meshB, m=0,mattargets=[2])
            groupBV.add(g,g.x,g.y)
            vfB = 1.-meshB.calcVol(frac=True)
            if vfB > v_void: break

        groupAG.optimise_materials(np.array([1,2,3,4,5,6,7,8]))
        groupBG.optimise_materials(np.array([1,2,3,4,5,6,7,8]))
        meshA.fillAll(-1)
        meshB.fillAll(-1)
    else:
        pass
    for xA,yA,gA,mA in zip(groupAG.xc,groupAG.yc,groupAG.grains,groupAG.mats):
        gA.place(xA,yA,mA,meshA)
        vfA = meshA.calcVol(frac=True)
        if vfA > v_grain: break
    for xB,yB,gB,mB in zip(groupBG.xc,groupBG.yc,groupBG.grains,groupBG.mats):
        gB.place(xB,yB,mB,meshB)
        vfB = meshB.calcVol(frac=True)
        if vfB > v_grain: break
    
    meshA.fillAll(9)
    meshB.fillAll(9)
    
    for xA,yA,gA in zip(groupAV.xc,groupAV.yc,groupAV.grains):
        gA.place(xA,yA,0,meshA,mattargets=[9])
        vfA = 1.-meshA.calcVol(frac=True)
        if vfA > v_void: break
    for xB,yB,gB in zip(groupBV.xc,groupBV.yc,groupBV.grains):
        gB.place(xB,yB,0,meshB,mattargets=[9])
        vfB = 1.-meshB.calcVol(frac=True)
        if vfB > v_void: break
    if first:
        groupAG.save()
        groupBG.save()
        groupAV.save()
        groupBV.save()
    
    meshA.blanketVel(-1500.,axis=1)
    meshB.blanketVel(+1500.,axis=1)
    
    meshC = pss.combine_meshes(meshA,meshB,axis=1)
    meshC.top_and_tail()
    meshC.viewMats()
    meshC.save(fname='regolith_vfG-{:2.2f}_vfV{:2.2f}_v3000.iSALE'.format(v_grain,v_void),compress=True)
    meshC.multiplyVels()
    meshC.save(fname='regolith_vfG-{:2.2f}_vfV{:2.2f}_v1500.iSALE'.format(v_grain,v_void),compress=True)
    meshC.multiplyVels()
    meshC.save(fname='regolith_vfG-{:2.2f}_vfV{:2.2f}_v750.iSALE'.format(v_grain,v_void),compress=True)
    
    meshA.fillAll(-1)
    meshB.fillAll(-1)
    meshC.fillAll(-1)
    first = False



