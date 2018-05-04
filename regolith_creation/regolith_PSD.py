import numpy as np
import pySALESetup as pss
import matplotlib.pyplot as plt
from math import ceil

# Cumulative Distribution Function
def CDF(x):
    # The CDF
    A = 2.908
    B = 0.028
    C = 0.320
    L = 0.643
    a = 99.4
    return (1./a)*A/(B+C*np.exp(-L*x))

# Population Distribution Function
def PDF(x):
    # The PDF
    A = 2.908
    B = 0.028
    C = 0.320
    L = 0.643
    a = 99.4
    pdf  = (1./a)*A*L*C*np.exp(L*x)/(B+C*np.exp(L*x))**2.
    return pdf

# legacy function, this is now integrated into pySALESetup
def lunar_pdf(x,LB_tol,UB_tol):                                                   
    # Integrate PDF at a value of x
    P     = abs(CDF(x+UB_tol) - CDF(x-LB_tol))
    return P

# Convert to krumbein phi
def PHI_(x):
    """
    x must be in SI (metres) for this function to work
    PHI is only calculated correctly when the arg is mm
    """
    return -1.*np.log2(x*1000.)

# reverse of krumbein phi
def DD_(x):
    return 2.**(-x)

# reverse of krumbein phi
# duplicate of above; except converts to metres
# and returns radius, not diameter
def reverse_phi(p):
    return (2**(-p))*.5*1.e-3

# Top and bottom mesh created separately
# Create four meshes
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
maxphi  = -np.log2(2*4*2.5e-3)
minphi  = -np.log2(2*200*2.5e-3)

NA = 20
NB = 20
NC = 20
ND = 20
# Generate N phi values and equiv radii (in cells)
phiA = np.linspace(minphi,maxphi,NA)
phiB = np.linspace(minphi,maxphi,NB)
phiC = np.linspace(minphi,maxphi,NC)
phiD = np.linspace(minphi,maxphi,ND)

RsA = reverse_phi(phiA)/meshA.cellsize
RsB = reverse_phi(phiB)/meshB.cellsize
RsC = reverse_phi(phiC)/meshC.cellsize
RsD = reverse_phi(phiD)/meshD.cellsize
#RsA = ((DD_(phiA)*.5*1.e-3)/meshA.cellsize)
#RsB = ((DD_(phiB)*.5*1.e-3)/meshB.cellsize)
#RsC = ((DD_(phiC)*.5*1.e-3)/meshC.cellsize)
#RsD = ((DD_(phiD)*.5*1.e-3)/meshD.cellsize)

# interval over which to calculate number from pdf
# No. = |CDF(x+h) - CDF(x-h)| * no. of areas
hA = abs(phiA[1]-phiA[0])
hB = abs(phiB[1]-phiB[0])
hC = abs(phiC[1]-phiC[0])
hD = abs(phiD[1]-phiD[0])

# baseline mu and sigma fitted to a normal distribution
mu = 3.5960554191
sigma = 2.35633102167

# standard distro
SDA = pss.SizeDistribution(func='normal',mu=mu,sigma=sigma)
# twice the std
sig2 = sigma * 2.
SDB = pss.SizeDistribution(func='normal',mu=mu,sigma=sig2)
# half the std
sig3 = sigma * .5
SDC = pss.SizeDistribution(func='normal',mu=mu,sigma=sig3)
# const std, increment mean by 1
mu2 = mu + 1.
SDD = pss.SizeDistribution(func='normal',mu=mu2,sigma=sigma)

print SDA.details()
print SDB.details()
print SDC.details()
print SDD.details()

# target area that ALL particles should take up at end
target_area = float(meshA.Ncells*vfrac)

# Generate 4 libraries, one for each domain
# of grains, and record the expected frequency
# of each.
for rA,pA in zip(RsA,phiA):
    freq = SDA.frequency(pA,hA)*target_area
    freq = int(freq/(np.pi*rA**2.))
    for f in range(freq):
        gA = pss.Grain(rA)
        grainsA.append(gA)
for rB,pB in zip(RsB,phiB):
    freq = SDB.frequency(pB,hB)*target_area
    freq = int(freq/(np.pi*rB**2.))
    for f in range(freq):
        gB = pss.Grain(rB)
        grainsB.append(gB)
for rC,pC in zip(RsC,phiC):
    freq = SDC.frequency(pC,hC)*target_area
    freq = int(freq/(np.pi*rC**2.))
    for f in range(freq):
        gC = pss.Grain(rC)
        grainsC.append(gC)
for rD,pD in zip(RsD,phiD):
    freq = SDD.frequency(pD,hD)*target_area
    freq = int(freq/(np.pi*rD**2.))
    for f in range(freq):
        gD = pss.Grain(rD)
        grainsD.append(gD)

# library of grains has been generated, now place them into the mesh! 
groupA = pss.Ensemble(meshA,name='normaldistA_mu=3.6_sg=2.4')
groupB = pss.Ensemble(meshB,name='normaldistB_mu=3.6_sg=4.8')
groupC = pss.Ensemble(meshC,name='normaldistC_mu=3.6_sg=1.2')
groupD = pss.Ensemble(meshD,name='normaldistD_mu=4.6_sg=2.4')

# place them in, but don't worry if not possible to fit all.
# allow for keyboard interrupt if there's a problem.
try:
    i = 0
    for gA in grainsA:
        gA.insertRandomly(meshA, m=1)
        groupA.add(gA,gA.x,gA.y)
    for gB in grainsB:
        gB.insertRandomly(meshB, m=1)
        groupB.add(gB,gB.x,gB.y)
    for gC in grainsC:
        gC.insertRandomly(meshC, m=1)
        groupC.add(gC,gC.x,gC.y)
    for gD in grainsD:
        gD.insertRandomly(meshD, m=1)
        groupD.add(gD,gD.x,gD.y)
except KeyboardInterrupt:
    pass


# optimise the material number distribution amongst grains
groupA.optimise_materials(np.array([1,2,3,4,5,6,7]))
groupB.optimise_materials(np.array([1,2,3,4,5,6,7]))
groupC.optimise_materials(np.array([1,2,3,4,5,6,7]))
groupD.optimise_materials(np.array([1,2,3,4,5,6,7]))

# wipe the mesh
meshA.fillAll(-1)
meshB.fillAll(-1)
meshC.fillAll(-1)
meshD.fillAll(-1)

# replace all grains with their new materials
for xA,yA,gA,mA in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats):
    gA.place(xA,yA,mA,meshA)
for xB,yB,gB,mB in zip(groupB.xc,groupB.yc,groupB.grains,groupB.mats):
    gB.place(xB,yB,mB,meshB)
for xC,yC,gC,mC in zip(groupC.xc,groupC.yc,groupC.grains,groupC.mats):
    gC.place(xC,yC,mC,meshC)
for xD,yD,gD,mD in zip(groupD.xc,groupD.yc,groupD.grains,groupD.mats):
    gD.place(xD,yD,mD,meshD)

# Fill each domain with a matrix material; A+B will form a mesh, as will C+D
meshA.fillAll(8)
meshB.fillAll(9)
meshC.fillAll(8)
meshD.fillAll(9)

# Calculate porosity required for each matrix
meshA.matrixPorosity(8,0.5,Print=True) 
print groupA.details()
meshB.matrixPorosity(9,0.5,Print=True) 
print groupB.details()
meshC.matrixPorosity(8,0.5,Print=True) 
print groupC.details()
meshD.matrixPorosity(9,0.5,Print=True) 
print groupD.details()

# Plot the particle size distribution created in each case
groupA.plotPSD()
groupB.plotPSD()
groupC.plotPSD()
groupD.plotPSD()

# Save the ensemble objects (pickle) for later use
groupA.save()
groupB.save()
groupC.save()
groupD.save()

# add a blanket velocity to each half
meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)
meshC.blanketVel(-1500.,axis=1)
meshD.blanketVel(+1500.,axis=1)

# combine the pairs of meshes
meshAB = pss.combine_meshes(meshA,meshB,axis=1)
meshCD = pss.combine_meshes(meshC,meshD,axis=1)

# top and tail each mesh (delete top and bottom 3 rows of cells)
meshAB.top_and_tail()
meshCD.top_and_tail()

# view each individual mesh
meshA.viewMats()
meshB.viewMats()
meshC.viewMats()
meshD.viewMats()

# view final meshes
meshAB.viewMats()
meshCD.viewMats()

# save final meshes as output files
meshAB.save(fname='regolith_PSD_AB.iSALE',compress=True)
meshCD.save(fname='regolith_PSD_CD.iSALE',compress=True)

# redo with new velocities if necessary.
#meshC.multiplyVels()
#meshC.save(fname='regolith_circles_v1500.iSALE',compress=True)
#meshC.multiplyVels()
#meshC.save(fname='regolith_circles_v750.iSALE',compress=True)

