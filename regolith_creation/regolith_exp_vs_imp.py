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
meshA.label='A'
meshB.label='B'

# target volume (area) fraction
vfrac = 0.5

# Store grain objects in list, 'grains'
grainsA = []
grainsB = []

# Minimum krubeim phi = min resolution (4 cppr)
# Max ... '' '' '' '' = max resolution (200 cppr) 
# Max res is one which still fits in the domain
minres  = 10
maxphi  = -np.log2(2*minres*2.5e-3)
minphi  = -np.log2(2*200*2.5e-3)

NA = 20
NB = 20
# Generate N phi values and equiv radii (in cells)
phiA = np.linspace(minphi,maxphi,NA)
phiB = np.linspace(minphi,maxphi,NB)

RsA = reverse_phi(phiA)/meshA.cellsize
RsB = reverse_phi(phiB)/meshB.cellsize
#RsA = ((DD_(phiA)*.5*1.e-3)/meshA.cellsize)
#RsB = ((DD_(phiB)*.5*1.e-3)/meshB.cellsize)
#RsC = ((DD_(phiC)*.5*1.e-3)/meshC.cellsize)
#RsD = ((DD_(phiD)*.5*1.e-3)/meshD.cellsize)

# interval over which to calculate number from pdf
# No. = |CDF(x+h) - CDF(x-h)| * no. of areas
hA = abs(phiA[1]-phiA[0])
hB = abs(phiB[1]-phiB[0])

# baseline mu and sigma fitted to a normal distribution
mu = 3.5960554191
sigma = 2.35633102167

# standard distro
SD = pss.SizeDistribution(func='normal',mu=mu,sigma=sigma*.3333333333333333)

print SD.details()

# target area that ALL particles should take up at end
target_area = float(meshA.Ncells*vfrac)

# Generate 4 libraries, one for each domain
# of grains, and record the expected frequency
# of each.
diff = 0
freqs = []
for r,p in zip(RsA,phiA):
    freq1 = SD.frequency(p,hA)*target_area
    freq2 = freq1/(np.pi*r**2.)
    freq = int(freq2)
    diff += (freq2-freq)*np.pi*r**2.
    freqs.append(freq)
    #print diff*np.pi*r**2.

ctr = 0
for r in RsA:
    test = diff/(np.pi*r**2.)
    if (1.-test)<=0.2:
        Rextra = r
        break
    ctr += 1

for r,freq in zip(RsA,freqs):
    if r == Rextra: freq+=1
    for f in range(freq):
        g = pss.Grain(r)
        grainsA.append(g)
        grainsB.append(g)


# library of grains has been generated, now place them into the mesh! 
groupA = pss.Ensemble(meshA,name='normaldistA')
groupB = pss.Ensemble(meshB,name='normaldistB')

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
except KeyboardInterrupt:
    pass


# optimise the material number distribution amongst grains
groupA.optimise_materials(np.array([1,2,3,4,5,6,7]))
groupB.optimise_materials(np.array([1,2,3,4,5,6,7]))

# wipe the mesh
meshA.fillAll(-1)
meshB.fillAll(-1)

# replace all grains with their new materials
for xA,yA,gA,mA in zip(groupA.xc,groupA.yc,groupA.grains,groupA.mats):
    gA.place(xA,yA,mA,meshA)
for xB,yB,gB,mB in zip(groupB.xc,groupB.yc,groupB.grains,groupB.mats):
    gB.place(xB,yB,mB,meshB)

meshA.fillAll(8)
meshB.fillAll(8)
v_voidA = meshA.VoidFracForTargetPorosity(8,bulk=0.5,final_por=0.5)
v_voidB = meshB.VoidFracForTargetPorosity(8,bulk=0.5,final_por=0.5)
GV = pss.Grain(eqr=4)
vfA = 0.
print v_voidA*100.,v_voidB*100.
while vfA < v_voidA:
    GV.insertRandomly(meshA, m=0,mattargets=[8])
    vfA = 1.-meshA.calcVol(frac=True)
    if vfA > v_voidA: break
vfB = 0.
while vfB < v_voidB:
    GV.insertRandomly(meshB, m=0,mattargets=[8])
    vfB = 1.-meshB.calcVol(frac=True)
    if vfB > v_voidB: break
# Fill each domain with a matrix material; A+B will form a mesh, as will C+D

# Calculate porosity required for each matrix
#meshA.matrixPorosity(8,0.5,Print=True) 
#print groupA.details()
#meshB.matrixPorosity(8,0.5,Print=True) 
#print groupB.details()

# Plot the particle size distribution created in each case
#groupA.plotPSD()

# Save the ensemble objects (pickle) for later use
groupA.save()
groupB.save()

# add a blanket velocity to each half
meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)

# combine the pairs of meshes
meshAB = pss.combine_meshes(meshA,meshB,axis=1)

# top and tail each mesh (delete top and bottom 3 rows of cells)
meshAB.top_and_tail()

# view final meshes
meshAB.viewMats()

# save final meshes as output files
meshAB.save(fname='regolith_PSD_minres{}cppr+voids_por0.50.iSALE'.format(minres),compress=True)

# redo with new velocities if necessary.
#meshC.multiplyVels()
#meshC.save(fname='regolith_circles_v1500.iSALE',compress=True)
#meshC.multiplyVels()
#meshC.save(fname='regolith_circles_v750.iSALE',compress=True)

