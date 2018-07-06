import numpy as np
import pySALESetup as pss
import matplotlib.pyplot as plt
from math import ceil
import copy
import random
random.seed(42)
plt.rcParams['text.usetex']=True

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
meshA = pss.Mesh(X=500,Y=500,cellsize=2.5e-6)
meshA.label='A'

# target volume (area) fraction
vfrac = 0.5

# Store grain objects in list, 'grains'
grainsA = []
grainsB = []

# Minimum krubeim phi = min resolution (4 cppr)
# Max ... '' '' '' '' = max resolution (200 cppr) 
# Max res is one which still fits in the domain
minres  = 10
maxphi  = -np.log2(10*2.5e-3)
minphi  = -np.log2(90*2.5e-3)

NA = 3
NB = 3
# Generate N phi values and equiv radii (in cells)
phiA = np.linspace(minphi,maxphi,NA)

RsA = reverse_phi(phiA)/meshA.cellsize

# interval over which to calculate number from pdf
# No. = |CDF(x+h) - CDF(x-h)| * no. of areas
hA = abs(phiA[1]-phiA[0])

# baseline mu and sigma fitted to a normal distribution
mu = 3.5960554191
sigma = 2.35633102167

# standard distro
SD = pss.SizeDistribution(func='normal',mu=mu,sigma=sigma)

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



# library of grains has been generated, now place them into the mesh! 
groupA = pss.Ensemble(meshA,name='normaldistA')

Rs = copy.deepcopy(RsA)
Fr = copy.deepcopy(freqs)
# place them in, but don't worry if not possible to fit all.
# allow for keyboard interrupt if there's a problem.
fig = plt.figure(figsize=(8,4))
fig1 = plt.figure(figsize=(6,4))
ax1a = fig1.add_subplot(131,aspect='equal')
ax2a = fig1.add_subplot(132,aspect='equal')
ax3a = fig1.add_subplot(133,aspect='equal')
ax1a.axis('off')
ax2a.axis('off')
ax3a.axis('off')
ax1 = fig.add_subplot(141,aspect='equal')
ax2 = fig.add_subplot(142,aspect='equal')
ax3 = fig.add_subplot(143,aspect='equal')
ax4 = fig.add_subplot(144,aspect='equal')
group = []
for ax in [ax1,ax2,ax3,ax4]:
    ctr = 0
    m = 0
    for r,freq in zip(Rs,Fr):
        if r == Rextra: freq+=1
        m += 1
        for f in range(freq):
            if ax==ax1:
                g = pss.Grain(r)
                g.insertRandomly(meshA, m=m)
                group.append(g)
            else:
                g = group[ctr]
                g.place(g.x,g.y,m,meshA)
                ctr += 1
    if ax != ax1: meshA.fillAll(m+1)
    for KK in range(meshA.NoMats):
        matter = np.copy(meshA.materials[KK,:,:])*(KK+1)
        matter = np.ma.masked_where(matter==0.,matter)
        if KK == 2 and ax==ax1:
            ax1a.pcolormesh(meshA.xi,meshA.yi,matter, cmap='terrain',vmin=1,vmax=meshA.NoMats+1)
        if KK == 2 and ax==ax2:
            ax3a.pcolormesh(meshA.xi,meshA.yi,matter, cmap='terrain',vmin=1,vmax=meshA.NoMats+1)
        ax.pcolormesh(meshA.xi,meshA.yi,matter, cmap='terrain',vmin=1,vmax=meshA.NoMats+1)
    ax.axis('off')
    Rs = list(Rs[:-1])
    Fr = list(Fr[:-1])
    meshA.fillAll(-1)
ax1.set_title('Fully Explicit')
ax2.set_title('Semi-Explicit')
ax3.set_title('Semi-Explicit')
ax4.set_title('Fully Implicit')
bbox_props = dict(boxstyle="darrow", fc='w', ec="k", lw=2)
ax1.text(0.52, 0.2, "Explicit | Implicit", ha="center", va="center",
                    size=15,bbox=bbox_props,transform=fig.transFigure)
bbox_props = dict(boxstyle="rarrow", fc='w', ec="k", lw=2)
ax2a.text(0.5, 0.5, "Parameterisation\nassumes\nuniform density", ha="center", va="center",
                    size=12,bbox=bbox_props)
#fig.tight_layout()
#fig.savefig('explicit_implicit_demonstration.png',dpi=300,transparent=True)
fig1.savefig('matrixunifromity_assum_demonstration.png',dpi=300,transparent=True)
#plt.show()


# view final meshes
#meshA.viewMats()

