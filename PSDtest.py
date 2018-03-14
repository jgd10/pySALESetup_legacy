import pySALESetup as pss
import scipy.special as scsp
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
def normalPDF(x,mu,sg):
    pdf = np.exp(-0.5*((x-mu)/sg)**2.)/np.sqrt(2.*np.pi*sg**2.)
    return pdf

def normalCDF(x,mu,sg):
    cdf = .5*(1.+scsp.erf((x-mu)/np.sqrt(2.*sg**2.)))
    return cdf

def PDF(x):
    # The PDF
    D = 0.00936177062374 
    E = 0.0875
    L = 0.643
    F = D*L
    #pdf  = (1./a)*A*L*C*np.exp(L*x)/(B+C*np.exp(L*x))**2.
    pdf = 10.*F*np.exp(-L*x)/(E+np.exp(-L*x))**2.
    return pdf
def CDF(x):
    # The CDF
    #A = 2.908
    #B = 0.028
    #C = 0.320
    #a = 99.4
    #D = A*C/a
    #E = B/C
    D = 0.00936177062374*10.
    E = 0.0875
    L = 0.643
    print D,E

    return D/(E+np.exp(-L*x))
def reverse_phi(p):
    return (2**(-p))*.5*1.e-3

m = pss.Mesh(X=600,Y=1200,cellsize=2.5e-6)
G = pss.Ensemble(m)
SD = pss.SizeDistribution(func=CDF)

maxphi  = -np.log2(2*4*2.5e-3)
minphi  = -np.log2(2*2500*2.5e-3)
N = 100

# Generate N phi values and equiv radii (in cells)
phi = np.linspace(minphi,maxphi,N)

#ax.hist(phi,bins=20)
#print len(Counter(phi).keys())
#print len(np.unique(phi))
dp = abs(phi[0]-phi[1])

Area = np.float64(.5*m.Ncells)
#Nparts = 1000.
#for p in phi:
#    freq = SD.frequency(p,dp)*Area
#    r = reverse_phi(p)/m.cellsize
#    #print SD.frequency(p,dp)*Area,np.pi*r**2.
#    freq = int(freq/(np.pi*r**2.))
#    for f in range(freq):
#        g = pss.Grain(r)
#        G.add(g,x=0,y=0)

#print G.details()
#print G.PSDdetails()
#G.plotPSD()
#fitted normal dist to regolith data has: mu = 3.5960554191 and sigma = 2.35633102167
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
phi = np.linspace(-4,9,100)
ax1.plot(phi,PDF(phi))
ax2.plot(phi,CDF(phi))
mu = 3.5960554191
sigma = 2.35633102167
ax1.plot(phi,normalPDF(phi,mu,sigma))
ax2.plot(phi,normalCDF(phi,mu,sigma))
sigma /= 2.
ax1.plot(phi,normalPDF(phi,mu,sigma))
ax2.plot(phi,normalCDF(phi,mu,sigma))
sigma *= 4.
ax1.plot(phi,normalPDF(phi,mu,sigma))
ax2.plot(phi,normalCDF(phi,mu,sigma))
mu += 1.
sigma = 2.35633102167
ax1.plot(phi,normalPDF(phi,mu,sigma))
ax2.plot(phi,normalCDF(phi,mu,sigma))
plt.show()

