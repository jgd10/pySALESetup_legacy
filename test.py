import pySALESetup as pss
import numpy as np

mesh1 = pss.Mesh(X=50,Y=150,cellsize=2.e-5)
grain1 = pss.Grain(eqr=5.)
grain2 = pss.Grain(eqr=5.)



# Test area A (square lattice) ymin = 2.e-3 ymax = 2.5e-3
XcoordsA = np.linspace(0.,1.e-3,5)
YcoordsA = np.linspace(2.e-3,2.5e-3,3)

# Test area B (hexagonal lattice) ymin = .5e-3 ymax = 1.e-3
XcoordsB = np.linspace(0.,1.e-3,5)
YcoordsB = np.linspace(.5e-3,1.e-3,3)

c = 0
for yA,yB in zip(YcoordsA,YcoordsB):
    if c == 0:
        c = 1
    else:
        c = 0
    for xA,xB in zip(XcoordsA,XcoordsB):
        if c == 0:
            xB += .125e-3
        elif c == 1:
            pass
        #xB -= .125e-3
        grain1.place(xA,yA,1,mesh1)
        grain2.place(xB,yB,2,mesh1)

fill = mesh1.calcVol([1,2])
vfrac = fill/float(mesh1.Ncells)

print "Total volume fraction of particles is: {:3.3f} %".format(vfrac*100.)


mesh1.fillAll(3)
print mesh1.matrixPorosity(3,50.)

#mesh1.viewMats()

mesh1.plateVel(0.,1.5e-3,1500.,axis=1)
mesh1.plateVel(1.5e-3,3.e-3,-1500.,axis=1)

#mesh1.viewVels()

mesh1.save()

