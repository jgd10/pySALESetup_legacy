import pySALESetup as pss
import numpy as np

mesh1 = pss.Mesh(X=200,Y=600,cellsize=.5e-5)

# square rot 00
poly_params=[[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]]
# square rot 45
poly_params=[[-1.,0.],[0.,1.],[1.,0.],[0.,-1.]]

#grain1 = pss.Grain(shape='polygon',eqr=10.,poly_params=poly_params)
#grain2 = pss.Grain(shape='polygon',eqr=10.,poly_params=poly_params)
#grain1 = pss.Grain(eqr=10.)
#grain2 = pss.Grain(eqr=10.)



# Test area A (square lattice) ymin = 2.e-3 ymax = 2.5e-3
XcoordsA = np.linspace(0.,1.e-3,5)
YcoordsA = np.linspace(2.e-3,3.e-3,5)

# Test area B (hexagonal lattice) ymin = .5e-3 ymax = 1.e-3
XcoordsB = np.linspace(0.,1.e-3,5)
YcoordsB = np.linspace(0.e-3,1.e-3,5)

c = 0
rot = 0
for yA,yB in zip(YcoordsA,YcoordsB):
    phi = rot*np.pi/180.
    print phi
    grain = pss.Grain(shape='polygon',eqr=10.,poly_params=poly_params,rot=phi)
    rot += 30.
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
        grain.place(xA,yA,1,mesh1)
        grain.place(xB,yB,2,mesh1)

fill = mesh1.calcVol([1,2])
vfrac = fill/float(mesh1.Ncells)

print "Total volume fraction of particles is: {:3.3f} %".format(vfrac*100.)


mesh1.fillAll(3)
mesh1.plateVel(0.,1.5e-3,1500.,axis=1)
mesh1.plateVel(1.5e-3,3.e-3,-1500.,axis=1)
mesh1.fillPlate(-1,2.988e-3,3.e-3)
mesh1.fillPlate(-1,0.,0.012e-3)
mesh1.matrixPorosity(3,50.)
mesh1.viewMats()
#mesh1.viewVels()

mesh1.save()

