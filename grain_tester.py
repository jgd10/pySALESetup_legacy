import pySALESetup as pss
import numpy as np

mesh1 = pss.Mesh(X=1000,Y=600,cellsize=.5e-5)

#  rot 00
poly_params=[[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]]
# square rot 45
poly_params=[[-1.,0.],[-.2,.2],[0.,1.],[.2,.2],[1.,0.],[.2,-.2],[0.,-1.],[-.2,-.2]]

# Two test areas A and B in each test area one type of grain is tested in 50 different orientations
# in principle each grain is independent of all the others. How will each respond?

Ngrains = 20

Xcoords  = np.linspace(0.,10.e-3,Ngrains)
yA = np.float64(.75e-3)
yB = np.float64(2.125e-3)

rot = np.linspace(0,2*np.pi,Ngrains)

c = 0
for x,r in zip(Xcoords,rot):
    grain = pss.Grain(shape='polygon',poly_params=poly_params,eqr=20.,rot=r)
    grain.place(x,yA,1,mesh1)
    grain.place(x,yB,2,mesh1)

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

mesh1.save(fname='grain_test.iSALE')

