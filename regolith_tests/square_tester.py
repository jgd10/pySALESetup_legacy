import pySALESetup as pss
import numpy as np

mesh1 = pss.Mesh(X=100,Y=500,cellsize=.5e-5)


x  = .25e-3
yA = 1.875e-3
yB = 0.625e-3

theta = 45.

r = np.pi*theta/180.


R = [[-1.,-1.],
     [-1.,1.],
     [1.,1.],
     [1.,-1.]]


#grain = pss.Grain(shape='file',File='grain_arearatio-0.725.txt',eqr=20.,rot=r)
grain1 = pss.Grain(shape='polygon',poly_params=R,eqr=20.,rot=r)
grain2 = pss.Grain(shape='polygon',poly_params=R,eqr=20.,rot=r)

grain1.place(x,yA,1,mesh1)
grain2.place(x,yB,2,mesh1)

fill = mesh1.calcVol([1,2])
vfrac = fill/float(mesh1.Ncells)

print "Total volume fraction of particles is: {:3.3f} %".format(vfrac*100.)


mesh1.fillAll(3)
mesh1.plateVel(0.,1.25e-3,1500.,axis=1)
mesh1.plateVel(1.25e-3,2.5e-3,-1500.,axis=1)
mesh1.fillPlate(-1,2.49e-3,2.5e-3)
mesh1.fillPlate(-1,0.,0.01e-3)
mesh1.matrixPorosity(3,50.)
mesh1.viewMats()
#mesh1.viewVels()

mesh1.save(fname='square_rot{:1.2f}deg.iSALE'.format(theta))

