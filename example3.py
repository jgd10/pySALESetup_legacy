import pySALESetup as pss
import numpy as np

mesh = pss.Mesh(X=100,Y=600,cellsize=1.e-5)

x  = 5.e-4
yA = 4.5e-3
yB = 1.5e-3

grainA = pss.Grain(shape='file',File='grain_library/grain_arearatio-0.651.txt',eqr=40.)
grainB = pss.Grain(shape='file',File='grain_library/grain_arearatio-0.723.txt',eqr=40.)


grainA.place(x,yA,1,mesh)
grainB.place(x,yB,2,mesh)


vfrac = mesh.calcVol([1,2],frac=True)

print "Total volume fraction of particles is: {:3.3f} %".format(vfrac*100.)


mesh.fillAll(3)
mesh.plateVel(0.,3.e-3,1500.,axis=1)
mesh.plateVel(3.e-3,6.e-3,-1500.,axis=1)
mesh.top_and_tail()
mesh.matrixPorosity(3,50.)
mesh.viewMats()

#mesh1.save(fname='circle_base.iSALE',compress=True)

