import pySALESetup as pss
import numpy as np

grain = pss.Grain()
mesh = pss.Mesh(X=600,Y=100)
mesh2 = pss.Mesh(X=600,Y=100)

#grain.view()



grain.place(300,50,m=2,target=mesh)
grain.place(300,60,m=3,target=mesh)

group1 = pss.Ensemble(mesh)
for i in range(100):
    grain.insertRandomly(mesh,1)
    group1.add(grain)


group1.optimise_materials(np.array([1,2,3,4,5]))

for x,y,g,m in zip(group1.xc,group1.yc,group1.grains,group1.mats):
    g.place(x,y,m,mesh2)

mesh2.viewMats()
