import pySALESetup as pss
import numpy as np

g = []
grain = pss.Grain()
g.append(grain)
grain = pss.Grain()

g.append(grain)
mesh = pss.Mesh(X=600,Y=100)
mesh2 = pss.Mesh(X=600,Y=100)

#grain.view()



g[0].place(300,50,m=2,target=mesh)
g[1].place(300,60,m=3,target=mesh)

print g[0].dum,g[1].dum

group1 = pss.Ensemble(mesh)
for i in range(100):
    grain.insertRandomly(mesh,1)
    group1.add(grain)


group1.optimise_materials(np.array([1,2,3,4,5]))

for x,y,g,m in zip(group1.xc,group1.yc,group1.grains,group1.mats):
    g.place(x,y,m,mesh2)

mesh2.viewMats()
