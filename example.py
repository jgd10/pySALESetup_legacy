import pySALESetup as pss
import numpy as np

g = []
#grain = pss.Grain()
#g.append(grain)

mesh = pss.Mesh(X=600,Y=100)
mesh2 = pss.Mesh(X=600,Y=100)

#grain.view()



#g[0].place(300,50,m=2,target=mesh)
#g[1].place(300,60,m=3,target=mesh)

import random
group1 = pss.Ensemble(mesh)
for i in range(100):
    rot = random.random()*np.pi
    ecc = random.random()
    grain = pss.Grain(shape='ellipse',rot=rot,elps_params=[10.,ecc])
    g.append(grain)
    grain.insertRandomly(mesh,1)
    group1.add(grain)


group1.optimise_materials(np.array([1,2,3,4,5]))

for x,y,g,m in zip(group1.xc,group1.yc,group1.grains,group1.mats):
    g.place(x,y,m,mesh2)

for x,y,g,m in zip(group1.xc,group1.yc,group1.grains,group1.mats):
    g.radius = g.radius+4
    g.mesh = pss.gen_ellipse(g.radius,g.angle,g.eccentricity) 
    g.place(x,y,6,mesh2)
mesh2.fillAll(m=7)

mesh1.plateVel(0.,1.5e-3,1500.,axis=1)
mesh1.plateVel(1.5e-3,3.e-3,-1500.,axis=1)

mesh2.viewMats()
