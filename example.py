import pySALESetup as pss
import numpy as np

g = []
mesh = pss.Mesh(X=600,Y=100)
mesh2 = pss.Mesh(X=600,Y=100)

#grain.view()

import random
group1 = pss.Ensemble(mesh)
for i in range(100):
    rot = random.random()*np.pi
    ecc = random.random()*0.5
    grain = pss.Grain(shape='ellipse',rot=rot,elps_params=[10.,ecc])
    g.append(grain)
    if i < 50: 
        grain.insertRandomly(mesh,1)
    else:
        grain.insertRandomwalk(mesh,1)
    group1.add(grain)


group1.optimise_materials(np.array([1,2,3,4,5,6,7]))

for x,y,g,m in zip(group1.xc,group1.yc,group1.grains,group1.mats):
    g.place(x,y,m,mesh2)

for x,y,g,m in zip(group1.xc,group1.yc,group1.grains,group1.mats):
    g.radius = g.radius+4
    g.mesh = pss.gen_ellipse(g.radius,g.angle,g.eccentricity) 
    g.place(x,y,8,mesh2)
mesh2.fillAll(m=9)

#mesh2.plateVel(0.,1.5e-3,1500.,axis=1)
#mesh2.plateVel(1.5e-3,3.e-3,-1500.,axis=1)

mesh2.viewMats()
