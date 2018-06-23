import numpy as np
import pySALESetup as pss
import matplotlib.pyplot as plt
from math import ceil


meshA = pss.Mesh(X=500,Y=1200,cellsize=2.5e-6)
meshB = pss.Mesh(X=500,Y=1200,cellsize=2.5e-6)

# target volume (area) fraction
vfrac = 0.5

# Store grain objects in list, 'grains'
grainsA = []
grainsB = []

gLarge = pss.Grain(25.)
gSmall = pss.Grain(5.)

groupA = pss.Ensemble(meshA,name='bimodaldistA')
groupB = pss.Ensemble(meshB,name='bimodaldistB')

vfracA = 0.
vfracB = 0.

try:
    while vfracA < 0.3:
        gLarge.insertRandomly(meshA,m=1)
        groupA.add(gLarge,gLarge.x,gLarge.y)
        vfracA = meshA.vfrac()
    while vfracB < 0.3:
        gLarge.insertRandomly(meshB,m=1)
        groupB.add(gLarge,gLarge.x,gLarge.y)
        vfracB = meshB.vfrac()
except KeyboardInterrupt:
    pass

groupA.optimise_materials(np.array([1,2,3,4,5,6,7,8]),populate=True)
groupB.optimise_materials(np.array([1,2,3,4,5,6,7,8]),populate=True)

meshA.fillAll(9)
meshB.fillAll(9)
meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)
# combine the pairs of meshes
meshAB = pss.combine_meshes(meshA,meshB,axis=1)
# top and tail each mesh (delete top and bottom 3 rows of cells)
meshAB.top_and_tail()
meshAB.viewMats()
meshAB.save(fname='regolith_bimodalimplicit.iSALE',compress=True)
meshA.deleteMat(9)
meshB.deleteMat(9)

try:
    while vfracA < 0.5:
        gSmall.insertRandomly(meshA,m=1)
        groupA.add(gSmall,gSmall.x,gSmall.y)
        vfracA = meshA.vfrac()
    while vfracB < 0.5:
        gSmall.insertRandomly(meshB,m=1)
        groupB.add(gSmall,gSmall.x,gSmall.y)
        vfracB = meshB.vfrac()
except KeyboardInterrupt:
    pass

groupA.optimise_materials(np.array([1,2,3,4,5,6,7,8]),populate=True)
groupB.optimise_materials(np.array([1,2,3,4,5,6,7,8]),populate=True)

meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)
# combine the pairs of meshes
meshAB = pss.combine_meshes(meshA,meshB,axis=1)
# top and tail each mesh (delete top and bottom 3 rows of cells)
meshAB.top_and_tail()
meshAB.viewMats()
meshAB.save(fname='regolith_bimodalexplicit.iSALE',compress=True)
