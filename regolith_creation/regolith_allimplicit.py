import numpy as np
import pySALESetup as pss
import matplotlib.pyplot as plt
from math import ceil

meshA = pss.Mesh(X=500,Y=1200,cellsize=2.5e-6)
meshB = pss.Mesh(X=500,Y=1200,cellsize=2.5e-6)
meshA.label='A'
meshB.label='B'

# target volume (area) fraction
vfrac = 0.5


# Fill each domain with a matrix material; A+B will form a mesh, as will C+D
meshA.fillAll(1)
meshB.fillAll(1)

meshA.blanketVel(-1500.,axis=1)
meshB.blanketVel(+1500.,axis=1)

# combine the pairs of meshes
meshAB = pss.combine_meshes(meshA,meshB,axis=1)

# top and tail each mesh (delete top and bottom 3 rows of cells)
meshAB.top_and_tail()

# view final meshes
meshAB.viewVels()

# save final meshes as output files
meshAB.save(fname='regolith_PSD_fullimplicit.iSALE',compress=True)

# redo with new velocities if necessary.
#meshC.multiplyVels()
#meshC.save(fname='regolith_circles_v1500.iSALE',compress=True)
#meshC.multiplyVels()
#meshC.save(fname='regolith_circles_v750.iSALE',compress=True)

