import pySALESetup as pss
import numpy as np

mesh = pss.Mesh(X=50,Y=50, cellsize=1.)

Vertx = [0,50,25]
Verty = [0,0,50]

tri = pss.Apparatus(xcoords=Vertx,ycoords=Verty)

tri.rotate(0.1)
tri.place(mesh,1)

mesh.viewMats()
