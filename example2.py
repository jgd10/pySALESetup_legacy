import pySALESetup as pss
import numpy as np

mesh = pss.Mesh(X=500,Y=500, cellsize=1.)

Vertx = [0,50,25]
Verty = [0,0,50]
tri = pss.Apparatus(xcoords=Vertx,ycoords=Verty)

Vertx = [0,250,250,0]
Verty = [0,0,250,250]
sqr = pss.Apparatus(xcoords=Vertx,ycoords=Verty)

Verts = np.array([[0,0],[300,300],[200,400],[0,200]])
print Verts[:,0],Verts[:,1]
ply = pss.Apparatus(xcoords=Verts[:,0],ycoords=Verts[:,1])

tri.rotate(0.1)
ply.place(mesh,1)
sqr.place(mesh,2)
tri.place(mesh,3)

mesh.viewMats()
