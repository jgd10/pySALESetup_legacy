import pySALESetup as pss
import numpy as np
import matplotlib.pyplot as plt

mesh = pss.Mesh(X=500,Y=500, cellsize=1.)

Vertx = [0,50,25]
Verty = [250,250,300]
tri = pss.Apparatus(xcoords=Vertx,ycoords=Verty)

Vertx2 = [250,400,400,200]
Verty2 = [0,0,400,400]
sqr = pss.Apparatus(xcoords=Vertx2,ycoords=Verty2)

Vertx = [0,300,200,0]
Verty = [0,500,100,200]
ply = pss.Apparatus(xcoords=Vertx,ycoords=Verty)

#tri.rotate(0.1)
#tri.place(mesh,3)
ply.place(mesh,1)
sqr.place(mesh,2)
mesh.fillAll(4)

fig = plt.figure()
ax = fig.add_subplot(111,aspect='equal')
for KK in range(mesh.NoMats):
    matter = np.copy(mesh.materials[KK,:,:])*(KK+1)
    matter = np.ma.masked_where(matter==0.,matter)
    ax.pcolormesh(mesh.xi,mesh.yi,matter, cmap='Dark2',vmin=0,vmax=mesh.NoMats)

ax.plot(Vertx,Verty,marker='o')
ax.plot(Vertx2,Verty2,marker='o')
ax.set_xlim(0,mesh.x)
ax.set_ylim(0,mesh.y)
ax.set_xlabel('$x$ [cells]')
ax.set_ylabel('$y$ [cells]')
plt.show()
