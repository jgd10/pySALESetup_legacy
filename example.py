import pySALESetup as pss
import numpy as np
import random
"""
This is a simple script that creates a particle bed of elliptical grains
with ice shrouds covering them. It creates two meshes and then merges them
to make one mirror-impact setup.
"""

# create two identical meshes
mesh1 = pss.Mesh(X=400,Y=200)
mesh2 = pss.Mesh(X=400,Y=200)
# use the meshes to make two ensembles;
# one ensemble for each domain
group1 = pss.Ensemble(mesh1)
group2 = pss.Ensemble(mesh2)
# we want to create (and place) 200 
# different particles in this case
for i in range(200):
    # generate some random params for each ellipse
    rot = random.random()*np.pi
    # force eccentricities to be between 0.5 and 0.8
    # (just for this script, so that they look elliptical)
    ecc = min(random.random()*0.75+0.5,0.8)
    # Create a Grain instance! elps_params = [major radius in cells, eccentricity]
    # default is 10 cells equivalent radius
    grain = pss.Grain(shape='ellipse',rot=rot,elps_params=[10.,ecc])
    # place the first 100 grains randomly into free spaces
    # for now we are just using one material.
    if i < 100: 
        # insert grain instance into mesh1
        grain.insertRandomly(mesh1,1)
        # add grain instance to group1
        # NB do this before placing grain again! grain.x, grain.y, etc.
        # are changed after every placement
        group1.add(grain)
        # insert grain instance into mesh2
        grain.insertRandomly(mesh2,1)
        # add grain instance to group2
        group2.add(grain)
    else:
        # place 2nd half of the grains into mesh such that each is in
        # contact with at least one other grain (randomWalk's purpose)
        grain.insertRandomwalk(mesh1,1)
        group1.add(grain)
        grain.insertRandomwalk(mesh2,1)
        group2.add(grain)

# Calculate the optimal material number distribution for 
# each group individually.
group1.optimise_materials(np.array([1,2,3,4,5,6,7]))
group2.optimise_materials(np.array([1,2,3,4,5,6,7]))

# delete all material in each domain
mesh1.fillAll(-1)
mesh2.fillAll(-1)

# use information stored in group1 and group2 to repopulate domain
# except NOW we can use the optimal materials from optimise_materials!
for x,y,g,m in zip(group1.xc,group1.yc,group1.grains,group1.mats):
    g.place(x,y,m,mesh1)
for x,y,g,m in zip(group2.xc,group2.yc,group2.grains,group2.mats):
    g.place(x,y,m,mesh2)

# add an elliptical shroud over each grain
for x1,y1,g1,x2,y2,g2 in zip(group1.xc,group1.yc,group1.grains,group2.xc,group2.yc,group2.grains):
    # increase grain radius
    r1 = g1.radius+4
    # generate new grain with new radius
    g1.mesh = pss.grainfromEllipse(r1,g1.angle,g1.eccentricity) 
    # place new grain in mesh1
    g1.place(x1,y1,8,mesh1)
    
    #repeat for mesh2
    r2 = g2.radius+4
    g2.mesh = pss.grainfromEllipse(r2,g2.angle,g2.eccentricity) 
    g2.place(x2,y2,8,mesh2)


# Fill all remaining space with material 9
mesh2.fillAll(m=9)
mesh1.fillAll(m=9)

# Give each mesh an opposing velocity (mirror impact)
mesh1.blanketVel(-500.)
mesh2.blanketVel(+500.)

# Combine the two meshes vertically
mesh3 = pss.combine_meshes(mesh1,mesh2,axis=1)

# View our handiwork!
mesh3.viewVels()
mesh3.viewMats(save=True)

# Save the new mesh, can specify a filename but defaults to
# meso_m.iSALE or meso_m.iSALE.gz if compress = True.
# (compressed input files are also taken as input by iSALE)
print "compressing..."
mesh3.save(compress=True)
print "saved"
