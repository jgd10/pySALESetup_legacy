import pySALESetup as pss

# pySALESetup primarily handles the high-rez zone
# create Mesh object to manipulate
mesh = pss.Mesh(X=200,Y=240,cellsize=1.5875e-4)

# create a Grain instance to be the projectile 
# 'eqr' == equivalent radius of the circle with the same area 
projectile = pss.Grain(eqr=20.,shape='circle')

# place the Grain instance into the mesh 
# NB first cell in the mesh is at position 0, not 1
projectile.place(x=0.,y=(169.*mesh.cellsize),m=1,target=mesh)

# fill the mesh below the projectile with material
mesh.fillPlate(m=2,MMin=0.,MMax=149.*mesh.cellsize,axis=1)

# assign a velocity to the projectile
mesh.matVel(vel=-7.e3,mat=1)

EastExtZone = pss.ExtZone(mesh,D=50,side='East',fill=2,Vx=0,Vy=0)
SouthExtZone = pss.ExtZone(mesh,D=50,side='South',fill=2,Vx=0,Vy=0)

fullMesh = pss.CombinedMesh(mesh,E=EastExtZone,S=SouthExtZone,ExtendFromMesh=True)
# View the material and velocity meshes (optional)
#fullMesh.viewMats()
#fullMesh.viewVels()

fullMesh.save(compress=True)

# create a SetupInp instance to construct the asteroid,
# and additional, input files
S = pss.SetupInp()

# populate high-rez categories from the existing Mesh instance
S.populate_fromMesh(mesh,S=SouthExtZone,E=EastExtZone)

# Fill in remaining input file slots
# add extension zone to input file
S.MeshGeomParams['GRIDH'][2] = 50 
S.MeshGeomParams['GRIDV'][0] = 50  
# all fields in dictionaries are lists, regardless of length
S.MeshGeomParams['GRIDEXT'][0] = 1.05
S.MeshGeomParams['GRIDSPCM'] = 3.e-3
# set cylindrical geometry to true
S.MeshGeomParams['CYL'][0] = 1
S.GlobSetupParams['T_SURF'][0] = 293.
# Max absolute velocity allowed (prevents spurious cells slowing down the sim)
S.GlobSetupParams['VEL_CUT'][0] = 10.e3

# set the materials
S.AdditionalParams['PARMAT'] = ['al-1100','al-1100']

# check that all parameter updates are correct and consistent
S.checkAllParams()

# write parameters to file
# NB asteroid.inp and additional.inp must already exist
# pySALESetup only supports editing of parameters that are 
# directly related to its use; more functionality is planned
# however
S.write_astinp()
S.write_addinp()

