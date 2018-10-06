"""
Save, and load, an Ensemble instance
"""
import pySALESetup as pss

# create a dummy mesh
m1 = pss.Mesh(X=50,Y=50)
# create a simple ensemble instance
E1 = pss.Ensemble(m1,name='E1')

# populate the Mesh and Ensemble
for i in range(5):
    a = pss.Grain(eqr=10+i,name='test')
    a.insertRandomly(m1,i+1)
    E1.add(a)

# View the mesh
m1.viewMats()
# print details of the Ensemble to screen
print E1.details()
# save the Ensemble
E1.save()

# load the ensemble
E1A = pss.Ensemble(m1,name='E1',Reload=True)
# check details have not changed!
print E1A.details()

