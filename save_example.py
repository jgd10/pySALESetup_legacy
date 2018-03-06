import pySALESetup as pss

m1 = pss.Mesh(X=50,Y=50)
E1 = pss.Ensemble(m1,name='E1')

for i in range(5):
    a = pss.Grain(eqr=10+i,name='test')
    a.insertRandomly(m1,i+1)
    E1.add(a)

m1.viewMats()
print E1.details()
print E1.xc
print E1.grains[-1].equiv_rad

E1.save()

E2 = pss.Ensemble(m1,name='E1',Reload=True)

print E2.details()
print E2.xc
print E2.grains[-1].equiv_rad

