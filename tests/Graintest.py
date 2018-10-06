import pySALESetup as pss
import unittest
from math import pi
from numpy import linspace

class GrainClassTests(unittest.TestCase):
    """ Tests that all grain creation routines work as expected """
    def test_initialiseCircle(self):
        for i in range(4,11):
            g = pss.Grain(eqr=i,rot=0., shape='circle')
            actual_area = int(pi*i*i)
            perc_diff = (abs(actual_area-g.area)/actual_area)*100.
            self.assertTrue(perc_diff<10.)
    
    def test_initialiseEllipse(self):
        for i in range(4,11):
            for j in linspace(0,0.8,5):
                for k in linspace(0,pi*.5,5):
                    g = pss.Grain(eqr=i,rot=k, shape='ellipse',elps_eccen=j) 
                    actual_area = int(pi*i*i)
                    perc_diff = (abs(actual_area-g.area)/actual_area)*100.
                    self.assertTrue(perc_diff<10.)

    # less rigorous for shapes with corners because they suffer from rotational errors (30% required)
    def test_initialiseSquare(self):
        for i in range(4,11):
            for k in linspace(0,pi*.25,5):
                g = pss.Grain(eqr=i,rot=k, shape='polygon',poly_params=[[-1.,-1.],[-1.,1.],[1.,1.],[1.,-1.]]) 
                actual_area = int(pi*i*i)
                perc_diff = (abs(actual_area-g.area)/actual_area)*100.
                self.assertTrue(perc_diff<30.)
    
    def test_initialiseRectangle(self):
        for i in range(4,11):
            for k in linspace(0,pi*.25,5):
                g = pss.Grain(eqr=i,rot=k, shape='polygon',poly_params=[[-.5,-1.],[-.5,1.],[.5,1.],[.5,-1.]]) 
                actual_area = int(pi*i*i)
                perc_diff = (abs(actual_area-g.area)/actual_area)*100.
                self.assertTrue(perc_diff<30.)
    
    def test_initialiseTriangle(self):
        for i in range(4,11):
            for k in linspace(0,pi*.333,5):
                g = pss.Grain(eqr=i,rot=k, shape='polygon',poly_params=[[-1.,-1.],[0.,1.],[1.,-1.]]) 
                actual_area = int(pi*i*i)
                perc_diff = (abs(actual_area-g.area)/actual_area)*100.
                self.assertTrue(perc_diff<30.)
    
    def test_initialiseRhombus(self):
        for i in range(4,11):
            for k in linspace(0,pi*.333,5):
                g = pss.Grain(eqr=i,rot=k, shape='polygon',poly_params=[[-1.,-1.],[-0.5,1.],[1.,1.],[0.5,-1.]]) 
                actual_area = int(pi*i*i)
                perc_diff = (abs(actual_area-g.area)/actual_area)*100.
                self.assertTrue(perc_diff<30.)

    def test_cropGrain(self):
        g = pss.Grain(eqr=10.)
        Is, Js, i_, j_ = g._cropGrain(0,0,100,100)
        self.assertTrue(Is[0]>=0)
        self.assertTrue(Js[0]>=0)
        self.assertTrue(i_[0]>=0)
        self.assertTrue(j_[0]>=0)
        diffI = Is[1]-Is[0]
        diffJ = Js[1]-Js[0]
        self.assertTrue(diffI==diffJ)
        maxI = max(Is)
        maxJ = max(Js)
        self.assertTrue(maxI<=g.Px)
        self.assertTrue(maxJ<=g.Py)
        diffi_ = i_[1]-i_[0]
        diffj_ = j_[1]-j_[0]
        self.assertTrue(diffi_==diffj_)
        self.assertTrue(diffi_==diffI)
        self.assertTrue(diffj_==diffJ)
        maxi_ = max(i_)
        maxj_ = max(j_)
        self.assertTrue(maxi_<=100)
        self.assertTrue(maxj_<=100)

class EnsembleClassTests(unittest.TestCase):
    """ Tests that all Ensemble routines work as expected """
    def test_initialiseEnsemble(self):
        M = pss.Mesh()
        E = pss.Ensemble(M)
        self.assertTrue(E.number==0)
        self.assertTrue(len(E.grains)==0)

    def test_addGrain(self):
        M = pss.Mesh()
        E = pss.Ensemble(M)
        g = pss.Grain()
        g.place(0.,0.,1,M)
        E.add(g)
        self.assertTrue(E.number==1)
        self.assertTrue(len(E.grains)==1)
        self.assertTrue(E.xc[0]==0.)
        self.assertTrue(E.yc[0]==0.)
        self.assertTrue(E.mats[0]==1)
        self.assertTrue(E.shapes[0]=='circle')

    def test_addGrain(self):
        M = pss.Mesh()
        E = pss.Ensemble(M)
        g = pss.Grain()
        g.place(0.,0.,1,M)
        E.add(g)
        self.assertTrue(E.number==1)
        self.assertTrue(len(E.grains)==1)
        self.assertTrue(E.xc[0]==0.)
        self.assertTrue(E.yc[0]==0.)
        self.assertTrue(E.mats[0]==1)
        self.assertTrue(E.shapes[0]=='circle')

    def test_DelGrain(self):
        M = pss.Mesh()
        E = pss.Ensemble(M)
        g = pss.Grain()
        g.place(0.,0.,1,M)
        E.add(g)
        E.add(g)
        E.Del(g)
        self.assertTrue(E.number==1)
        self.assertTrue(len(E.grains)==1)
        self.assertTrue(len(E.phi)==1.)



if __name__ == '__main__':
    unittest.main()
