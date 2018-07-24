import random
import warnings
import numpy as np
from PIL import Image
from math import ceil
import cPickle as pickle
import scipy.special as scsp
from scipy import stats as scst
import matplotlib.pyplot as plt
import matplotlib.path   as mpath
from collections import Counter, OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pss_functions import *
from pss_domainclasses import *
from pss_grainclasses import *

class Apparatus:
    """
    Polygon object inserted into a Mesh instance. Example of apparatus
    """
    def __init__(self,xcoords,ycoords):
        """
        This function fills all cells within a polygon defined by the vertices in arrays 
        xcoords and ycoords; coords MUST be in clockwise order!
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ~if mat == -1. this fills the cells with VOID and overwrites everything.~
        ~and additionally sets all velocities in those cells to zero            ~
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        # last point MUST be identical to first; append to end if necessary
        assert len(xcoords)==len(ycoords), "ERROR: xcoords and ycoords must have same length!"

        if xcoords[0] != xcoords[-1] or ycoords[0] != ycoords[-1]:
            xcoords = np.append(xcoords,xcoords[0])
            ycoords = np.append(ycoords,ycoords[0])
        self.x = np.array(xcoords)
        self.y = np.array(ycoords)

        # find the box that entirely encompasses the polygon
        self.L = np.array([np.amin(xcoords),np.amax(xcoords)])
        self.T = np.array([np.amin(ycoords),np.amax(ycoords)])
        

    def rotate(self, angle):
        ct    = np.cos(angle)
        st    = np.sin(angle)
        self.x = self.x*ct - self.y*st
        self.y = self.x*st + self.y*ct

        
    def place(self,target,m,overwrite_mats=None):
        """
        inserts the object into the target mesh using the coords stored in Apparatus instance.
        Preference is given to materials already present in the target mesh and these are 
        not overwritten
        if m == -1 then this erases all material it is placed on. 
        To only overwrite certain materials set 
        m = 0 and specify which to overwrite in overwrite_mats. 
        Any other materials will be left untouched.

        Args:
            target:         Mesh
            m:              int; material number, -1 == void
            overwrite_mats: list; if not None then materials to be overwritten.
        """
        if m == 0: assert overwrite_mats is not None, "ERROR: if placing void, and specifying material to overwrite, some material MUST be specified!"
        # if item is a rectangle, go to faster rectangle routine
        if len(self.x) == 5 and len(np.unique(self.x)) == 2 and len(np.unique(self.y)) == 2:
            self._place_rectangle(target,m)
        else:
        # else use point-in-polygon method available to matplotlib
            # create path from coords of shape
            self.path = mpath.Path(np.column_stack((self.x,self.y)))
            # Check all coordinates in mesh and return bool of which in shape and which not
            inshape = self.path.contains_points(zip(target.xi.flatten(),target.yi.flatten()))
            # This array is flattened so reshape to correct dims
            inshape = inshape.reshape(target.x,target.y)
            
            # if material = void then delete the matter in shape
            if m == -1:
                # select the necessary material using new arrays of indices
                materials[:][inshape] *= 0.
                target.VX[inshape] *= 0.
                target.VY[inshape] *= 0.
                target.mesh[inshape] *= 0.
            # if m is a valid material (m>0) then fill that material as in prev place functions
            elif m > 0:
                temp_materials = np.copy(target.materials[int(m)-1])
                temp_materials[inshape*(np.sum(target.materials,axis=0)<1.)] = 1. #- np.sum(materials,axis=0)  
                temp_2 = np.sum(target.materials,axis=0)*temp_materials
                temp_materials -= temp_2
                target.materials[int(m)-1] += temp_materials
            # if m set to 0 and materials specified to be overwritten, then delete only those.
            elif overwrite_mats is not None and m == 0:
                for mm in overwrite_mats:
                    materials[mm-1][inshape] *= 0.
                # Check vels after this because not all matter in the shape may necessarily have been deleted
                target.checkVels()


            return
    
    def _place_rectangle(self,target,m):
        """
        places a rectangle of material in the target at provided coords 
        if mat == -1. this fills the cells with VOID and overwrites everything.
        and additionally sets all velocities in those cells to zero            
        Should only be called if self.place() is placing an unrotated rectangle;
        this method is significantly faster than the default and rectangles are very
        common.
        
        Args:
            target: Mesh
            m:      int; material number, -1 == void
        """
        X1 = np.amin(self.x)
        X2 = np.amax(self.x)
        
        Y1 = np.amin(self.y)
        Y2 = np.amax(self.y)
        
        inrectangle = (target.xx<=X2)*(target.xx>=X1)*(target.yy<=Y2)*(target.yy>=Y1)
        if m == -1:
            for mm in range(9):
                materials[mm][inrectangle] *= 0.
            target.VX[inrectangle] *= 0.
            target.VY[inrectangle] *= 0.
        else:
            temp_materials = np.copy(target.materials[int(m)-1])
            temp_materials[inrectangle*(np.sum(target.materials,axis=0)<1.)] = 1.
            temp_2 = np.sum(target.materials,axis=0)*temp_materials
            temp_materials -= temp_2
            target.materials[int(m)-1] += temp_materials
        return
