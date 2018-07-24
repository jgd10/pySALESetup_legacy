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
from pss_grainclasses import *
from pss_objectclasses import *

class Mesh:
    """
    This is the domain class and it tracks all materials placed into it. Main features
    are the material fields--NB. of materials is ALWAYS 9 (the maximum). When saving,
    if a material is not used it is not included in the output file--, the velocity fields
    which include both x and y component fields, and the 'mesh' field. This acts like a test
    domain with only one material field and is mostly used internally.
    """
    def __init__(self,X=500,Y=500,cellsize=2.e-6,mixed=False,label='None'):
        """
        Initialise the Mesh class. Defaults are typical for mesoscale setups which this module
        was originally designed for.

        Args:
            X:        int
            Y:        int
            cellsize: float; equivalent to GRIDSPC in iSALE
            mixed:    bool
            label:    str
        """
        self.x = X
        self.y = Y
        self.Ncells = X*Y
        self.width = X*cellsize
        self.height = Y*cellsize
        self.area = self.Ncells*cellsize
        self.xc = np.arange(X)+0.5
        self.yc = np.arange(Y)+0.5
        self.yi, self.xi = np.meshgrid(self.yc,self.xc)
        self.xx, self.yy = self.xi*cellsize,self.yi*cellsize
        self.mesh = np.zeros((X,Y))
        self.materials = np.zeros((9,X,Y))
        self.VX = np.zeros((X,Y))
        self.VY = np.zeros((X,Y))
        self.cellsize = cellsize
        self.NoMats = 9
        self.mats = range(1,9+1)
        self.mixed = mixed
        self.name = label


    def checkVels(self):
        """
        Ensures no void cells have velocities.
        """
        total = np.sum(self.materials,axis=0)
        # make sure that all void cells have no velocity
        self.VX[total==0.] = 0.
        self.VY[total==0.] = 0.
    
    def max_porosity_variation(self,partitions=2):
        """
        Function that finds the largest varition in porosity across the entire mesh. 
        This will give incorrect answers when the mesh is not purely granular.
        returns the maximum difference between two partitions of the same orientation.
        """
        mesh = np.sum(self.materials,axis=0)
        mesh[mesh>1.] = 1.
        pores = 1.-mesh
    
        # create arrays to store vert and horiz partition porosities
        pores_T = np.ones(partitions)*-9999.
        pores_L = np.ones(partitions)*-9999.
    
        # divide the mesh into intervals divT and divL wide
        divT = int(float(self.x)/float(partitions))
        divL = int(float(self.y)/float(partitions))
        for p in range(partitions):
            pores_T[p] = np.mean(mesh[p*divT:(p+1)*divT,:])
            pores_L[p] = np.mean(mesh[:,p*divL:(p+1)*divL])
        
        # find the maximum difference between partitions
        maxdiff_T = np.amax(pores_T) - np.amin(pores_T)
        maxdiff_L = np.amax(pores_L) - np.amin(pores_L)
        
        # Find the largest of these two
        maxdiff = max(maxdiff_T,maxdiff_L)
        return maxdiff

    def viewVels(self,save=False,fname='vels.png'):
        """
        View velocities in a simple plot, and save the plot as a png if wanted.
        """
        self.checkVels()
        fig = plt.figure()
        if self.x > self.y:
            subplotX, subplotY = 211, 212
            orientation='horizontal'
        else:
            subplotX, subplotY = 121, 122
            orientation='vertical'
        axX = fig.add_subplot(subplotX,aspect='equal')
        axY = fig.add_subplot(subplotY,aspect='equal')

        dividerX = make_axes_locatable(axX)
        dividerY = make_axes_locatable(axY)
        
        pvx = axX.pcolormesh(self.xi,self.yi,self.VX, 
                cmap='PiYG',vmin=np.amin(self.VX),vmax=np.amax(self.VX))
        pvy = axY.pcolormesh(self.xi,self.yi,self.VY, 
                cmap='coolwarm',vmin=np.amin(self.VY),vmax=np.amax(self.VY))
        
        axX.set_title('$V_x$')
        axY.set_title('$V_y$')

        if orientation == 'horizontal':
            caxX = dividerX.append_axes("bottom", size="5%", pad=0.5)
            caxY = dividerY.append_axes("bottom", size="5%", pad=.5)
        elif orientation == 'vertical':
            caxX = dividerX.append_axes("right", size="5%", pad=0.05)
            caxY = dividerY.append_axes("right", size="5%", pad=0.05)

        cbX = fig.colorbar(pvx,orientation=orientation,ax=axX,cax=caxX)
        cbY = fig.colorbar(pvy,orientation=orientation,ax=axY,cax=caxY)
        for ax in [axX,axY]:
            ax.set_xlim(0,self.x)
            ax.set_ylim(0,self.y)
            ax.set_xlabel('$x$ [cells]')
            ax.set_ylabel('$y$ [cells]')
        cbX.set_label('ms$^{-1}$')
        cbY.set_label('ms$^{-1}$')
        fig.tight_layout()
        if save: fig.savefig(fname,bbox_inches='tight',dpi=300)
        plt.show()

    def viewMats(self,save=False,fname='mats.png'):
        """
        View all material fields in a simple matpltolib plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        for KK in range(self.NoMats):
            matter = np.copy(self.materials[KK,:,:])*(KK+1)
            matter = np.ma.masked_where(matter==0.,matter)
            ax.pcolormesh(self.xi,self.yi,matter, cmap='terrain',vmin=1,vmax=self.NoMats+1)
        ax.set_xlim(0,self.x)
        ax.set_ylim(0,self.y)
        ax.set_xlabel('$x$ [cells]')
        ax.set_ylabel('$y$ [cells]')
        if save: fig.savefig(fname,bbox_inches='tight',dpi=300)
        plt.show()
    
    def flipMesh(self,axis=1):
        """
        `Flip' the Mesh instance along either the horizontal (axis=0) or vertical
        (axis=1) axes. Alternatively, this can flip the material order with 
        axis = -1 (this does not change anything else).
        """
        if axis == 0:
            self.materials = self.materials[:,::-1,:]
            self.mesh = self.mesh[::-1,:]
            self.VX = self.VX[::-1,:]
            self.VY = self.VY[::-1,:]
        elif axis == 1:
            self.materials = self.materials[:,:,::-1]
            self.mesh = self.mesh[:,::-1]
            self.VX = self.VX[:,::-1]
            self.VY = self.VY[:,::-1]
        elif axis == -1:
            self.materials = self.materials[::-1,:,:]
        return


    def top_and_tail(self,num=3,axis=1):
        """
        Sets top and bottom 3 rows/columns to void cells. 
        Recommended when edges moving away from boundary
        are porous. Prevents erroneous tension/densities.
        """
        if axis == 0:
            self.materials[:,:num,:]  *= 0.
            self.materials[:,-num:,:] *= 0.
            self.mesh[:num,:]  *= 0.
            self.mesh[-num:,:] *= 0.
            self.VX[:num,:]  *= 0.
            self.VX[-num:,:] *= 0.
            self.VY[:num,:]  *= 0.
            self.VY[-num:,:] *= 0.
        elif axis == 1:
            self.materials[:,:,:num]  *= 0.
            self.materials[:,:,-num:] *= 0.
            self.mesh[:,:num]  *= 0.
            self.mesh[:,-num:] *= 0.
            self.VX[:,:num]  *= 0.
            self.VX[:,-num:] *= 0.
            self.VY[:,:num]  *= 0.
            self.VY[:,-num:] *= 0.

    def deleteMat(self,m):
        """
        removes all cells of material m from the mesh
        """
        assert m>0, "ERROR: Material must be a number from 1-9"
        self.materials[m-1,:,:] *= 0.
        return

    def fillAll(self,m):
        """
        Fills entire mesh with material m. if m ==-1, it will fill it with void (essentially 
        deleting all contents of the mesh). Otherwise existing material is prioritised.
        """
        if m == -1:
            # Erase all materials
            self.materials *= 0.
            self.VX *= 0.
            self.VY *= 0.
            self.mesh *= 0.
        else:
            # sum across material axes
            present_mat = np.sum(self.materials,axis=0)                      
            # If cell [i,j] is full, present_mat[i,j] = 1., so mat_tofill[i,j] = 0.
            mat_tofill  = 1. - present_mat
            # Fill chosen material mesh with the appropriate quantities
            self.materials[m-1] = mat_tofill 
    
    def fillPlate(self,m,MMin,MMax,axis=0):
        """
        Fills all rows (or columns if axis = 1) of mesh with material m between MMin and MMax. 
        if m ==-1, it will fill it with void (essentially deleting existing matter). Otherwise 
        existing material is prioritised.
        """
        if axis == 0:
            condition = (self.yy<=MMax)*(self.yy>=MMin)
        elif axis == 1:
            condition = (self.xx<=MMax)*(self.xx>=MMin)
        if m == -1:
            # Erase all materials
            self.materials[:,condition] *= 0.
            self.VX[condition] *= 0.
            self.VY[condition] *= 0.
            self.mesh[condition] *= 0.
        else:
            # make a copy of the current material's mmesh
            temp_materials = np.copy(self.materials[m-1])
            # Sum all cells along material axes
            summed_mats = np.sum(self.materials,axis=0)
            # in the temp mesh, set all cells that are in the region of interest AND ARE ALSO not full
            # to 1.
            temp_materials[condition*(summed_mats<1.)] = 1. #- np.sum(materials,axis=0)  
            # temp_materials now is the materials mesh but partially filled cells are also 1
            # temp_2 
            temp_2 = summed_mats*temp_materials
            temp_materials -= temp_2
            self.materials[m-1] += temp_materials

    def multiplyVels(self,multiplier=.5,axis=1):
        """
        This function multiplies all velocities by a 'multiplier' factor.
        works on whole mesh.
        """
        assert axis==0 or axis==1 or axis==2, "ERROR: axis can only be horizontal (0), vertical (1), or both (2)!"
        if axis == 0:
            self.VX *= multiplier
        elif axis == 1:
            self.VY *= multiplier
        elif axis == 2:
            self.VX *= multiplier
            self.VY *= multiplier

    def matVel(self,vel,mat,axis=1):
        """
        Assign a blanket velocity to one material 
        Useful before merging meshes or when using other objects in iSALE.
        """
        assert axis==0 or axis==1 or axis==2, "ERROR: axis can only be horizontal (0), vertical (1), or both (2)!"
        matpresence = self.materials[mat-1]

        if axis == 0:
            self.VX[matpresence==1.] = vel
        elif axis == 1:
            self.VY[matpresence==1.] = vel
        elif axis == 2:
            assert len(vel)==2, "ERROR: when giving both a value, vel must be of form vel = [xvel,yvel]"
            self.VX[matpresence==1.] = vel[0]
            self.VY[matpresence==1.] = vel[1]

    def blanketVel(self,vel,axis=1):
        """
        Assign a blanket velocity to whole domain. 
        Useful before merging meshes or when using other objects in iSALE.
        """
        assert axis==0 or axis==1 or axis==2, "ERROR: axis can only be horizontal (0), vertical (1), or both (2)!"
        if axis == 0:
            self.VX[:,:] = vel
        elif axis == 1:
            self.VY[:,:] = vel
        elif axis == 2:
            assert len(vel)==2, "ERROR: when giving both a value, vel must be of form vel = [xvel,yvel]"
            self.VX[:,:] = vel[0]
            self.VY[:,:] = vel[1]
    def plateVel(self,ymin,ymax,vel,axis=0):
        """
        Assign velocity in a plate shape; works both horizontally and vertically.
        """
        assert ymin<ymax, "ERROR: ymin must be greater than ymax!"
        assert axis==0 or axis==1 or axis==2, "ERROR: axis can only be horizontal (0), vertical (1), or both (2)!"
        if axis == 0:
            self.VX[(self.yy>=ymin)*(self.yy<=ymax)] = vel
        elif axis == 1:
            self.VY[(self.yy>=ymin)*(self.yy<=ymax)] = vel
        elif axis == 3:
            assert len(vel)==2, "ERROR: when giving both a value, vel must be of form vel = [xvel,yvel]"
            self.VX[(self.yy>=ymin)*(self.yy<=ymax)] = vel[0]
            self.VY[(self.yy>=ymin)*(self.yy<=ymax)] = vel[1]

    def calcVol(self,m=None,frac=False):
        """
        Calculate area of non-void in domain for material(s) m. 
        Area is in cells NOT S.I.
        """
        if m is None:
            Vol = np.sum(self.materials)
        elif type(m) == int:
            Vol = np.sum(self.materials[m-1])
        elif type(m) == list:
            Vol = 0
            for mm in m:
                Vol += np.sum(self.materials[mm-1])
        else:
            pass
        if frac: Vol /= float(self.Ncells)
        return Vol

    def vfrac(self,xbounds=None,ybounds=None):
        """
        Calculate the volume fraction of material within a user-specified box
        """
        box = np.sum(self.materials,axis=0)
        if xbounds is None and ybounds is not None:
            # ensure all cells in box outside of ymin and ymax won't be considered
            condition = (self.yy>ybounds[0])*(self.yy<ybounds[1])
            vf = float(np.sum(box[condition]))/np.size(box[condition])
        elif ybounds is None and xbounds is not None:
            # ensure all cells in box outside of xmin and xmax won't be considered
            condition = (self.xx>xbounds[0])*(self.xx<xbounds[1])
            vf = float(np.sum(box[condition]))/np.size(box[condition])
        elif xbounds is not None and ybounds is not None:
            # Same proceedure if both given
            condition = (self.xx>xbounds[0])*(self.xx<xbounds[1])*(self.yy>ybounds[0])*(self.yy<ybounds[1])
            vf = float(np.sum(box[condition]))/np.size(box[condition])
        else:
            vf = float(np.sum(box))/float(np.size(box))
        return vf
    
    def details(self):
        """
        creates easily printable string of details of this instance
        """
        deets  = "Mesh instance called {}\n".format(self.name)
        deets += "Domain size: {} x {}\n".format(self.x,self.y)
        deets += "Cell size: {} m\n".format(self.cellsize)
        MMM = []
        for i in range(9):
            if np.amax(self.materials[i,:,:])>0: MMM.append(i+1)
        deets += "Materials used: {}\n".format(MMM)
        deets += "Max & Min velocities\n"
        deets += "Vx: {} m/s & {} m/s\n".format(np.amax(self.VX),np.amin(self.VX)) 
        deets += "Vy: {} m/s & {} m/s".format(np.amax(self.VY),np.amin(self.VY))
        return deets
    
    def VoidFracForTargetPorosity(self,matrix,bulk=0.5,final_por=0.65):
        """
        Calculates the void fraction required to reduce the matrix distension to specified porosity
        (as an area fraction of the domain)
        """
        initial_por = self.matrixPorosity(matrix,bulk,void=False,Print=False)/100.
        MatrixVol = self.calcVol(matrix)
        VoidInMatrix = MatrixVol*initial_por
        TargetVoidInMatrix = (final_por/initial_por)*VoidInMatrix
        VoidVolumeFrac = (VoidInMatrix - TargetVoidInMatrix)/self.Ncells
        return VoidVolumeFrac

    def matrixPorosity(self,matrix,bulk,void=False,Print=True):
        """
        calculates the necessary matrix porosity to achieve a target bulk porosity
        given current domain occupance.
        Args:
            matrix: int; material number of the matrix
            bulk:   float; bulk porosity (e.g. bulk = 0.5 --50% porous--)
            void:   bool; is there explicit void in the mesh? True/False
            Print:  bool; if True print out the result, if false, just return the result
        """
        # if bulk porosity a percentage, convert to fraction < 1 and >= 0
        if bulk >= 1.: bulk /= 100.
        matrix_vol = self.calcVol(matrix)
        other = list(self.mats)
        other.remove(matrix)
        other_vol = self.calcVol(other)
        other_vf = other_vol/float(matrix_vol+other_vol)
        
        total_vf = 1. - bulk
        assert other_vf <= total_vf, "ERROR: The non-matrix volume fraction {:2.2f}% is too high for that bulk porosity {:2.2f}%!".format(other_vf*100.,bulk*100.)
        remain_vf = total_vf - other_vf
        # remaining solid volume (in matrix)
        remain_vol = remain_vf*(matrix_vol+other_vol)
        matrix_por = (matrix_vol - remain_vol)/float(matrix_vol)
        if Print:
            distension = 1./(1.-matrix_por)
            print "bulk porosity = {:3.3f}%".format(bulk*100.)
            print "Matrix: porosity = {:3.3f}% and distension = {:3.3f}".format(matrix_por*100.,distension)
        return matrix_por*100.


    def save(self,fname='meso_m.iSALE',info=False,compress=False):
        """
        A function that saves the current mesh as a text file that can be read, 
        verbatim into iSALE.
        This compiles the integer indices of each cell, 
        as well as the material in them and the fraction
        of matter present. It saves all this as the filename specified by the user, 
        with the default as 
        meso_m.iSALE
        
        This version of the function works for continuous and solid materials, 
        such as a multiple-plate setup.
        It does not need to remake the mesh as there is no particular matter present.
        
        Args:
            fname   : The filename to be used for the text file being used
            info    : Include particle ID (i.e. #) as a column in the final file 
            compress: compress the file? For large domains it is often necessary to avoid very large files; uses gz
        
        returns nothing but saves all the info as a txt file called 'fname' and 
        populates the materials mesh.
        """
        self.checkVels()
        ncells = self.x*self.y
        if info:
            OI    = np.zeros((ncells))
            PI    = np.zeros((ncells))
    
        XI    = np.zeros((ncells))    
        YI    = np.zeros((ncells))
        UX    = np.zeros((ncells))
        UY    = np.zeros((ncells))
        K     = 0
        
        # make list of used material numbers
        # iterate through them all
        usedmats = self.mats[:]
        for mm in self.mats:
            # If a material hasn't been used...
            total = np.sum(self.materials[mm-1])
            # ...remove from usedmats list
            if total == 0.: usedmats.remove(mm)
        NM = len(usedmats)
        FRAC = np.zeros((NM,ncells))
        for j in range(self.x):
            for i in range(self.y):
                XI[K] = i
                YI[K] = j
                UX[K] = self.VX[j,i]
                UY[K] = self.VY[j,i]
                if info:
                    PI[K] = self.mesh[j,i]
                for mm in range(NM):
                    FRAC[mm,K] = self.materials[usedmats[mm]-1,j,i]
                K += 1
        FRAC = self._checkFRACs(FRAC)
        HEAD = '{},{}'.format(K,NM)
        #print HEAD
        if info:
            ALL  = np.column_stack((XI,YI,UX,UY,FRAC.transpose(),PI))                                       
        else:
            ALL  = np.column_stack((XI,YI,UX,UY,FRAC.transpose()))
        if compress: fname += '.gz'
        np.savetxt(fname,ALL,header=HEAD,fmt='%5.3f',comments='')
    
    def save_oldver(self,fname='meso_m.iSALE'):
        """
        identical to save, except it exports to an old version of output which has no velocity info
        
        Args:
            fname   : The filename to be used for the text file being used
        """
        ncells = self.x*self.y
        
        XI    = np.zeros((ncells))    
        YI    = np.zeros((ncells))
        UX    = np.zeros((ncells))
        UY    = np.zeros((ncells))
        K     = 0
        
        # make list of used material numbers
        usedmats = list(self.mats)
        # iterate through them all
        for mm in self.mats:
            # If a material hasn't been used...
            total = np.sum(self.materials[mm-1])
            # ...remove from usedmats list
            if total == 0.: usedmats.remove(mm)
        NM = len(usedmats)
        FRAC = np.zeros((NM,ncells))
        for j in range(self.x):
            for i in range(self.y):
                XI[K] = i
                YI[K] = j
                UX[K] = self.VX[j,i]
                UY[K] = self.VY[j,i]
                for mm in range(NM):
                    FRAC[mm,K] = self.materials[usedmats[mm]-1,j,i]
                K += 1
        FRAC = self._checkFRACs(FRAC)
        HEAD = '{},{}'.format(K,NM)
        print "Output mesh {} has shape {} x {}".format(fname,self.x,self.y)
        ALL  = np.column_stack((XI,YI,FRAC.transpose()))                                               
        np.savetxt(fname,ALL,header=HEAD,fmt='%5.3f',comments='')
    
    def _checkFRACs(self,FRAC):
        """
        This function checks all the volume fractions in each cell and deals with 
        any occurrences where they add to more than one
        by scaling down ALL fractions in that cell, such that it is only 100% full.
        
        FRAC : Array containing the full fractions in each cell of each material
        """
        
        if self.mixed==True:
            for i in range(self.Ncells):
                SUM = np.sum(FRAC[:,i])
                if SUM > 1.:
                    FRAC[:,i] /= SUM
                else:
                    pass
        else:
            for i in range(self.Ncells):
                SUM = np.sum(FRAC[:,i])
                if SUM > 1.:
                    done = False
                    for j in range(Self.NoMats):
                        if FRAC[j,i] > 0 and done == False: 
                            FRAC[j,i] = 1.
                            done = True
                        else:
                            FRAC[j,i] = 0.
                else:
                    pass
        
        return FRAC
    
    def _setboundsWholemesh(self,axis):
        if axis == 0:
            bounds = [np.amin(self.xx),np.amax(self.xx)]
        elif axis == 1:
            bounds = [np.amin(self.xx),np.amax(self.xx)]
        return bounds
