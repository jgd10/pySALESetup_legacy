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

from pss_domainclasses import *
from pss_objectclasses import *
import pss_functions as psf

class Grain:
    """
    Instances can be created separate to a Mesh class and inserted at will. The main feature of Each instance
    is the mesh. This is a mini-domain that contains all the cells that are filled or not. This constitutes the 'grain'. Other
    properties are stored in relation to it such as the shape type, area, rotation, and 'equivalent radius'. Equivalent radius 
    is the radius a circle of equal area would possess. This allows for easy relative scaling of grains which are different 
    shapes.
    """
    def __init__(self, eqr=10., rot=0., shape='circle', 
            File=None, elps_params=None, poly_params=None, mixed=False, name='grain1', Reload=False):
        """
        When initialised the type of shape must be specified. Currently pySALESetup can handle N-sided polygons, 
        circles and ellipses. Other shapes can be added if and when necessary (e.g. hybrids).
        Mixed cells mode has not been fully tested yet.   

        Args:
            eqr:         float; radius of a circle with the same area
            rot:         float; angle to rotate the grain by.
            shape:       string; label for the shape of the grain
            File:        string; if file specified contains path to file
            elps_params: list of floats; [major radius (cells), eccentricity]
            poly_params: 2D list of floats; [[X0,Y0],[X1,Y1],...,[XN,YN]] coords of all vertices on a 2x2 grid -1 <= X,Y <= 1
            mixed:       bool; mixed cells on or off
            name:        string; label for the instance
            Reload:      bool; if true, attempt to get attributes from pickle file in form name.obj
        """
        self.name = name
        if Reload:
            self.load()
            print 'loaded {}'.format(self.name)
        else:
            self.equiv_rad = eqr
            self.angle = rot
            self.shape = shape
            self.mixed = mixed
            self.hostmesh = None
            self.mat = None
        if self.shape == 'circle' or self.shape =='sphere':
            self.shape = 'circle'
            self.radius = self.equiv_rad
            self.mesh = psf.grainfromCircle(self.equiv_rad)
            self.area = np.sum(self.mesh)
        elif self.shape == 'file':
            assert File is not None, "No file path provided to create grain"
            self.mesh = psf.grainfromVertices(fname=File,eqv_rad=self.equiv_rad,mixed=self.mixed,rot=self.angle)
            self.area = np.sum(self.mesh)
            self.radius = np.sqrt(self.area/np.pi)
        elif self.shape == 'ellipse':
            assert len(elps_params) == 2, "ERROR: ellipse creation requires elps_params to be of the form [major radius, eccentricity]"
            self.mesh = psf.grainfromEllipse(elps_params[0],self.angle,elps_params[1])
            self.area = np.sum(self.mesh)
            self.eccentricity = elps_params[1]
            self.radius = np.sqrt(self.area/np.pi)
        elif self.shape == 'polygon':
            assert len(poly_params) >= 3, "ERROR: Polygon creation requires at least 3 unique coordinates"
            self.mesh = psf.grainfromVertices(R=poly_params,eqv_rad=self.equiv_rad,mixed=self.mixed,rot=rot)
            self.area = np.sum(self.mesh)
        else:
            print "ERROR: unsupported string -- {} --  used for shape.".format(self.shape)
        # more to be added...
        self.Px,self.Py = np.shape(self.mesh)
    def details(self):
        """creates easily printable string of details of this instance"""
        deets = "Grain instance called {}:\nshape: {}\nrotation: {:3.2f} rad\n".format(self.name,self.shape,self.angle)
        deets += "equivalent radius: {:3.2f} cells\ncurrent coords: ({:2.2g}, {:2.2g})\n".format(self.equiv_rad,self.x,self.y)
        deets += "mixed cells: {}".format(self.mixed)
        return deets
    def view(self,save=False,show=True,fname='grain.png'):
        """ view the grain in a simple plot """
        fig = plt.figure(figsize=(3.,3.))
        ax = fig.add_subplot(111,aspect='equal')
        ax.imshow(self.mesh,cmap='binary')
        ax.set_xlabel('x [cells]')
        ax.set_ylabel('y [cells]')
        fig.tight_layout()
        if show: plt.show()
        if save: fig.savefig(fname,dpi=200,transparency=True)
        ax.cla()
        
    def _cropGrain(self,x,y,LX,LY):
        """
        If a grain is placed, such that it overlaps with the edge of the domain, the mesh 
        and grain need to be 'cropped', such that the target mesh and grain mesh subsets match.
        
        Args:
            x,y   : int; the coords of the grain placement (in cells)
            LX,LY : int; x-width and y-width (in cells) of the target mesh

        Returns:
            Is,Js : list of ints; indices for the target mesh
            i_,j_ : list of ints; indices for the minimesh

        """
        self.Px,self.Py = np.shape(self.mesh)
        Px, Py = self.Px,self.Py
        i_edge = int(x - Px/2.)
        j_edge = int(y - Py/2.)
        i_finl = int(x + Px/2.)
        j_finl = int(y + Py/2.)
        # NB 'i' refers to the target mesh indices whereas 'I' refers to shape's mesh indices
        # if shape is 'sticking out' of target mesh it must be cropped appropriately
        # i_edge is the x=0 edge (i.e. left vertical wall)
        if i_edge < 0:
            # when comparing the two meshes, shape will be sliced from I_initial to I_final
            I_initial = abs(i_edge)                                                                      
            i_edge    = 0                                                                                
        else:                                             
            I_initial = 0                                                                                
        # Same again but for the 'horizontal' 'ceiling'
        if j_edge < 0:                                                                                   
            J_initial = abs(j_edge) 
            j_edge = 0
        else:
            J_initial = 0
        
        # Perform the same calculations but for the other two sides
        I_final = Px 
        if (i_finl)>LX:                                                                                                                                     
            I_final -= abs(LX-i_finl)                                                                     
            # additional -1 because end of array index is always included
            # but the reverse is not true for start: e.g. A[0:2] will include the
            # '0'th element but NOT the '2'th element
            I_final -= 1
            i_finl   = LX-1
        J_final = Py
        if (j_finl)>LY:
            J_final -= abs(LY-j_finl) 
            J_final -= 1
            j_finl   = LY-1
        
        Is = [I_initial,I_final]
        Js = [J_initial,J_final]
        i_ = [i_edge,i_finl]
        j_ = [j_edge,j_finl]
        return Is,Js,i_,j_

    def place(self,x,y,m,target,num=None,mattargets=None):
        """
        Inserts the shape into the correct materials mesh at coordinate x, y.
        
        Args:
            x, y   : float; The x and y coords at which the shape is to be placed. (These are the shape's origin point)
            m      : int; this is the index of the material
            target : Mesh; the target mesh, must be a 'Mesh' instance
            num    : int; the 'number' of the particle.

        if m == -1 then void is inserted, overwriting all existing material
        if m == 0  then void is inserted, ONLY overwriting materials specified in targets
        
        existing material takes preference and is not displaced

        nothing is returned.
        """
        if self.mixed: 
            assert num is None, "ERROR: Particle number not yet supported in mixed-cell systems"
        else:
            if num is None: num = 1.
        if m == 0: assert mattargets is not None, "ERROR: When creating void pores targets material(s) must be set"
        assert type(x)==type(y), "ERROR: x and y coords must have the same type"
        self.x = x
        self.y = y
        self.mat = m
        if (type(x)==int and type(y)==int) | (type(x)==np.int64 and type(y)==np.int64):
            self.x = x*target.cellsize
            self.y = y*target.cellsize
        else:
            x = int(x/target.cellsize)
            y = int(y/target.cellsize)
        if type(m) == float or type(m) == np.float64: m = int(m)
        Is,Js,i_,j_ = self._cropGrain(x,y,target.x,target.y)
        # slice ammunition grain for the target
        temp_shape = self.mesh[Is[0]:Is[1],Js[0]:Js[1]]  
        if self.mixed == False:
            for p in range(Js[1]-Js[0]):
                for o in range(Is[1]-Is[0]):
                    #print o,i_[0],p,j_[0],np.shape(temp_shape),np.shape(target.materials[0])
                    if temp_shape[o,p] == 1.: 
                        # if m = -1 then cells filled with void instead; also deletes existing material
                        if m == -1: 
                            target.materials[:,o+i_[0],p+j_[0]] = 0.
                            target.mesh[o+i_[0],p+j_[0]] = 0.
                        elif m == 0:
                            for tm in mattargets:
                                if target.materials[tm-1,o+i_[0],p+j_[0]] != 0.: 
                                    target.mesh[o+i_[0],p+j_[0]] = 0.
                                target.materials[tm-1,o+i_[0],p+j_[0]] = 0.
                        elif m > 0 and np.sum(target.materials[:,o+i_[0],p+j_[0]]) == 0.:
                            target.materials[m-1,o+i_[0],p+j_[0]] = 1.
                            target.mesh[o+i_[0],p+j_[0]] = num

        # shape number not yet stored in mixed cases
        elif self.mixed == True:
            for o in range(Js[1]-Js[0]):
                for p in range(Is[1]-Is[0]):
                    if temp_shape[o,p] > 0.: 
                        if m == -1: 
                            target.materials[:,o+j_[0],p+i_[0]] = 0.
                            target.mesh[o+j_[0],p+i_[0]] = 0.
                        elif m > 0:
                            tot_present = np.sum(target.materials[:,o+j_[0],p+i_[0]])
                            space_left = 1. - tot_present
                            if temp_shape[o,p] > space_left: 
                                new_mat = space_left
                            else:
                                new_mat = temp_shape[o,p]
                            target.materials[m-1,o+j_[0],p+i_[0]] += new_mat
        self.hostmesh = target


    def insertRandomly(self,target,m,xbounds=None,ybounds=None,nooverlap=False,mattargets=None):
        """
        insert grain into bed in an empty space. By default select from whole mesh, 
        alternatively, choose coords from region bounded by xbounds = [xmin,xmax] and 
        ybounds = [ymin,ymax]. Position is defined by grain CENTRE
        so overlap with box boundaries is allowed. If no overlap wanted, reduce box size
        within function.

        Args:
            target:  Mesh
            m:       int; material number
            xbounds: list
            ybounds: list
            nooverlap: bool
        """
        if m == 0:
            assert mattargets is not None, "When inserting void that does not overwrite everything, the materials to be overwritten must be specified in mattargets"
        if nooverlap and ybounds is not None: 
            ybounds[0] += self.radius*target.cellsize
            ybounds[1] -= self.radius*target.cellsize
        if nooverlap and xbounds is not None: 
            xbounds[0] += self.radius*target.cellsize
            xbounds[1] -= self.radius*target.cellsize
        target.mesh = np.sum(target.materials,axis=0)
        target.mesh[target.mesh>1.] = 1.
        box = None
        # If NOT overwriting everything we want to place pores in regions of target material
        # OR void! (if any is present) but NOT over material we are not overwriting. To form the
        # box we select coords from, target material & void should be 0.
        if m == 0:
            box = np.zeros_like(target.mesh)
            for ii in range(9):
                if np.isin(ii+1,mattargets):
                    pass
                else:
                    box += target.materials[ii]
        else:
            box = np.sum(target.materials,axis=0)
        if xbounds is None and ybounds is not None:
            # ensure all cells in box outside of ymin and ymax won't be considered
            box[(target.yy>ybounds[1])] = 9999.
            box[(target.yy<ybounds[0])] = 9999.
        elif ybounds is None and xbounds is not None:
            # ensure all cells in box outside of xmin and xmax won't be considered
            box[(target.xx>xbounds[1])] = 9999.
            box[(target.xx<xbounds[0])] = 9999.
        elif xbounds is not None and ybounds is not None:
            # Same proceedure if both given
            box[(target.yy>ybounds[1])] = 9999.
            box[(target.yy<ybounds[0])] = 9999.
            box[(target.xx>xbounds[1])] = 9999.
            box[(target.xx<xbounds[0])] = 9999.
        nospace = True
        counter = 0
        passes  = 0
        indices = np.where(box==0.)
        indices = np.column_stack(indices)
        assert len(indices) > 0, 'ERROR: domain is completely full of material; no more space can be found'
        pore = False
        if m==0: pore = True
        while nospace:
            x,y   = random.choice(indices)
            nospace, overlap = self._checkCoords(x,y,target,pore=pore)
            counter += 1
            if counter>10000:
                nospace = True
                passes= 1
                print "No coords found after {} iterations; exiting".format(counter)
                break
        if nospace:
            pass
        else:
            if m == 0:
                self.place(x,y,m,target,mattargets=mattargets)
            else:
                self.place(x,y,m,target)
        return 
    def _checkCoords(self,x,y,target,overlap_max=0.,pore=False): 
        """
        Checks if the grain will overlap with any other material;
        and if it can be placed.
        
        It works by initially checking the location of the generated coords and 
        ammending them if the shape overlaps the edge of the mesh. Then the two arrays
        can be compared. If a pore is being placed it does not need to check anything
        void overlapping with void or other material is not an issue because the materials
        allowed to be overwritten are already specified.
        
        Args:
            x:                 float; The x coord of the shape's origin
            y:                 float; The equivalent y coord
            target:            Mesh; 
            overlap_max:       float; maximum number of overlapping cells allowed
            pore:              bool; are the coordinates of a pore being checked?
        Returns:
            nospace:           boolean;
            overlapping_cells: float; number of cells that overlap with other grains

        the value of nospace is returned. False is a failure, True is success.
        """
        nospace = False             
        if pore is not True:
            Is,Js,i_,j_ = self._cropGrain(x,y,target.x,target.y)
            
            temp_shape = np.copy(self.mesh[Is[0]:Is[1],Js[0]:Js[1]])
            temp_mesh  = np.copy(target.mesh[i_[0]:i_[1],j_[0]:j_[1]])
            test       = np.minimum(temp_shape,temp_mesh)

            overlapping_cells = np.sum(test)
                                                                            
            if (overlapping_cells > overlap_max):
                nospace = True
            elif (overlapping_cells == 0):
                pass
        else:
            overlapping_cells = 0.
        return nospace, overlapping_cells

    def insertRandomwalk(self,target,m,xbounds=None,ybounds=None,mattargets=None):
        """
        Similar to insertRandomly. Randomly walk until allowed contact established and place
        Initial coordinates taken from box and on a void cell
        Move grain by random increments of equivalent radius in x and y
        Once in contact with another grain 
            (contact defined as 'overlaps by 100th the area of active grain')
        place into mesh
        If bounded by x/ybounds; do not allow motion outside of these.
        
        Args:
            target:  Mesh
            m:       int; material number
            xbounds: list
            ybounds: list
        """
        if m == 0:
            assert mattargets is not None, "When inserting void that does not overwrite everything, the materials to be overwritten must be specified in mattargets"
        target.mesh = np.sum(target.materials,axis=0)
        target.mesh[target.mesh>1.] = 1.
        box = None
        box = np.sum(target.materials,axis=0)
        XCELLMAX = target.x 
        XCELLMIN = 0
        YCELLMAX = target.y
        YCELLMIN = 0
        if xbounds is None and ybounds is not None:
            # ensure all cells in box outside of ymin and ymax won't be considered
            box[(target.yy>ybounds[1])] = 9999.
            box[(target.yy<ybounds[0])] = 9999.
            YCELLMAX = int((ybounds[1]-np.amin(target.yy))/target.cellsize)
            YCELLMIN = int((ybounds[0]-np.amin(target.yy))/target.cellsize)
        elif ybounds is None and xbounds is not None:
            # ensure all cells in box outside of xmin and xmax won't be considered
            box[(target.xx>xbounds[1])] = 9999.
            box[(target.xx<xbounds[0])] = 9999.
            XCELLMAX = int((xbounds[1]-np.amin(target.xx))/target.cellsize)
            XCELLMIN = int((xbounds[0]-np.amin(target.xx))/target.cellsize)
        elif xbounds is not None and ybounds is not None: 
            # Same proceedure if both given
            box[(target.yy>ybounds[1])] = 9999.
            box[(target.yy<ybounds[0])] = 9999.
            box[(target.xx>xbounds[1])] = 9999.
            box[(target.xx<xbounds[0])] = 9999.
            XCELLMAX = int((xbounds[1]-np.amin(target.xx))/target.cellsize)
            XCELLMIN = int((xbounds[0]-np.amin(target.xx))/target.cellsize)
            YCELLMAX = int((ybounds[1]-np.amin(target.yy))/target.cellsize)
            YCELLMIN = int((ybounds[0]-np.amin(target.yy))/target.cellsize)
        # Max number of overlapping cells should scale with area.
        cell_limit = max(np.sum(self.area)/100.,1)
        # area ~= 110 cells for 6cppr
        # Does NOT need to be integer since values in the mesh are floats, 
        # and it is their sum that is calculated.
        touching   = False
        counter    = 0  

        indices = np.where(box==0.)                                                          
        indices = np.column_stack(indices)
        x,y   = random.choice(indices)
        end = False
        r_int = int(self.radius)
        while not touching:    
            if x > XCELLMAX: x = XCELLMIN
            if x < XCELLMIN: x = XCELLMAX
            if y > YCELLMAX: y = YCELLMAX
            if y < YCELLMIN: y = YCELLMIN
            nospace, overlap = self._checkCoords(x,y,target,overlap_max=cell_limit)
            counter += 1
            if counter>=10000:
                print "No coords found after {} iterations; exiting".format(counter)
                break
        
            if nospace or (nospace == False  and overlap == 0):

                dx = random.randint(-r_int,r_int)
                dy = random.randint(-r_int,r_int)
                x += dx
                y += dy
            elif nospace == False and overlap > 0 and overlap <= cell_limit:
                touching = True
                break
        x = np.int64(x)
        y = np.int64(y)
        if touching:
            self.x = x
            self.y = y
            self.mat = m
            if m == 0:
                self.place(x,y,m,target,targets=mattargets)
            else:
                self.place(x,y,m,target)
        return

    def save(self):
        """save class as self.name.obj"""
        file = open(self.name+'.obj','w')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return

    def load(self):
        """try load self.name.txt"""
        file = open(self.name+'.obj','r')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        return

class Ensemble:
    """
    A class wherein can be stored information on grains 
    In addition to storing the information of any grains added to it, it has some other
    functions. None are more useful than optimise_materials, that will tell you the 
    optimum material distribution (given a certain number) for grains in your ensemble.

    This class was designed primarily to allow for multiple ensembles in the same domain
    e.g. a bimodal particle bed; particles from two different materials. Ensemble classes
    can store their information separately and optimise their materials separately.
    """
    def __init__(self,hostmesh,name='Ensemble1',Reload=False):
        """
        Args:
            host_mesh: Mesh
            name:      string
            Reload:    bool; see Grain instance
        """
        assert hostmesh is not None, "ERROR: ensemble must have a host mesh"
        self.name = name
        if Reload:
            self.load()
            self.hostmesh = hostmesh
            print 'loaded {}'.format(self.name)
        else:
            self.hostmesh = hostmesh
            self.grains = []
            self.number = 0
            self.rots   = []
            self.radii = [] 
            self.areas = []
            self.phi = [] #self._krumbein_phi()
            self.xc = []
            self.yc = []
            self.mats = []
            self.shapes = []

    def add(self,particle,x=None,y=None):
        """
        Add a grain to the Ensemble. Allow x and y to be specified when adding because
        grain may not be placed yet and have no value, for example. There are other, more
        niche, reasons for this as well.

        Args:
            particle: Grain
            x:        float
            y:        float
        """
        self.grains.append(particle)
        self.number += 1
        self.rots.append(particle.angle)
        self.radii.append(particle.radius)
        if x is None: x = particle.x
        if y is None: y = particle.y
        self.xc.append(x)
        self.yc.append(y)
        self.mats.append(particle.mat)
        self.phi.append(-np.log2(particle.radius*2.*self.hostmesh.cellsize*1.e3))
        self.areas.append(particle.area)
        self.shapes.append(particle.shape)
    
    def details(self):
        """
        creates easily printable string of details of this instance
        """
        deets  = "Ensemble instance called {}:\nGrain shapes: {}\n".format(self.name,np.unique(self.shapes))
        deets += "Number of grains: {}\n".format(self.number)
        deets += "Mean equivalent radius: {:3.2f} cells; std: {:3.2f} cells\n".format(np.mean(self.radii),np.std(self.radii))
        deets += "Largest radius: {:3.2f} cells\n".format(np.amax(self.radii))
        deets += "Smallest radius: {:3.2f} cells\n".format(np.amin(self.radii))
        deets += "Materials used: {}\n".format(np.unique(self.mats))
        return deets

    def PSDdetails(self):
        """
        creates easily printable string of details of the size distribution within this Ensemble
        """
        self.calcPSD()
        deets  = "Ensemble instance called {}:\nGrain shapes: {}\n".format(self.name,np.unique(self.shapes))
        deets += "The distribution of grains have:\n"
        deets += "means: {:2.2f} radius; {:2.2f} phi; {:2.2f} area\n".format(*self.means)
        deets += "medians: {:2.2f} radius; {:2.2f} phi; {:2.2f} area\n".format(*self.medians)
        deets += "modes: {:2.2f} radius; {:2.2f} phi; {:2.2f} area\n".format(*self.modes)
        deets += "variances: {:2.2f} radius; {:2.2f} phi; {:2.2f} area\n".format(*self.variances)
        deets += "skews: {:2.2f} radius; {:2.2f} phi; {:2.2f} area\n".format(*self.skews)
        return deets

    def calcPSD(self):
        """
        Generates a Particle Size Distribution (PSD) from the Ensemble
        """
        # use in-built functions to generate the frequencies and vfrac at this instance
        self.frequency()
        self._vfrac()
        self.means    = np.array([np.mean(self.radii),np.mean(self.phi),np.mean(self.areas)])
        self.medians  = np.array([np.median(self.radii),np.median(self.phi),np.median(self.areas)])
        self.modes  =   np.array([scst.mode(self.radii,axis=None)[0][0],scst.mode(self.phi,axis=None)[0][0],scst.mode(self.areas,axis=None)[0][0]])
        self.variances =np.array([np.var(self.radii),np.var(self.phi),np.var(self.areas)])
        self.skews  =   np.array([scst.skew(self.radii),scst.skew(self.phi),scst.skew(self.areas)])

    def plotPSD(self,forceradii=False,returnplots=False):
        """
        Generates a Particle Size Distribution (PSD) from the Ensemble
        Args:
            returnplots: bool; if true the function returns ax1 ax2 and fig for further modding
            forceradii:  bool; if true use radii instead of phi
        """
        self.calcPSD()
        
        # Frequency of each unique area
        areaFreq = np.array(self.areaFreq.values())
        # Array of the unique areas
        areas    = np.array(self.areaFreq.keys())
        # Probability Distribution Function
        PDF = areaFreq*areas/np.sum(areaFreq*areas)
        # ... as a percentage of total volume fraction occupied
        PDF *= 100.
        # Cumulative Distribution Function
        CDF = np.cumsum(PDF)
        # krumbein phi, standard measure of grains ize in PSDs
        rad = np.sqrt(areas/np.pi)
        phi = self._krumbein_phi(rad)

        # create plot for the PSD
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        if forceradii:
            rad = np.unique(self.radii)
            ax1.plot(rad,PDF,marker='x',mfc='None',linestyle=' ',color='crimson',mew=1.5)
            ax2.plot(rad,CDF,marker='+',mfc='None',linestyle=' ',color='crimson',mew=1.5)
            ax1.set_xlabel('Equivalent Radius [mm]')
            ax2.set_xlabel('Equivalent Radius [mm]')
        else:
            ax1.plot(phi,PDF,marker='x',mfc='None',linestyle=' ',color='crimson',mew=1.5)
            ax2.plot(phi,CDF,marker='+',mfc='None',linestyle=' ',color='crimson',mew=1.5)
            ax1.set_xlabel('$\phi$' + '\n' + '$= -log_2(Equivalent Diameter)$')
            ax2.set_xlabel('$\phi$' + '\n' + '$= -log_2(Equivalent Diameter)$')
        ax1.set_title('PDF')
        ax2.set_title('CDF')
        ax1.set_ylabel('%.Area')
        
        # return plots if needed, otherwise show figure
        #if returnplots: return ax1,ax2,fig
        fig.tight_layout()
        plt.show()

    def _krumbein_phi(self,rad=None):
        """
        Convert all radii in the ensemble to the Krumbein Phi, which is commonly used in 
        particle size distributions.
        """
        phi = []
        # Krumbein phi = -log_2(D/D0) where D0 = 1 mm and D = diameter
        if rad is None:
            for r in self.radii:
                # convert radii (which are ALWAYS in cells) to mm
                r_mm = r*self.hostmesh.cellsize*1.e3
                # Phi can ONLY take input in mm!
                p = -np.log2(2.*r_mm)
                phi.append(p)
        else:
            for r in rad:
                r_mm = r*self.hostmesh.cellsize*1.e3
                p = -np.log2(2.*r_mm)
                phi.append(p)
        return phi
    def _reverse_phi(self,phi):
        """
        reverse of krumbein_phi; code supplies input here
        """
        rad = []
        for p in phi:
            rad.append((2**(-p))*.5)
        # NB rad will be in milimetres! convert to cells:
        # Also cellsize always in metres
        rad_cells = rad/(self.hostmesh.cellsize*1.e3)
        return rad
    def _vfrac(self):
        """
        Calculate the area fraction the ensemble occupies in the domain
        """
        self.vfrac = np.sum(self.areas)/float(self.hostmesh.Ncells)

    def print_vfrac(self):
        """
        Print out the area fraction; this is a user-accessible function
        """
        print self._vfrac()
    
    def frequency(self):
        """
        Calculate the frequencies of each grain based on their size.
        """
        self._krumbein_phi()
        self.frequencies = Counter(self.phi)
        self.areaFreq = Counter(self.areas)
        self.frequencies = OrderedDict(sorted(self.frequencies.items(),reverse=True))
        self.areaFreq = OrderedDict(sorted(self.areaFreq.items(),reverse=True))

    def area_weights():
        ensemble_area = _vfrac()*self.hostmesh.area
        self.area_weights = [100.*a/ensemble_area for a in self.areas]
    
    def optimise_materials(self,mats=np.array([1,2,3,4,5,6,7,8,9]),populate=True,target=None):                        
        """
        This function has the greatest success and is based on that used in JP Borg's work 
        with CTH.
    
        Function to assign material numbers to each particle
        This function tries to optimise the assignments such
        that as few particles of the same material are in co
        ntact as possible. It works by creating an array of
        all the particle material numbers within a box, 6 x 6
        radii around each particle coord, as well as the corres
        ponding coords.
    
        Then they are sorted into the order closest -> furthest.
        Only the first M are considered (in this order). M is
        the number of different materials being used. The
        corresponding array of materials is checked against mats.
        
        If they contain all the same elements, there are no repeats
        and all material numbers are used up => use that of the 
        particle furthest away.
    
        If they do not, there is at least one repeat, select the
        remaining material number or randomly select one of those
        left, if there are more than one.
    
        Continue until all the particles are assigned.
        
        Args:
            mats: array; containing all the material numbers to be assigned
    
        Returns:
            MAT:  array; containg an optimal material number for each grain
        """
        if type(mats) == list: mats = np.array(mats)
        # No. of particles
        N    = self.number                                                               
        M    = len(mats)
        # Length of one cell
        L    = self.hostmesh.cellsize                                                        
        assert type(mats)==np.ndarray,"ERROR: material list must be a numpy array!"
        matsARR = np.array(mats)
        # Array for all material numbers of all particles
        MAT  = np.array(self.mats)                                                       
        xcARR = np.array(self.xc)
        ycARR = np.array(self.yc)
        # Counts the number of particles that have been assigned
        # Loop every particle and assign each one in turn.    
        # perform the optimisation algorithm 3 times to improve material 
        # assignment
        for loop in range(3):
            i = 0                                                                            
            while i < N:                                                                     
                Ns = max(self.grains[i].Px,self.grains[i].Py)
                xc = self.xc[i]
                yc = self.yc[i]
                # Create a 'box' around each particle (in turn) that is 6Ns x 6Ns
                lowx   = xc - 3.*Ns*L                                                        
                higx   = xc + 3.*Ns*L
                lowy   = yc - 3.*Ns*L
                higy   = yc + 3.*Ns*L
                # consider grains in the box that are also NOT the grain being considered!
                condition =(lowx<=self.xc)*(self.xc<=higx)*(lowy<=self.yc)*(self.yc<=higy)*(self.xc!=xc)*(self.yc!=yc) 
                # Array containing a list of all material numbers within the 'box' 
                boxmat =     MAT[condition]                                                  
                # Array containing the corresponding xcoords
                boxx   =   xcARR[condition]                                                  
                # and the ycoords
                boxy   =   ycARR[condition]                                                  
                nn     =  np.size(boxmat)
                # Calculate the distances to the nearest particles
                D   = np.sqrt((boxx - xc)**2. + (boxy - yc)**2.)                             
                # Sort the particles into order of distance from the considered particle
                ind = np.argsort(D)                                                          
                # Sort the materials into the corresponding order
                BXM = boxmat[ind]                                                            
                # Only select the M closest particles
                DU  = np.unique(BXM[:M])                                                     
                if np.array_equal(DU, matsARR):                                              
                    # If the unique elements in this array equate the array of       
                    # materials then all are taken
                    # Set the particle material to be of the one furthest from 
                    # the starting particle
                    mm     = BXM[-1]
                    MAT[i] = mm                                                              
                    # Else there is a material in mats that is NOT in DU
                else:                                                                        
                    # This finds the indices of all elements that only appear in 
                    # mats and not DU
                    indices = np.in1d(matsARR,DU,invert=True)                                
                    # Randomly select one to be the current particle's material number
                    mm      = np.random.choice(matsARR[indices],1)
                    MAT[i]  = mm                                                             
                # Increment i
                i += 1                                                                       
        self.mats = list(MAT.astype(int))
        if populate: 
            target = self.hostmesh
            if target is not None: mesh = target
            target.fillAll(-1)
            target = psf.populateMesh(target,self)
        return 

    def fabricTensor_discs(self,tolerance=0.):
        """
        Calculates the fabric tensor of an Ensemble consisting of
        perfectly circular disks. They do NOT have to be identical in size!

        N.B. a tolerance is needed because circles are pixellated in iSALE (standard in 
        eulerian codes). A tolerance should be chosen to ensure the result is accurate;
        typically the cellsize or diagonal length of a cell is a good value.
    
        Args:
    
        tolerance: float; (see N.B. above)
    
        output:
    
            Z: The coordination number (average number of contacts/grain)
            A: The fabric Anisotropy
            F: The fabric tensor
        """
        # convert all radii and coords to arrays to aid the function
        a  = np.array(self.radii).astype(np.float64)
        a *= self.hostmesh.cellsize
        ic = np.array(self.xc)
        jc = np.array(self.yc)
        for shp in self.shapes:
            assert shp == 'circle', "ERROR: ALL grain shaps in an ensemble MUST be circles for this function to give an accurate answer"
        # simple function to compute all possible differences between elements in an array
        def difference_matrix(a):
            x = np.reshape(a, (len(a), 1))
            return x - x.transpose()
        # This function similarly finds all possible combinations of elements in an array.
        def addition_matrix(a):
            x = np.reshape(a, (len(a), 1))
            return x + x.transpose()
    
        # Number of particles
        Np = self.number
        # two arrays; one with all possible x-distances bewteen particles (li)
        # one with all possible y-distances (lj)
        # NB each has a shape of Np x Np and the matrices are 'flipped' sign-wise along the diagonal
        # and the diagonals are zeros
        li = difference_matrix(ic)
        lj = difference_matrix(jc)
        # s is all possible combinations of radii in the same order. i.e. 
        # elements of s are the minimum corresponding distances required for a 'contact'
        # Add a tolerance because we are not necessarily working with perfect circles, there is 
        # an uncertainty on the minimum length required for a contact 
        s  = addition_matrix(a) + tolerance
        # The magnitude (length) of each branch vector is
        L  = np.sqrt(li**2. + lj**2.)
        L[L==0] = 1.
        # normalise all branch vectors
        li /= L 
        lj /= L
        # set all branch vectors that are not contacts equal to 9999.
        li[L>=s] = 9999.
        lj[L>=s] = 9999.
        # Remove any branches of zero length (impossibilities) and flatten the arrays
        # NB every contact appears twice in each array!! Each has a size of 2*Nc (No. contacts)
        ni = li[li!=9999.]
        nj = lj[lj!=9999.]
        
        F = np.zeros((2,2))
        F[0,0] = np.sum(ni**2.)/float(Np)
        F[1,1] = np.sum(nj**2.)/float(Np)
        
        F[0,1] = np.sum(ni*nj)/float(Np)
        F[1,0] = F[0,1]
    
        Z = F[0,0] + F[1,1]
        A = F[0,0] - F[1,1]
        self.Z = Z
        self.A = A
        return Z, A, F

    def save(self):
        """save class as self.name.obj"""
        file = open(self.name+'.obj','w')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return

    def load(self):
        """try load self.name.obj"""
        file = open(self.name+'.obj','r')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)
        return

class SizeDistribution:
    """
    A size distribution is typically represented by a CDF (cumulative distribution function).
    This class creates one with user-specified CDF. CDFs are of the form 'frequency' vs 'var'
    and in granular distributions the independent variable is typically krumbein phi, or radius,
    however this class allows other types. 'frequency' is often volume (area in 2D) or weight.
    Both options are available, as is pure dimensionless frequency. Phi and area are the defaults.
    """
    #def __init__(self,freqtype = 'area', independent = 'phi'):
    #    self.freqtype = freqtype
    #    self.independent = independent

    def __init__(self,func=None,lims=None,mu=None,sigma=None,Lamb=None,k=None):
        self.func = func
        if callable(func): self.cdf = func

        elif func == 'uniform':
            assert lims is not None, "ERROR: function must have size limits (ind var)"
            assert mu is None and sigmas is None, "ERROR: uniform distribution has no mu or sigma params"
            self.lims = lims
            self.mean = .5*(lims[0]+lims[1])
            self.median = self.mean
            self.mode = None
            self.variance = (1./12.)*(lims[1]-lims[0])**2.
            self.skew = 0.
            self.cdf = self._uniform
        
        elif func == 'normal':
            assert mu is not None and sigma is not None, "ERROR: normal and lognormal defined by mu and sigma values"
            assert lims is None, "ERROR: normal has no lims at this stage"
            self.mu = mu
            self.sigma = sigma
            self.mean = mu
            self.median = mu
            self.mode = mu
            self.variance = sigma**2.
            self.skew = 0.
            self.cdf = self._normal
        
        elif func == 'lognormal':
            assert mu is not None and sigma is not None, "ERROR: normal and lognormal defined by mu and sigma values"
            assert lims is None, "ERROR: lognormal has no lims at this stage"
            self.mu = mu
            self.sigma = sigma
            self.mean = np.exp(mu+0.5*sigma**2.)
            self.median = np.exp(mu)
            self.mode = np.exp(mu-sigma**2.)
            self.variance = (np.exp(sigma**2.)-1.)*np.exp(2.*mu+sigma**2.)
            self.skew = (np.exp(sigma**2.)+2.)*np.sqrt(np.exp(sigma**2.) - 1.)
            self.cdf = self._lognormal

        elif func == 'weibull2':
            assert lims is None, "ERROR: lognormal has no lims at this stage"
            assert Lamb is not None and k is not None, "ERROR: Lamb and k must be defined for this distribution" 
            if Lamb < 0:
                warnings.warn("lambda must be >= 0, not {:2.2f}; setting to zero this time".format(Lamb))
                Lamb = 0.
            if k < 0.:
                warnings.warn("k must be >= 0, not {:2.2f}; setting to zero this time".format(k))
                k = 0.
            self.Lamb = Lamb
            self.k = k
            self.mean = Lamb*scsp.gamma(1.+1./k)
            self.median = Lamb*(np.log(2.))**(1./k)
            if k > 1:
                self.mode = Lamb*((k-1)/k)**(1./k)
            else:
                self.mode = 0
            self.variance = (Lamb**2.)*(scsp.gamma(1.+2./k)-(scsp.gamma(1.+1./k))**2.)
            self.skew = (scsp.gamma(1.+3./k)*Lamb**3. - 3.*self.mean*self.variance - self.mean**3.)
            self.skew /= self.variance**(3./2.)
            self.cdf = self._weibull2

    def details(self):
        deets = "distribution has the following properties:\n"
        if callable(self.func):
            deets += "type: user defined\n"
        elif self.func is not 'uniform':
            deets += "type: {}\n".format(self.func)
        else:
            deets += "type: uniform, over the range {}\n".format(self.lims)
        deets += "mean = {:2.3f}\n".format(self.mean)
        deets += "median = {:2.3f}\n".format(self.median)
        deets += "mode = {:2.3f}\n".format(self.mode)
        deets += "variance = {:2.3f}\n".format(self.variance)
        deets += "skewness = {:2.3f}\n".format(self.skew)
        return deets

    def frequency(self,x,dx):
        """
        Integrates over the probability density function of the chosen distribution to return an estimated frequency
        limits MUST be provided in the form of dx, which allows for uneven limits and is always applied as + and -
        the given value of x. Returns the probability DENSITY! this must be converted to a useful value outside of 
        the function.
        """
        if type(dx) != list: 
            warnings.warn('dx must be a list, set to list this time')
            dx = [dx*.5,dx*.5]
        if self.func == 'lognormal':
            assert x>=0., "ERROR: Lognormal distribution only works for input greater than 0"
        f = np.float64(abs(self.cdf(x+dx[1]) - self.cdf(x-dx[0])))
        return f

    
    def _uniform(self,x):
        """
        CDF for a uniform probability density function between minx and maxx
        """
        minx = self.lims[0]
        maxx = self.lims[1]
        f = (x-minx)/(maxx-minx)
        if x < minx: 
            f = 0.
        elif x >= maxx:
            f = 1.
        return f

    def _normal(self,x):
        """
        CDF for a normal probability density function centred on mu with std sigma
        """
        mu = self.mu
        sigma = self.sigma
        f = .5*(1.+scsp.erf((x-mu)/(sigma*np.sqrt(2.))))
        return f
    
    def _lognormal(self,x):
        """
        CDF for a log-normal probability density function centred on mu with std sigma
        """ 
        mu = self.mu
        sigma = self.sigma
        f = .5 + .5*scsp.erf((np.log(x)-mu)/(sigma*np.sqrt(2.)))
        return f

    def _weibull2(self,x):
        """
        CDF for a Weibull 2-parameter distribution; lambda is the 'scale' of the distribution
        k is the 'shape'. This distribution is typically used for PSDs generated by
        grinding, milling, and crushing operations.
        """
        Lamb = self.Lamb
        k = self.k
        if x >= 0:
            f = 1.-np.exp(-(x/Lamb)**k)
        else:
            f = 0.
        return f
