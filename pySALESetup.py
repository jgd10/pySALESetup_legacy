import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.path   as mpath

def polygon_area(X,Y):
    N = np.size(X)
    A = 0
    for i in range(1,N):
        A += (X[i-1]*Y[i]-X[i]*Y[i-1])*.5
    return abs(A)

def gen_shape_fromvertices(R=None,fname='shape.txt',mixed=False,eqv_rad=10.,rot=0.,min_res=5):
    """
    This function generates a mesh0 from a text file containing a list of its vertices
    in normalised coordinates over a square grid of dimensions 1 x 1. Centre = (0,0)
    coordinates must be of the form:
    j   i
    x   x
    x   x
    x   x
    .   .
    .   .
    .   .
    and the last coordinate MUST be identical to the first. Additionally function will take
    an array R instead, of the same form.
    --------------------------------------------------------------------------
    |kwargs    |  Meaning                                                    | 
    --------------------------------------------------------------------------
    |mixed     |  partially filled cells on or off                           |
    |rot       |  rotation of the grain (radians)                            |
    |areascale |  Fraction between 0 and 1, indicates how to scale the grain |
    |min_res   |  Minimum resolution allowed for a grain                     |
    --------------------------------------------------------------------------

    """
    # If no coords provided use filepath
    
    if R is None:
        J_ = np.genfromtxt(fname,comments='#',usecols=0,delimiter=',')
        I_ = np.genfromtxt(fname,comments='#',usecols=1,delimiter=',')
    # else use provided coords
    else:
        if type(R) == list:
            R = np.array(R)
        J_ = R[:,0]
        I_ = R[:,1]

    # if coords not yet normalised; normalise them
    if np.amax(I_)>1. or np.amax(J_)>1.:
        MAXI  = np.amax(abs(I_))
        MAXJ  = np.amax(abs(J_))
        MAX   = max(MAXI,MAXJ)
        J_   /= MAX 
        I_   /= MAX 

    # last point MUST be identical to first; append to end if necessary
    if J_[0] != J_[-1]:
        J_ = np.append(J_,J_[0])
        I_ = np.append(I_,I_[0])
    
    # equivalent radius is known and polygon area is known
    # scale shape as appropriate
    radius = np.sqrt(polygon_area(I_,J_)/np.pi)
    lengthscale = eqv_rad/radius
    J_   *= lengthscale
    I_   *= lengthscale

    # rotate points according by angle rot 
    theta = rot 
    ct    = np.cos(theta)
    st    = np.sin(theta)
    J     = J_*ct - I_*st
    I     = J_*st + I_*ct

    # find max radii from centre and double it for max width
    radii    = np.sqrt(I**2+J**2)
    maxwidth = int(2*np.amax(radii)+2)
    maxwidth = max(maxwidth,min_res)
    if maxwidth%2!=0: maxwidth+=1
    # Add double max rad + 1 for mini mesh dims
    mesh_ = np.zeros((maxwidth,maxwidth))

    # define ref coord as 0,0 and centre to mesh_ centre
    qx = 0.                                                                                 
    qy = 0.                                                                                 
    y0 = float(maxwidth/2.)
    x0 = y0
    
    I += x0
    J += y0
    path = mpath.Path(np.column_stack((I,J)))
    for i in range(maxwidth):
        for j in range(maxwidth):
            in_shape = path.contains_point([i+.5,j+.5])
            if in_shape and mixed == False: mesh_[i,j] = 1.
            elif in_shape and mixed == True:
                for ii in np.arange(i,i+1,.1):
                    for jj in np.arange(j,j+1,.1):
                        in_shape2 = path.contains_point([ii+.05,jj+.05])
                        if in_shape2: mesh_[i,j] += .01

    return mesh_

def gen_circle(r_):
    """
    This function generates a circle within the base mesh0. It very simply converts
    each point to a radial coordinate from the origin (the centre of the shape.
    Then assesses if the radius is less than that of the circle in question. If it 
    is, the cell is filled.
    
    r_ : radius of the circle, origin is assumed to be the centre of the mesh0
    
    mesh0 and an AREA are returned
    """
    N = int(2.*r_+2.)
    mesh0 = np.zeros((N,N))
    x0 = r_ + 1.                                                                                   
    y0 = r_ + 1.                                                                                   
    for j in range(N):                                                                                 
        for i in range(N):                        
            xc = 0.5*(i + (i+1)) - x0                                                                   
            yc = 0.5*(j + (j+1)) - y0                                                                   
            
            r = (xc/r_)**2. + (yc/r_)**2.                                                               
            if r<=1:                                                                                    
                mesh0[j,i] = 1.0                                                                        
    return mesh0

def gen_ellipse(r_,a_,e_):
    """
    This function generates an ellipse in mesh0. It uses a semi-major axis of r_
    a rotation of a_ and an eccentricity of e_. It otherwise works on
    principles similar to those used in gen_circle.
    
    r_ : the semi major axis (in cells)
    a_ : the angle of rotation (in radians)
    e_ : the eccentricity of the ellipse
    """
    mesh0 = np.zeros((2*r_+2,2*r_+2))
    x0 = r_ + 1                                                                                   
    y0 = r_ + 1                                                                                   
    # A is the semi-major radius, B is the semi-minor radius
    A = r_
    B = A*np.sqrt(1.-e_**2.)                                                                                
    for j in range(Ns):
        for i in range(Ns):
            xc = 0.5*(i + (i+1)) - x0
            yc = 0.5*(j + (j+1)) - y0 
            
            xct = xc * np.cos(a_) - yc * np.sin(a_)
            yct = xc * np.sin(a_) + yc * np.cos(a_)
            r = (xct/A)**2. + (yct/B)**2.
            
            if r<=1:
                mesh0[j,i] = 1.
    return mesh0

class Grain:
    def __init__(self, eqr=10., rot=0., shape='circle', File=None, trng_coords=None, rect_coords=None, elps_params=None, poly_params=None, mixed=False):
        self.equiv_rad = eqr
        self.angle = rot
        self.shape = shape
        self.mixed = mixed
        self.hostmesh = None
        if self.shape == 'circle' or self.shape =='sphere':
            self.radius = self.equiv_rad
            self.mesh = gen_circle(self.equiv_rad)
        elif self.shape == 'file':
            assert File is not None, "No file path provided to create grain"
            self.mesh = gen_shape_fromvertices(fname=File,eqv_rad=self.equiv_rad,mixed=self.mixed,rot=self.angle)
        elif self.shape == 'ellipse':
            assert len(elps_params) == 2, "ERROR: ellipse creation requires elps_params to be of the form [major radius, eccentricity]"
            self.mesh = gen_ellipse(elps_params[0],self.angle,elps_params[1])
        elif self.shape == 'polygon':
            assert len(poly_params) >= 3, "ERROR: Polygon creation requires at least 3 unique coordinates"
            self.mesh = gen_shape_fromvertices(R=poly_params,eqv_rad=self.equiv_rad,mixed=self.mixed,rot=rot)
        else:
            print "ERROR: unsupported string -- {} --  used for shape.".format(self.shape)
        # more to be added...
        self.Px,self.Py = np.shape(self.mesh)
    def view(self):
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        ax.imshow(self.mesh,cmap='binary')
        fig.show()
        ax.cla()
        
    def cropGrain(self,x,y,Px,Py,LX,LY):
        """
        If a grain is partially out of the mesh it needs to be cropped appropriately before 
        a check is done or it is placed in. That is the purpose of this function.
        x,y   : the coords of the grain placement (in cells)
        Px,Py : shape of the grain's mini-mesh, in cells (the actual grain is not needed at this stage)
        LX,LY : x-width and y-width (in cells) of the target mesh
        _____________________________________________________________
        returns
        _____________________________________________________________
        Is,Js : indices for the target mesh
        i_,j_ : indices for the minimesh

        """
        i_edge = x - Px/2
        j_edge = y - Py/2
        i_finl = x + Px/2
        j_finl = y + Py/2
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
            i_finl   = LX-1
        J_final = Py
        if (j_finl)>LY:
            J_final -= abs(LY-j_finl) 
            j_finl   = LY-1
        #print i_edge,j_edge
        Is = [I_initial,I_final]
        Js = [J_initial,J_final]
        i_ = [i_edge,i_finl]
        j_ = [j_edge,j_finl]
        return Is,Js,i_,j_

    def place(self,x,y,m,target,num=None):
        """
        This function inserts the shape  into the
        correct materials mesh at coordinate x, y.
        
        x, y   : The x and y coords at which the shape is to be placed. (These are the shape's origin point)
        m      : this is the index of the material
        target : the target mesh, must be a 'Mesh' instance
        num    : the 'number' of the particle.
        
        existing material takes preference and is not displaced

        nothing is returned.
        """
        if self.mixed: 
            assert num is None, "ERROR: Particle number not yet supported in mixed-cell systems"
        else:
            if num is None: num = 1.
        assert type(x)==type(y), "ERROR: x and y coords must have the same type"
        assert abs(m) > 0, "ERROR: material number must not be 0. -1 is void."
        self.x = x
        self.y = y
        if type(x)==int and type(y)==int:
            self.x = x*target.cellsize
            self.y = y*target.cellsize
        else:
            x = int(x/target.cellsize)
            y = int(y/target.cellsize)
        if type(m) == float or type(m) == np.float64: m = int(m)
        Is,Js,i_,j_ = self.cropGrain(x,y,self.Px,self.Py,target.x,target.y)
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


    def insertRandomly(self,target,m,xbounds=None,ybounds=None):
        """
        insert grain into bed in an unoccupied region of void. By default use whole mesh, alternatively
        choose coords from region bounded by xbounds = [xmin,xmax] and ybounds = [ymin,ymax]
        """
        box = None
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
        while nospace:                                                                                 
            y,x   = random.choice(indices)                                                                
            nospace = checkCoords_full(x,y,target)                                                          
            counter += 1                                                                                  
            if counter>5000:                                                                              
                nospace = True                                                                                
                passes= 1                                                                                 
                print "No coords found after {} iterations; exiting".format(counter)
                break
        if nospace:
            pass
        else:
            self.x = x
            self.y = y
            Is,Js,i_,j_ = self.cropGrain(self.Px,self.Py,target.x,target.y)
            self.materials[m-1, Is[0]:Is[1],Js[0],Js[1]] = self.mesh[i_[0]:i_[1],j_[0]:j_[1]]
        return 
    def checkCoords(self,x,y,target):                                                                        
        """
        This function checks if the grain will overlap with any other material;
        and if it can be placed.
        
        It works by initially checking the location of the generated coords and 
        ammending them if the shape overlaps the edge of the mesh. Then the two arrays
        can be compared.
        
        shape : the array based on mesh0 containg the shape
        x     : The x coord of the shape's origin
        y     : The equivalent y coord
        
        the value of nospace is returned. False is a failure, True is success.
        """
        #cell_limit = (np.pi*float(cppr_max)**2.)/100.                                                
        nospace = False                                                                                    
        Is,Js,i_,j_ = self.cropGrain(x,y,self.Px,self.Py,target.x,target.y)
        
        temp_shape = np.copy(shape[Js[0]:Js[1],Is[0]:Is[1]])                                    
        temp_mesh  = np.copy(mesh[j_[0]:j_[1],i_[0]:i_[1]])                                            
        test       = np.minimum(temp_shape,temp_mesh)                                                       
                                                                                                            
        if (np.sum(test) > 0):                                                                              
            nospace = True
            pass                                                                                            
        elif(np.sum(test) == 0):                                                                            
            pass
        return nospace                                                                                  

    def insert_randomwalk(self):
        self.hostmesh = target
        return

class Ensemble:
    def __init__(self,grains=[],rots=[],radii=[],hostmesh=[]):
        self.grains = grains
        self.number = 0
        self.rots   = rots
        self.eqiv_r = radii
        self.host_mesh = hostmesh

    def add(self,particle):
        assert particle.hostmesh is not None, "ERROR: grain must be placed in a mesh to be part of an ensemble"
        self.grains.append(particle)
        self.number += 0
        self.rots.append(particle.rot)
        self.radii.append(particle.equiv_rad)
        self.host_mesh.append(particle.hostmesh)
    
class Mesh:
    def __init__(self,X=500,Y=500,mats=9,cellsize=2.e-6,mixed=False):
        self.x = X
        self.y = Y
        self.Ncells = X*Y
        self.width = X*cellsize
        self.height = Y*cellsize
        self.xc = np.arange(X)+0.5
        self.yc = np.arange(Y)+0.5
        self.yi, self.xi = np.meshgrid(self.yc,self.xc)
        self.xx, self.yy = self.xi*cellsize,self.yi*cellsize
        self.mesh = np.zeros((X,Y))
        self.materials = np.zeros((mats,X,Y))
        self.VX = np.zeros((X,Y))
        self.VY = np.zeros((X,Y))
        self.cellsize = cellsize
        self.NoMats = mats
        self.mats = range(1,mats+1)
        self.mixed = mixed
    def checkVels(self):
        total = np.sum(self.materials,axis=0)
        # make sure that all void cells have no velocity
        self.VX[total==0.] = 0.
        self.VY[total==0.] = 0.
    
    def viewVels(self,save=False,fname='vels.png'):
        self.checkVels()
        fig = plt.figure()
        if self.x > self.y:
            subplotX, subplotY = 211, 212
        else:
            subplotX, subplotY = 121, 122
        axX = fig.add_subplot(subplotX,aspect='equal')
        axY = fig.add_subplot(subplotY,aspect='equal')
        axX.pcolormesh(self.xi,self.yi,self.VX, cmap='coolwarm',vmin=np.amin(self.VY),vmax=np.amax(self.VX))
        axY.pcolormesh(self.xi,self.yi,self.VY, cmap='coolwarm',vmin=np.amin(self.VY),vmax=np.amax(self.VY))
        axX.set_title('$V_x$')
        axY.set_title('$V_y$')
        for ax in [axX,axY]:
            ax.set_xlim(0,self.x)
            ax.set_ylim(0,self.y)
            ax.set_xlabel('$x$ [cells]')
            ax.set_ylabel('$y$ [cells]')
        fig.tight_layout()
        if save: fig.savefig(fname,bbox_inches='tight',dpi=300)
        plt.show()

    def viewMats(self,save=False,fname='mats.png'):
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        for KK in range(self.NoMats):
            matter = np.copy(self.materials[KK,:,:])*(KK+1)
            matter = np.ma.masked_where(matter==0.,matter)
            ax.pcolormesh(self.xi,self.yi,matter, cmap='plasma',vmin=0,vmax=self.NoMats)
        ax.set_xlim(0,self.x)
        ax.set_ylim(0,self.y)
        ax.set_xlabel('$x$ [cells]')
        ax.set_ylabel('$y$ [cells]')
        if save: fig.savefig(fname,bbox_inches='tight',dpi=300)
        plt.show()

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

    def plateVel(self,ymin,ymax,vel,axis=0):
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

    def calcVol(self,m=None):
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
        return Vol

    def matrixPorosity(self,matrix,bulk,void=False,Print=True):
        # bulk porosity must be taken as a percentage!
        bulk /= 100.
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
            print "porosity = {:3.3f}% and distension = {:3.3f}".format(matrix_por*100.,distension)
        return matrix_por*100.


    def save(self,fname='meso_m.iSALE',noVel=False,info=False):
        """
        A function that saves the current mesh as a text file that can be read, verbatim into iSALE.
        This compiles the integer indices of each cell, as well as the material in them and the fraction
        of matter present. It saves all this as the filename specified by the user, with the default as 
        meso_m.iSALE
        
        This version of the function works for continuous and solid materials, such as a multiple-plate setup.
        It does not need to remake the mesh as there is no particular matter present.
        
        fname   : The filename to be used for the text file being used
        mixed   : Are mixed cells used?
        noVel   : Does not include velocities in meso_m.iSALE file
        info    : Include particle ID (i.e. #) as a column in the final file 
        
        returns nothing but saves all the info as a txt file called 'fname' and populates the materials mesh.
        """
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
                if info:
                    PI[K] = self.mesh[j,i]
                for mm in range(NM):
                    FRAC[mm,K] = self.materials[usedmats[mm]-1,j,i]
                K += 1
        FRAC = self._checkFRACs(FRAC)
        HEAD = '{},{}'.format(K,NM)
        print HEAD
        if noVel:
            ALL  = np.column_stack((XI,YI,FRAC.transpose()))                                               
        elif info:
            ALL  = np.column_stack((XI,YI,UX,UY,FRAC.transpose(),PI))                                       
        elif info and noVel:
            ALL  = np.column_stack((XI,YI,FRAC.transpose(),PI))                                              
        else:
            ALL  = np.column_stack((XI,YI,UX,UY,FRAC.transpose()))                                             
        
        np.savetxt(fname,ALL,header=HEAD,fmt='%5.3f',comments='')
    
    def _checkFRACs(self,FRAC):
        """
        This function checks all the volume fractions in each cell and deals with any occurrences where they add to more than one
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
        
#class subMesh(Mesh):
    #    def __init__(self,mesh,R0,R1,I0,I1):




class Polygonal_object:
    def __init__(self,xcoords,ycoords,m,target):
        # last point MUST be identical to first; append to end if necessary
        if xcoords[0] != xcoords[-1]:
            xcoords = np.append(xcoords,xcoords[0])
            ycoords = np.append(ycoords,ycoords[0])
        self.x = xcoords
        self.y = ycoords
        self.hostmesh = target
        self.material = m

        """
        This function fills all cells within a polygon defined by the vertices in arrays 
        xcoords and ycoords
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ~if mat == -1. this fills the cells with VOID and overwrites everything.~
        ~and additionally sets all velocities in those cells to zero            ~
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """
        path = mpath.Path(np.column_stack((xcoords,ycoords)))
        # find the box that entirely encompasses the polygon
        L1 = np.amin(xcoords) 
        L2 = np.amax(xcoords)
        T1 = np.amin(ycoords)
        T2 = np.amax(ycoords)
        
        # find the coordinates of every point in that box
        Xc_TEMP = XX[(target.xc<=L2)*(target.xc>=L1)*(target.yc<=T2)*(target.yc>=T1)]
        Yc_TEMP = YY[(target.xc<=L2)*(target.xc>=L1)*(target.yc<=T2)*(target.yc>=T1)]

        # store all indices of each coord that is in the polygon in two arrays
        x_success = np.ones_like(Xc_TEMP,dtype=int)*-9999
        y_success = np.ones_like(Yc_TEMP,dtype=int)*-9999

        # cycle through all these points
        # store successes in arrays
        k = 0
        for x, y in zip(Xc_TEMP,Yc_TEMP):
            in_shape = path.contains_point([x,y])
            if in_shape:
                x_success[k]   = np.where((XX==x)*(YY==y))[0][0] 
                y_success[k]   = np.where((XX==x)*(YY==y))[1][0]
                k += 1
        x_suc = x_success[x_success!=-9999] 
        y_suc = y_success[y_success!=-9999] 

        if m == -1:
            # select the necessary material using new arrays of indices
            for mm in range(target.NoMats):
               materials[mm][x_suc,y_suc] *= 0.
            target.VX[x_suc,y_suc] *= 0.
            target.VY[x_suc,y_suc] *= 0.
            target.mesh[x_suc,y_suc] *= 0.
        else:
            temp_materials = np.copy(target.materials[int(m)-1,x_suc,y_suc])
            temp_materials[(np.sum(target.materials[:,x_suc,y_suc],axis=0)<1.)] = 1. #- np.sum(materials,axis=0)  
            temp_2 = np.sum(target.materials[:,x_suc,y_suc],axis=0)*temp_materials
            temp_materials -= temp_2
            target.materials[int(m)-1,x_suc,y_suc] += temp_materials

