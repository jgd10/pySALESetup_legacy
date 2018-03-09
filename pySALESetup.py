import random
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.path   as mpath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
from PIL import Image

def polygon_area(X,Y):
    """
    Returns exact area of a polygon
    
    Args:
        X: [x coords] 
        Y: [y coords]
    Returns:
        A: Scalar float
    """
    N = np.size(X)
    A = 0
    for i in range(1,N):
        A += (X[i-1]*Y[i]-X[i]*Y[i-1])*.5
    return abs(A)

def combine_meshes(mesh2,mesh1,axis=1):
    """
    Combines two mesh classes, either horizontally or vertically and creates a new Mesh instance
    for the result. Material fractions are carried over, as are velocities, and the 'mesh' param.

    Args:
        mesh2: Mesh instance
        mesh1: Mesh instance

    Returns:
        New: new Mesh instance
    """
    assert mesh1.cellsize == mesh2.cellsize, "ERROR: meshes use different cellsizes {} & {}".format(mesh1.cellsize,mesh2.cellsize)
    if axis == 0: assert mesh1.y == mesh2.y, "ERROR: Horizontal merge; meshes must have same y; not {} & {}".format(mesh1.x,mesh2.x)
    if axis == 1: assert mesh1.x == mesh2.x, "ERROR: Vertical merge; meshes must have same x; not {} & {}".format(mesh1.y,mesh2.y)
    
    if axis == 0: 
        Xw = mesh1.x + mesh2.x
        Yw = mesh1.y
    if axis == 1: 
        Yw = mesh1.y + mesh2.y
        Xw = mesh1.x
    New = Mesh(X=Xw,Y=Yw,cellsize=2.e-6,mixed=False)
    New.materials = np.concatenate((mesh1.materials,mesh2.materials),axis=1+axis)
    New.mesh = np.concatenate((mesh1.mesh,mesh2.mesh),axis=axis)
    New.VX = np.concatenate((mesh1.VX,mesh2.VX),axis=axis)
    New.VY = np.concatenate((mesh1.VY,mesh2.VY),axis=axis)

    return New

def MeshfromPSSFILE(fname='meso_m.iSALE.gz',cellsize=2.e-6,NumMats=9):
    """
    Generate a Mesh instance from an existing meso output file. NB NumMats
    MUST be set explicitly because the function does not have the capbility 
    to read from file yet.

    Args: 
        fname: string
        cellsize: float
        NumMats: int
    """
    # import all fields from input file, NB coords must be int
    # in numpy genfromtxt & loadtxt handle .gz files implicitly
    CellInd = np.genfromtxt(fname,skip_header=1,usecols=(0,1)).astype(int)
    CellVel = np.genfromtxt(fname,skip_header=1,usecols=(2,3))
    # No. mats not necessarily 9, so use string to specify how many cols
    matcols = range(4,4+NumMats) 
    CellMat = np.genfromtxt(fname,skip_header=1,usecols=(matcols))
    
    # Extract mesh size from index cols. Indices start at 0, so want + 1
    # for actual size
    nx = int(np.amax(CellInd[:,1])+1)
    ny = int(np.amax(CellInd[:,0])+1)
    
    # Create the mesh instance & use cellsize as not stored in input file!!
    mesh = Mesh(X=nx,Y=ny,cellsize=cellsize)

    # initialise a counter (k) and cycle through all coords
    k = 0
    for j,i in CellInd:
        # At each coordinate cycle through all the materials and assign to mesh
        for m in range(NumMats):
            mesh.materials[m,i,j] = CellMat[k,m]
        # additionally assign each velocity as needed
        mesh.VX[i,j] = CellVel[k,0]
        mesh.VY[i,j] = CellVel[k,1]
        # increment counter
        k += 1
    # return Mesh instance at end
    return mesh

def MeshfromBMP(imname,cellsize=2.e-6):
    """
    Function that populates a Mesh instance from a bitmap, or similar.
    When opened by PIL the result MUST be convertible to a 2D array of
    grayscale values (0-255).

    Different shades are treated as different materials, however, white is ignored
    and treated as 'VOID'.

    NB bmp can NOT have colour info or an alpha channel.

    Args:
        A: 2D array of grayscale integer; black - white values (0 - 255)
        cellsize: float; equivalent to GRIDSPC, size of each cell
    Returns:
        mesh: Mesh
    """
    im = Image.open(imname) 
    B = np.asarray(im)
    A = np.copy(B)
    A = np.rot90(A,k=3)
    
    
    nx, ny = np.shape(A)
    #white is considered 'VOID' and should not be included
    ms = np.unique(A[A!=255])    
    Nms = np.size(ms)
    assert Nms <= 9, "ERROR: More than 9 different shades present (apart from white = void)"
    mesh = Mesh(nx,ny,cellsize=cellsize)
    
    m = 0
    for c in ms:
        mesh.materials[m][A==c] = 1.
        m += 1
    
    return mesh

def grainfromVertices(R=None,fname='shape.txt',mixed=False,eqv_rad=10.,rot=0.,min_res=5):
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

    Args:
        mixed:       logical; partially filled cells on or off                           
        rot:         float; rotation of the grain (radians)                            
        areascale:   float; Fraction between 0 and 1, indicates how to scale the grain 
        min_res:     int; Minimum resolution allowed for a grain                     
    Returns:
        mesh_:       square array with filled cells, with value 1

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



def grainfromCircle(r_):
    """
    This function generates a circle within the base mesh0. It very simply converts
    each point to a radial coordinate from the origin (the centre of the shape.
    Then assesses if the radius is less than that of the circle in question. If it 
    is, the cell is filled.
    
    Args:
        r_: radius of the circle, origin is assumed to be the centre of the mesh0

    Returns:
        mesh0: square array of floats
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

def grainfromEllipse(r_,a_,e_):
    """
    This function generates an ellipse in mesh0. It uses a semi-major axis of r_
    a rotation of a_ and an eccentricity of e_. It otherwise works on
    principles similar to those used in grainfromEllipse
    
    Args:
        r_ : int; the semi major axis (in cells)
        a_ : float; the angle of rotation (in radians)
        e_ : float; the eccentricity of the ellipse
    Returns:
        mesh0: square array of floats
    """
    N = int(2.*r_+2.)
    mesh0 = np.zeros((N,N))
    x0 = r_ + 1                                                                                   
    y0 = r_ + 1                                                                                   
    # A is the semi-major radius, B is the semi-minor radius
    A = r_
    B = A*np.sqrt(1.-e_**2.)                                                                                
    for j in range(N):
        for i in range(N):
            xc = 0.5*(i + (i+1)) - x0
            yc = 0.5*(j + (j+1)) - y0 
            
            xct = xc * np.cos(a_) - yc * np.sin(a_)
            yct = xc * np.sin(a_) + yc * np.cos(a_)
            r = (xct/A)**2. + (yct/B)**2.
            
            if r<=1:
                mesh0[j,i] = 1.
    return mesh0

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
            self.mesh = grainfromCircle(self.equiv_rad)
            self.area = np.sum(self.mesh)
        elif self.shape == 'file':
            assert File is not None, "No file path provided to create grain"
            self.mesh = grainfromVertices(fname=File,eqv_rad=self.equiv_rad,mixed=self.mixed,rot=self.angle)
            self.area = np.sum(self.mesh)
            self.radius = np.sqrt(self.area/np.pi)
        elif self.shape == 'ellipse':
            assert len(elps_params) == 2, "ERROR: ellipse creation requires elps_params to be of the form [major radius, eccentricity]"
            self.mesh = grainfromEllipse(elps_params[0],self.angle,elps_params[1])
            self.area = np.sum(self.mesh)
            self.eccentricity = elps_params[1]
            self.radius = np.sqrt(self.area/np.pi)
        elif self.shape == 'polygon':
            assert len(poly_params) >= 3, "ERROR: Polygon creation requires at least 3 unique coordinates"
            self.mesh = grainfromVertices(R=poly_params,eqv_rad=self.equiv_rad,mixed=self.mixed,rot=rot)
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
    def view(self):
        """ view the grain in a simple plot """
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        ax.imshow(self.mesh,cmap='binary')
        ax.set_xlabel('x [cells]')
        ax.set_ylabel('y [cells]')
        fig.show()
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

    def place(self,x,y,m,target,num=None):
        """
        Inserts the shape into the correct materials mesh at coordinate x, y.
        
        Args:
            x, y   : float; The x and y coords at which the shape is to be placed. (These are the shape's origin point)
            m      : int; this is the index of the material
            target : Mesh; the target mesh, must be a 'Mesh' instance
            num    : int; the 'number' of the particle.
        
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
        insert grain into bed in an empty space. By default select from whole mesh, 
        alternatively, choose coords from region bounded by xbounds = [xmin,xmax] and 
        ybounds = [ymin,ymax]. Position is defined by grain CENTRE
        so overlap with box boundaries is allowed.

        Args:
            target: Mesh
            m:      int; material number
            xbounds: list
            ybounds: list
        """
        target.mesh = np.sum(target.materials,axis=0)
        target.mesh[target.mesh>1.] = 1.
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
            x,y   = random.choice(indices)                                                                
            nospace, overlap = self._checkCoords(x,y,target)                                                          
            counter += 1                                                                                  
            if counter>10000:                                                                              
                nospace = True                                                                                
                passes= 1                                                                                 
                print "No coords found after {} iterations; exiting".format(counter)
                break
        if nospace:
            pass
        else:
            #self.x = x
            #self.y = y
            #self.mat = m
            self.place(x,y,m,target)
        return 
    def _checkCoords(self,x,y,target,overlap_max=0.):                                                                        
        """
        Checks if the grain will overlap with any other material;
        and if it can be placed.
        
        It works by initially checking the location of the generated coords and 
        ammending them if the shape overlaps the edge of the mesh. Then the two arrays
        can be compared.
        
        Args:
            x:                 float; The x coord of the shape's origin
            y:                 float; The equivalent y coord
            target:            Mesh; 
            overlap_max:       float; maximum number of overlapping cells allowed
        Returns:
            nospace:           boolean;
            overlapping_cells: float; number of cells that overlap with other grains

        the value of nospace is returned. False is a failure, True is success.
        """
        #cell_limit = (np.pi*float(cppr_max)**2.)/100.                                                
        nospace = False                                                                                    
        Is,Js,i_,j_ = self._cropGrain(x,y,target.x,target.y)
        
        temp_shape = np.copy(self.mesh[Is[0]:Is[1],Js[0]:Js[1]])                                    
        temp_mesh  = np.copy(target.mesh[i_[0]:i_[1],j_[0]:j_[1]])                                            
        test       = np.minimum(temp_shape,temp_mesh)                                                       

        overlapping_cells = np.sum(test)
                                                                                                            
        if (overlapping_cells > overlap_max):                                                                              
            nospace = True
        elif(overlapping_cells == 0):                                                                        
            pass
        return nospace, overlapping_cells

    def insertRandomwalk(self,target,m,xbounds=None,ybounds=None):
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
            if counter>100000:                                                                              
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
        self.phi.append(-np.log2(particle.equiv_rad))
        self.areas.append(particle.area)
        self.shapes.append(particle.shape)
    
    def details(self):
        """creates easily printable string of details of this instance"""
        deets  = "Ensemble instance called {}:\nGrain shapes: {}\n".format(self.name,np.unique(self.shapes))
        deets += "Number of grains: {}\n".format(self.number)
        deets += "Mean equivalent radius: {:3.2f} cells; std: {:3.2f} cells\n".format(np.mean(self.radii),np.std(self.radii))
        deets += "Materials used: {}\n".format(np.unique(self.mats))
        return deets

    def calcPSD(self,forceradii=False,returnplots=False):
        """
        Generates a Particle Size Distribution (PSD) from the Ensemble
        Args:
            returnplots: bool; if true the function returns ax1 ax2 and fig for further modding
            forceradii:  bool; if true use radii instead of phi
        """
        # use in-built functions to generate the frequencies and vfrac at this instance
        self.frequency()
        self._vfrac()
        
        # Frequency of each unique area
        areaFreq = np.array(self.areaFreq.values())
        # Array of the unique areas
        areas    = np.array(self.areaFreq.keys())
        # Probability Distribution Function
        PDF = areaFreq*areas/(self.hostmesh.Ncells*self.vfrac)
        # ... as a percentage of total volume fraction occupied
        PDF *= 100.
        # Cumulative Distribution Function
        CDF = np.cumsum(PDF)
        # krumbein phi, standard measure of grains ize in PSDs
        phi = np.unique(self._krumbein_phi())

        # create plot for the PSD
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        if forceradii:
            rad = np.unique(self.radii)
            ax1.plot(rad,PDF,marker='+',mfc='None',linestyle=' ',color='crimson',mew=1.5)
            ax2.plot(rad,CDF,marker='+',mfc='None',linestyle=' ',color='crimson',mew=1.5)
            ax1.set_xlabel('Equivalent Radius [mm]')
            ax2.set_xlabel('Equivalent Radius [mm]')
        else:
            ax1.plot(phi,PDF,marker='+',mfc='None',linestyle=' ',color='crimson',mew=1.5)
            ax2.plot(phi,CDF,marker='+',mfc='None',linestyle=' ',color='crimson',mew=1.5)
            ax1.set_xlabel('$\phi$' + '\n' + '$= -log_2(Equivalent Diameter)$')
            ax2.set_xlabel('$\phi$' + '\n' + '$= -log_2(Equivalent Diameter)$')
        ax1.set_title('PDF')
        ax2.set_title('CDF')
        ax1.set_ylabel('%.Area')
        
        # return plots if needed, otherwise show figure
        if returnplots:
            return ax1,ax2,fig
        else:
            fig.tight_layout()
            fig.show()

    def _krumbein_phi(self):
        """
        Convert all radii in the ensemble to the Krumbein Phi, which is commonly used in 
        particle size distributions.
        """
        phi = []
        # Krumbein phi = -log_2(D/D0) where D0 = 1 mm and D = diameter
        for r in self.radii:
            p = -np.log2(2*r)
            phi.append(p)
        return phi

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

    def area_weights():
        ensemble_area = _vfrac()*self.hostmesh.area
        self.area_weights = [100.*a/ensemble_area for a in self.areas]
    
    def optimise_materials(self,mats=np.array([1,2,3,4,5,6,7,8,9])):                        
        """
        This function has the greatest success and is based on that used in JP Borg's work with CTH.
    
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
        i = 0                                                                            
        # Loop every particle and assign each one in turn.    
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
    
class Mesh:
    """
    This is the domain class and it tracks all materials placed into it. Main features
    are the material fields--NB. of materials is ALWAYS 9 (the maximum). When saving,
    if a material is not used it is not included in the output file--, the velocity fields
    which include both x and y component fields, and the 'mesh' field. This acts like a test
    domain with only one material field and is mostly used internally.
    """
    def __init__(self,X=500,Y=500,cellsize=2.e-6,mixed=False):
        """
        Initialise the Mesh class. Defaults are typical for mesoscale setups which this module
        was originally designed for.

        Args:
            X:        int
            Y:        int
            cellsize: float; equivalent to GRIDSPC in iSALE
            mixed:    bool
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
        View velocities in a simple plot and save file if wanted.
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
        
        pvx = axX.pcolormesh(self.xi,self.yi,self.VX, cmap='PiYG',vmin=np.amin(self.VX),vmax=np.amax(self.VX))
        pvy = axY.pcolormesh(self.xi,self.yi,self.VY, cmap='coolwarm',vmin=np.amin(self.VY),vmax=np.amax(self.VY))
        
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
            ax.pcolormesh(self.xi,self.yi,matter, cmap='cividis',vmin=0,vmax=self.NoMats)
        ax.set_xlim(0,self.x)
        ax.set_ylim(0,self.y)
        ax.set_xlabel('$x$ [cells]')
        ax.set_ylabel('$y$ [cells]')
        if save: fig.savefig(fname,bbox_inches='tight',dpi=300)
        plt.show()

    def top_and_tail(self,num=3,axis=1):
        """
        Sets top and bottom 3 rows/columns to void cells. Recommended when edges moving away from boundary
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

    def blanketVel(self,vel,axis=1):
        """
        Assign a blanket velocity to whole domain. Useful before merging meshes or when using other objects in iSALE.
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

    def calcVol(self,m=None):
        """
        Calculate area of non-void in domain for material(s) m. 
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
        return Vol

    def matrixPorosity(self,matrix,bulk,void=False,Print=True):
        """
        calculate sthe necessary matrix porosity to achieve a target bulk porosity
        given current domain occupance.
        """
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


    def save(self,fname='meso_m.iSALE',noVel=False,info=False,compress=False):
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
        compress: compress the file? For large domains it is often necessary to avoid very large files; uses gz
        
        returns nothing but saves all the info as a txt file called 'fname' and populates the materials mesh.
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
        #print HEAD
        if noVel:
            ALL  = np.column_stack((XI,YI,FRAC.transpose()))                                               
        elif info:
            ALL  = np.column_stack((XI,YI,UX,UY,FRAC.transpose(),PI))                                       
        elif info and noVel:
            ALL  = np.column_stack((XI,YI,FRAC.transpose(),PI))                                              
        else:
            ALL  = np.column_stack((XI,YI,UX,UY,FRAC.transpose()))                                             
        if compress: fname += '.gz'
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
        Preference is given to materials already present in the target mesh and these are not overwritten
        if m == -1 then this erases all material it is placed on. To only overwrite certain materials set 
        m = 0 and specify which to overwrite in overwrite_mats. Any other materials will be left untouched.

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
            temp_materials[inrectangle*(np.sum(target.materials,axis=0)<1.)] = 1. #- np.sum(materials,axis=0)  
            temp_2 = np.sum(target.materials,axis=0)*temp_materials
            temp_materials -= temp_2
            target.materials[int(m)-1] += temp_materials
        return


print " ===================================================== "
print "               _______   __   ________    __           "
print "    ___  __ __/ __/ _ | / /  / __/ __/__ / /___ _____  "
print "   / _ \/ // /\ \/ __ |/ /__/ _/_\ \/ -_) __/ // / _ \ "
print "  / .__/\_, /___/_/ |_/____/___/___/\__/\__/\_,_/ .__/ "
print " /_/   /___/                                   /_/     "
print "                                      by J. G. Derrick "
print " ===================================================== "
