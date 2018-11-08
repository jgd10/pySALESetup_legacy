from __future__ import print_function
from __future__ import absolute_import

import random
import warnings
import numpy as np
from PIL import Image
from math import ceil
try:
    import cPickle as pickle
except:
    import pickle
import scipy.special as scsp
from scipy import stats as scst
import matplotlib.pyplot as plt
import matplotlib.path   as mpath
from collections import Counter, OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import domainclasses as psd
#from . import grainclasses as psg
#from . import objectclasses as pso

def quickFill(grain,target,volfrac,ensemble,
        material=1,method='insertion',xbnds=None,ybnds=None,nooverlap=False,mattargets=None):
    """
    Fills a region of mesh with the same grain instance. Most input values are identical to insertRandomly/insertRandomwalk
    because of this.
    Function does not have compatability with a material number of 0 yet, but it does work with -1. I.e. it can handle
    complete overwriting but not selective overwriting.
    
    Args:
        grain: Grain instance
        target: Mesh instance
        volfrac: float
        ensemble: Ensemble instance
        material: integer is one of -1, 1, 2, 3, 4, 5, 6, 7, 8, 9
        method: string
        xbnds: list (len == 2)  or None
        ybnds: list (len == 2)  or None
        nooverlap: bool
        mattargets: list or None
    Returns:
        Nothing returned

    """
    assert method == 'insertion' or method == 'random walk', 'ERROR: quickFill only supports insertion and random walk methods'
    assert material != 0, 'ERROR: function does not have advanced pore creation capability yet. This is to be integrated at a later date'
    if xbnds is None: xbnds = target._setboundsWholemesh(axis=0)
    if ybnds is None: ybnds = target._setboundsWholemesh(axis=1)
    # Set current volume fraction
    if material == -1 or material == 0: 
        current_volume_fraction = 1.-target.vfrac(xbounds=xbnds[:],ybounds=ybnds[:])
    else:
        current_volume_fraction = target.vfrac(xbounds=xbnds[:],ybounds=ybnds[:])

    if material == 0:
        pass
        # advanced pore-creation capability has not yet been integrated into this function
        # more will be added at a later date.

    # insert grains until target volume fraction achieved
    while current_volume_fraction <= volfrac:
        # insert randomly into a specified region as material 1 for now
        if method == 'insertion':
            grain.insertRandomly(target,material,xbounds=xbnds[:],ybounds=ybnds[:],nooverlap=nooverlap)
        elif method == 'random walk':
            grain.insertRandomwalk(target,material,xbounds=xbnds[:],ybounds=ybnds[:])
        # add Grain instance to Ensemble
        ensemble.add(grain)
        # calculate the new volume fraction in the new region
        prev_vfrac=current_volume_fraction
        if material == -1 or material == 0:
           current_volume_fraction = 1. -  target.vfrac(xbounds=xbnds[:],ybounds=ybnds[:])
        else:
           current_volume_fraction = target.vfrac(xbounds=xbnds[:],ybounds=ybnds[:])
    return 

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
    assert N==np.size(Y), "ERROR: x and y index arrays are unequal in size"
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
    # cellsize and mixed not important here because all material already placed and output is independent of cellsize
    New = psd.Mesh(X=Xw,Y=Yw,cellsize=mesh1.cellsize,mixed=False,label=mesh2.name+mesh1.name)
    New.materials = np.concatenate((mesh1.materials,mesh2.materials),axis=1+axis)
    New.mesh = np.concatenate((mesh1.mesh,mesh2.mesh),axis=axis)
    New.VX = np.concatenate((mesh1.VX,mesh2.VX),axis=axis)
    New.VY = np.concatenate((mesh1.VY,mesh2.VY),axis=axis)

    return New

def populateMesh(mesh,ensemble):
    """
    Populate a mesh, given an Ensemble.
    """
    # use information stored in the ensemble to repopulate domain
    # except NOW we can use the optimal materials from optimise_materials!
    for x,y,g,m in zip(ensemble.xc,ensemble.yc,ensemble.grains,ensemble.mats):
        g.remove()
        g.place(x,y,m,mesh)
    return mesh

def MeshfromPSSFILE(fname='meso_m.iSALE.gz',cellsize=2.5e-6,NumMats=9):
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
    mesh = psd.Mesh(X=nx,Y=ny,cellsize=cellsize)

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
    mesh = psd.Mesh(nx,ny,cellsize=cellsize)
    
    m = 0
    for c in ms:
        mesh.materials[m][A==c] = 1.
        m += 1
    
    return mesh

def grainfromVertices(R=None,fname='shape.txt',mixed=False,eqv_rad=10.,rot=0.,radians=True,min_res=4):
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
    if radians is not True: rot = rot*np.pi/180.
    assert eqv_rad > 0, "ERROR: Equivalent radius must be greater than 0!"
    # If no coords provided use filepath
    
    if R is None:
        J_ = np.genfromtxt(fname,comments='#',usecols=0,delimiter=',')
        I_ = np.genfromtxt(fname,comments='#',usecols=1,delimiter=',')
    # else use provided coords
    elif type(R) == list:
        R = np.array(R)
    if type(R) == np.ndarray:
        J_ = R[:,0]
        I_ = R[:,1]


    # if coords not yet normalised; normalise them onto the range -1. to 1.
    if np.amax(abs(I_)>1.) or np.amax(abs(J_))>1.:
        MAXI  = np.amax(I_)
        MINI  = np.amin(I_)
        MAXJ  = np.amax(J_)
        MINJ  = np.amin(J_)
        diffI = MAXI - MINI
        diffJ = MAXJ - MINJ

        # scale coords onto whichever coordinates have the largest difference
        if diffI>diffJ:
            I_ = 2.*(I_-MINI)/(MAXI-MINI) - 1.
            J_ = 2.*(J_-MINI)/(MAXI-MINI) - 1.
        else:
            I_ = 2.*(I_-MINJ)/(MAXJ-MINJ) - 1.
            J_ = 2.*(J_-MINJ)/(MAXJ-MINJ) - 1.

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
    assert r_>0, "ERROR: Radius must be greater than 0!"
    N = int(2.*ceil(r_)+2.)
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

def grainfromEllipse(r_,a_,e_,radians=True):
    """
    This function generates an ellipse in mesh0. It uses a semi-major axis of r_
    a rotation of a_ and an eccentricity of e_. It otherwise works on
    principles similar to those used in grainfromCircle
    
    Args:
        r_ : float; the equivalent radius of a circle with the same area
        a_ : float; the angle of rotation (in radians)
        e_ : float; the eccentricity of the ellipse
    Returns:
        mesh0: square array of floats
    """
    if radians is not True: a_ = a_*np.pi/180.
    assert e_ >= 0 and e_ < 1, "ERROR: eccentricity can not be less than 0 and must be less than 1; {} is not allowed".format(e_)
    assert r_>0, "ERROR: Radius must be greater than 0!"
    N = int(2.*ceil(r_)+2.)
    mesh0 = np.zeros((N,N))
    x0 = r_ + 1                                                                                   
    y0 = r_ + 1                                                                                   
    # A is the semi-major radius, B is the semi-minor radius
    A = r_/((1.-e_**2.)**.25)
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
