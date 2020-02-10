import numpy as np
from typing import Union, List, Dict, Iterable, Tuple, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.special as scsp


class Mesh:
    def __init__(self, mesh_length=500, mesh_width=500, cellsize=2.e-6, mixed=False, label='None'):
        self.width = mesh_width
        self.length = mesh_length
        self.cell_size = cellsize
        self._mixed = mixed
        self._name = label
        self._mats = ['mat {}'.format(i+1) for i in range(9)]
        self._mesh = np.array([[{'i': i,
                                 'j': j,
                                 'mat 1': 0.,
                                 'mat 2': 0.,
                                 'mat 3': 0.,
                                 'mat 4': 0.,
                                 'mat 5': 0.,
                                 'mat 6': 0.,
                                 'mat 7': 0.,
                                 'mat 8': 0.,
                                 'mat 9': 0.,
                                 'vel i': 0.,
                                 'vel j': 0.} for i in range(self.width)]
                               for j in range(self.length)])

    def __repr__(self):
        return '<Mesh: {} ({} x {})>'.format(self.name, self.width, self.length)

    def plot(self, view: bool = True, save: bool = False, file_name: str = None):
        if file_name is None:
            file_name = './{}.png'.format(self.name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(self.name)
        materials = {'mat {}'.format(i+1): np.zeros((self.width, self.length)) for i in range(9)}
        for cell in self._mesh.flatten():
            for i, mat in enumerate(self._mats):
                materials[mat][cell['i'], cell['j']] = cell[mat]*(i+1)

        x = np.linspace(0, self.width*self.cell_size, int(self.width))
        y = np.linspace(0, self.length*self.cell_size, int(self.length))
        xx, yy = np.meshgrid(x, y)
        for mat_mesh in materials.values():
            mat_mesh = np.ma.masked_where(mat_mesh == 0., mat_mesh)
            im = ax.pcolormesh(xx, yy, mat_mesh, vmin=1, vmax=9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('Material #')
        if save:
            fig.savefig(file_name)

        if view:
            plt.show()

    def save(self, file_name: str = None, compress: bool = True):
        if file_name is None:
            file_name = 'meso_m_{}.iSALE'.format(self.name)
        if compress:
            file_name += '.gz'
        num_cells = self._mesh.size
        columns = {'i': np.zeros((num_cells)),
                   'j': np.zeros((num_cells)),
                   'vel i': np.zeros((num_cells)),
                   'vel j': np.zeros((num_cells))}
        columns.update({'mat {}'.format(i+1): np.zeros((num_cells)) for i in range(9)})
        for i, cell in enumerate(self._mesh.flatten()):
            columns['i'][i] = cell['i']
            columns['j'][i] = cell['j']
            columns['vel i'][i] = cell['vel i']
            columns['vel j'][i] = cell['vel j']
            for j in range(9):
                mat = 'mat {}'.format(j+1)
                columns[mat][i] = cell[mat]
        all_ = np.column_stack((columns['i'],
                                columns['j'],
                                columns['vel i'],
                                columns['vel j'],
                                *[columns['mat {}'.format(i+1)] for i in range(9)]))
        head = '{}, {}'.format(num_cells, 9)
        np.savetxt(file_name, all_, header=head, fmt='%5.3f', comments='')

    @property
    def contains_mixed_cells(self):
        return self._mixed

    @property
    def name(self):
        return self._name

    def assign_velocity_to_rectangle(self, velocity: Tuple[float, float],
                                     xlims: Tuple[float, float],
                                     ylims: Tuple[float, float]):
        for point in points_within_rectangle(self.cell_size, xlims, ylims):
            cell = self._mesh[point]
            self.add_velocity_to_cell(velocity, cell)

    def assign_velocity_to_material(self, velocity: Tuple[float, float], material: int):
        assert 0 < material <= 9, "material must be an integer between 1 and 9"
        mat = 'mat {}'.format(material)
        for cell in self._mesh.flatten():
            if cell[mat] > 0.:
                self.add_velocity_to_cell(velocity, cell)

    def insert_rectangle(self, material: int, xlims: Tuple[float, float], ylims: Tuple[float, float], ):
        assert 0 <= material <= 9, "material must be an integer between 0 and 9 (0 = Void)"
        for point in points_within_rectangle(self.cell_size, xlims, ylims):
            cell = self._mesh[point]
            self.add_material_to_cell(material, cell)

    def insert_circle(self, material: int, centre: Tuple[float, float], radius: float):
        assert 0 <= material <= 9, "material must be an integer between 0 and 9 (0 = Void)"
        for point in points_within_circle(self.cell_size, centre, radius):
            cell = self._mesh[point]
            self.add_material_to_cell(material, cell)

    def insert_ellipse(self,
                       material: int,
                       centre: Tuple[float, float],
                       equivalent_radius: float,
                       eccentricity: float,
                       rotation: float = None):
        assert 0 <= eccentricity < 1., "Eccentricity must reside on the interval [0, 1)"
        assert 0 <= material <= 9, "material must be an integer between 0 and 9 (0 = Void)"
        for point in points_within_ellipse(self.cell_size, centre, equivalent_radius, eccentricity, rotation):
            cell = self._mesh[point]
            self.add_material_to_cell(material, cell)

    def add_material_to_cell(self, material: int, cell: Dict[str, Union[int, float]]):
        if material == 0:
            for mat in self._mats:
                cell[mat] = 0.
            cell['vel i'] = 0.
            cell['vel j'] = 0.
        else:
            current_mat_fraction = sum([cell[m] for m in self._mats])
            fraction_to_add = 1. - current_mat_fraction
            if fraction_to_add > 0.:
                cell['mat {}'.format(material)] += fraction_to_add

    @staticmethod
    def add_velocity_to_cell(velocity: Tuple[float, float], cell):
        cell['vel i'] = velocity[0]
        cell['vel j'] = velocity[1]


class SizeDistribution:
    """
    A size distribution is typically represented by a CDF (cumulative distribution function).
    This class creates one with user-specified CDF. CDFs are of the form 'frequency' vs 'var'
    and in granular distributions the independent variable is typically krumbein phi, or radius,
    however this class allows other types. 'frequency' is often volume (area in 2D) or weight.
    Both options are available, as is pure dimensionless frequency. Phi and area are the defaults.
    """
    def __init__(self, name: str):
        self._type = None
        self._mean = None
        self._std = None
        self._median = None
        self._mode = None
        self._variance = None
        self._skew = None
        self._cdf = None
        self._limits = None
        self._lambda = None
        self._k = None

    @classmethod
    def custom_distribution(cls, name: str, func: Callable):
        new = cls(name)
        new._func = func
        new._cdf = func
        new._type = 'custom'
        return new

    @classmethod
    def uniform_distribution(cls, name: str, size_limits: Tuple[float, float]):
        uniform = cls(name)
        uniform._type = 'uniform'
        uniform._limits = size_limits
        uniform._mean = .5*(sum(size_limits))
        uniform._median = uniform._mean
        uniform._variance = (1. / 12.) * (size_limits[1] - size_limits[0]) ** 2.
        uniform._cdf = uniform._uniform
        uniform._skew = 0.
        return uniform

    @classmethod
    def normal_distribution(cls, name: str, mean: float, standard_deviation: float):
        normal = cls(name)
        normal._type = 'normal'
        normal._mean = mean
        normal._std = standard_deviation
        normal._median = mean
        normal._mode = mean
        normal._variance = standard_deviation**2.
        normal._skew = 0.
        normal._cdf = normal._normal
        return normal

    @classmethod
    def lognormal_distribution(cls, name: str, mu: float, standard_deviation: float):
        lognormal = cls(name)
        lognormal._type = 'lognormal'
        lognormal._mean = np.exp(mu + 0.5 * standard_deviation ** 2.)
        lognormal._std = standard_deviation
        lognormal._median = np.exp(mu)
        lognormal._mode = np.exp(mu - standard_deviation ** 2.)
        lognormal._variance = (np.exp(standard_deviation ** 2.) - 1.) * np.exp(2. * mu + standard_deviation ** 2.)
        lognormal._skew = (np.exp(standard_deviation ** 2.) + 2.) * np.sqrt(np.exp(standard_deviation ** 2.) - 1.)
        lognormal._cdf = lognormal._lognormal
        return lognormal

    @classmethod
    def weibull2_distribution(cls, name: str, scale_parameter: float, shape_parameter: float):
        assert 0. <= scale_parameter, "the scale parameter must be >= 0, not {:2.2f}".format(scale_parameter)
        assert 0. <= shape_parameter, "the shape parameter must be >= 0, not {:2.2f}".format(shape_parameter)
        weibull2 = cls(name)
        weibull2._type = 'weibull2'
        weibull2._lambda = scale_parameter
        weibull2._k = shape_parameter
        weibull2._mean = scale_parameter * scsp.gamma(1. + 1. / shape_parameter)
        weibull2._median = scale_parameter * (np.log(2.)) ** (1. / shape_parameter)
        if shape_parameter > 1:
            weibull2._mode = scale_parameter * ((shape_parameter - 1) / shape_parameter) ** (1. / shape_parameter)
        else:
            weibull2._mode = 0
        weibull2._variance = (scale_parameter ** 2.) * \
                             (scsp.gamma(1. + 2. / shape_parameter) - (scsp.gamma(1. + 1. / shape_parameter)) ** 2.)
        weibull2._skew = (scsp.gamma(1. + 3. / shape_parameter) * scale_parameter ** 3. -
                          3. * weibull2._mean * weibull2._variance - weibull2._mean ** 3.)
        weibull2._skew /= weibull2._variance ** (3. / 2.)
        weibull2._cdf = weibull2._weibull2
        weibull2._type = 'weibull2'
        return weibull2

    def details(self):
        deets = "distribution has the following properties:\n"
        deets += "type: {}\n".format(self._type)
        deets += "mean = {:2.3f}\n".format(self._mean)
        deets += "median = {:2.3f}\n".format(self._median)
        deets += "mode = {:2.3f}\n".format(self._mode)
        deets += "variance = {:2.3f}\n".format(self._variance)
        deets += "skewness = {:2.3f}\n".format(self._skew)
        return deets

    def frequency(self, x: float, dx: Tuple[float, float]):
        """
        Integrates over the probability density function of the chosen distribution to return an estimated frequency
        limits MUST be provided in the form of dx, which allows for uneven limits and is always applied as + and -
        the given value of x. Returns the probability DENSITY! this must be converted to a useful value outside of
        the function.
        """
        if self._type == 'lognormal':
            assert x >= 0., "ERROR: Lognormal distribution only works for input greater than 0"
        f = np.float64(abs(self._cdf(x + dx[1]) - self._cdf(x - dx[0])))
        return f

    def _uniform(self, x: float):
        """
        CDF for a uniform probability density function between minx and maxx
        """
        assert self._limits is not None
        min_x = self._limits[0]
        max_x = self._limits[1]
        f = (x - min_x) / (max_x - min_x)
        if x < min_x:
            f = 0.
        elif x >= max_x:
            f = 1.
        return f

    def _normal(self, x: float):
        """
        CDF for a normal probability density function centred on mu with std sigma
        """
        mu = self._mean
        sigma = self._std
        f = .5 * (1. + scsp.erf((x - mu) / (sigma * np.sqrt(2.))))
        return f

    def _lognormal(self, x: float):
        """
        CDF for a log-normal probability density function centred on mu with std sigma
        """
        mu = self._mean
        sigma = self._std
        f = .5 + .5 * scsp.erf((np.log(x) - mu) / (sigma * np.sqrt(2.)))
        return f

    def _weibull2(self, x: float):
        """
        CDF for a Weibull 2-parameter distribution; lambda is the 'scale' of the distribution
        k is the 'shape'. This distribution is typically used for PSDs generated by
        grinding, milling, and crushing operations.
        """
        assert self._lambda is not None
        assert self._k is not None
        lamb = self._lambda
        k = self._k
        if x >= 0:
            f = 1. - np.exp(-(x / lamb) ** k)
        else:
            f = 0.
        return f


class Ensemble:
    def __init__(self, name: str, host_mesh: Mesh, size_distribution: SizeDistribution):
        self.name = name
        self._host = host_mesh
        self._dist = size_distribution


def rotate_point(x: float, y: float, rot: float):
    xct = x * np.cos(rot) - y * np.sin(rot)
    yct = x * np.sin(rot) + y * np.cos(rot)
    return xct, yct


def points_within_rectangle(cell_size: float, xlims: Tuple[float, float], ylims: Tuple[float, float]):
    # points do not have to snap to the grid

    xmin = xlims[0]/cell_size
    xmax = xlims[1]/cell_size

    ymin = ylims[0]/cell_size
    ymax = ylims[1]/cell_size

    imin, imax = int(xmin)-1, int(xmax)+1
    jmin, jmax = int(ymin)-1, int(ymax)+1

    valid_points = [(i, j)
                    for i in range(imin, imax)
                    for j in range(jmin, jmax)
                    if (xmin <= i+.5 <= xmax) and (ymin <= j+.5 <= ymax)]
    return valid_points


def points_within_circle(cell_size: float, centre: Tuple[float, float], radius: float):
    # points do not have to snap to the grid

    radius = (radius/cell_size)
    rad2 = radius**2.
    xcentre = centre[0]/cell_size
    ycentre = centre[1]/cell_size

    imin, imax = int(xcentre-1.1*radius), int(xcentre+1.1*radius)
    jmin, jmax = int(ycentre-1.1*radius), int(ycentre+1.1*radius)

    valid_points = [(i, j)
                    for i in range(imin, imax)
                    for j in range(jmin, jmax)
                    if ((i+.5-xcentre)**2. + (j+.5-ycentre)**2. <= rad2)]
    return valid_points


def points_within_ellipse(cell_size: float,
                          centre: Tuple[float, float],
                          equivalent_radius: float,
                          eccentricity: float,
                          rotation: float = None):

    # A is the semi-major radius, B is the semi-minor radius
    semi_major = equivalent_radius/((1. - eccentricity ** 2.) ** .25)
    semi_minor = semi_major * np.sqrt(1. - eccentricity ** 2.)

    semi_major /= cell_size
    semi_minor /= cell_size

    xcentre = centre[0]/cell_size
    ycentre = centre[1]/cell_size

    imin, imax = int(xcentre-1.5*semi_major), int(xcentre+1.5*semi_major)
    jmin, jmax = int(ycentre-1.5*semi_major), int(ycentre+1.5*semi_major)

    valid_points = []

    for i in range(imin, imax):
        for j in range(jmin, jmax):
            if rotation is not None:
                xc, yc = rotate_point(i+.5-xcentre, j+.5-ycentre, rotation)
            else:
                xc, yc = i+.5-xcentre, j+.5-ycentre
            if (xc/semi_major)**2. + (yc/semi_minor)**2. <= 1.:
                valid_points.append((i, j))

    return valid_points


def translate_vertices(vertices: List[Tuple[float, float]],
                       new_centroid: Tuple[float, float],
                       old_centroid: Tuple[float, float] = (0., 0.)):
    displacement = (new_centroid[0]-old_centroid[0], new_centroid[1]-old_centroid[1])
    new_vertices = [(x+displacement[0], y+displacement[1]) for x, y in vertices]
    return new_vertices


def points_within_polygon(cell_size: float,
                          vertices: Iterable[Tuple[float, float]],
                          rotation: float = None):

    try:
        from shapely.geometry import Polygon, Point
        from shapely import affinity
    except ImportError as exception:
        print('{}; Shapely must be installed to use points_within_polygon'.format(exception))
        raise

    verts = [(vx/cell_size, vy/cell_size) for vx, vy in vertices]
    poly = Polygon(verts)

    if rotation is not None:
        centroid = poly.centroid.coords[0]
        zero_verts = translate_vertices(verts, (0., 0.), old_centroid=centroid)
        rot_verts = [rotate_point(vx, vy, rotation) for vx, vy in zero_verts]
        verts = translate_vertices(rot_verts, centroid, old_centroid=(0., 0.))
        poly = Polygon(verts)

    imin, jmin, imax, jmax = poly.bounds
    imin, imax = int(imin), int(imax)
    jmin, jmax = int(jmin), int(jmax)

    valid_points = [(i, j)
                    for i in range(imin, imax)
                    for j in range(jmin, jmax)
                    if Point(i+.5, j+.5).within(poly)]

    return valid_points

