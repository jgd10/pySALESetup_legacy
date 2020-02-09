import numpy as np
from typing import Union, List, Dict, Iterable, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

    def insert_rectangle(self, material: int, xlims: Tuple[float, float], ylims: Tuple[float, float]):
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

