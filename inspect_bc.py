import numpy as np
from scipy.io import FortranFile
import vtk

from simra_to_vtk import __main__ as s2vtk


def read_many(lines, n, tp, skip=True):
    if skip:
        next(lines)
    values = []
    while len(values) < n:
        values.extend(map(tp, next(lines).split()))
    return np.array(values)


def split_sparse(values):
    return values[0::2].astype(int), values[1::2]


def make_mask(n, indices, values=1.0):
    retval = np.zeros((n,))
    retval[indices-1] = values
    return retval


with FortranFile('mesh.dat', 'r') as f:
    npts, nelems, imax, jmax, kmax, _ = f.read_ints('u4')
    coords = f.read_reals(dtype='f4').reshape(npts, 3)
    elems = f.read_ints('u4').reshape(-1, 8) - 1

print(f'{imax} × {jmax} × {kmax}')


with open('boun.dat', 'r') as f:
    lines = iter(f)
    next(lines)                 # Skip 'Boundary conditions'
    *ints, z0 = next(f).split()
    nfixu, nfixv, nfixw, nfixp, nfixe, nfixk, nlog = map(int, ints)
    z0 = float(z0)

    z0_var = read_many(lines, nlog, float, skip=False)
    ifixu, fixu = split_sparse(read_many(lines, 2*nfixu, float))
    ifixv, fixv = split_sparse(read_many(lines, 2*nfixv, float))
    ifixw, fixw = split_sparse(read_many(lines, 2*nfixw, float))

print(f'{nfixu} {nfixv} {nfixw}')


grid = s2vtk.convert_grid(coords, elems)
pointdata = grid.GetPointData()
npts, _ = coords.shape
s2vtk.add_array(pointdata, make_mask(npts, ifixu), 'u-mask')
s2vtk.add_array(pointdata, make_mask(npts, ifixu, fixu), 'u-vals')
s2vtk.add_array(pointdata, make_mask(npts, ifixv), 'v-mask')
s2vtk.add_array(pointdata, make_mask(npts, ifixv, fixv), 'v-vals')
s2vtk.add_array(pointdata, make_mask(npts, ifixw), 'w-mask')
s2vtk.add_array(pointdata, make_mask(npts, ifixw, fixw), 'w-vals')

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName('hunt-bc.vtu')
writer.SetInputData(grid)
writer.Write()
