from itertools import product
import numpy as np
from scipy.io import FortranFile
import sys


def single_slice(total, axis, *args):
    index = [slice(None, None)] * total
    index[axis] = slice(*args)
    return tuple(index)


def unstagger(data, axis):
    index = [slice(None, None),] * data.ndim

    plus = list(index)
    plus[axis] = slice(1, None)
    plus = tuple(plus)

    minus = list(index)
    minus[axis] = slice(0, -1)
    minus = tuple(minus)

    return (
        data[single_slice(data.ndim, axis, 1, None)] +
        data[single_slice(data.ndim, axis, 0, -1)]
    ) / 2


def refine(data, axis):
    newshape = list(data.shape)
    newshape[axis] = newshape[axis] * 2 - 1
    retval = np.zeros_like(data, shape=tuple(newshape))
    retval[single_slice(data.ndim, axis, 0, None, 2)] = data
    retval[single_slice(data.ndim, axis, 1, None, 2)] = unstagger(data, axis)
    return retval


def structured_cells(*cellshape):
    nodeshape = tuple(s + 1 for s in cellshape)
    ranges = [range(k) for k in cellshape]
    nidxs = [np.array(q) for q in zip(*product(*ranges))]
    eidxs = np.zeros((len(nidxs[0]), 2**len(nidxs)), dtype='u4')
    i, j, k = nidxs

    eidxs[:,0] = np.ravel_multi_index((i, j, k), nodeshape)
    eidxs[:,3] = np.ravel_multi_index((i+1, j, k), nodeshape)
    eidxs[:,2] = np.ravel_multi_index((i+1, j+1, k), nodeshape)
    eidxs[:,1] = np.ravel_multi_index((i, j+1, k), nodeshape)
    eidxs[:,4] = np.ravel_multi_index((i, j, k+1), nodeshape)
    eidxs[:,7] = np.ravel_multi_index((i+1, j, k+1), nodeshape)
    eidxs[:,6] = np.ravel_multi_index((i+1, j+1, k+1), nodeshape)
    eidxs[:,5] = np.ravel_multi_index((i, j+1, k+1), nodeshape)

    return eidxs


if len(sys.argv) != 3:
    print('Usage: python3 refine.py [input] [output]')
    sys.exit(1)

_, infile, outfile = sys.argv


with FortranFile(infile, 'r', header_dtype='u4') as mesh:
    npts, nelems, imax, jmax, kmax, _ = mesh.read_ints('u4')
    nodes = mesh.read_reals('f4').reshape(jmax, imax, kmax, 3)
    cells = mesh.read_ints('u4').reshape(-1, 8) - 1


nodes = refine(nodes, axis=0)
nodes = refine(nodes, axis=1)
nodes = refine(nodes, axis=2)
cells = structured_cells(2*(jmax-1), 2*(imax-1), 2*(kmax-1)).reshape(-1, 8)


with FortranFile(outfile, 'w', header_dtype='u4') as mesh:
    mesh.write_record(np.array([nodes.size // 3, cells.size // 8, *nodes.shape[:3], 0], dtype='u4'))
    mesh.write_record(nodes.flatten())
    mesh.write_record(cells.flatten() + 1)
