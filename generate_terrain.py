from itertools import zip_longest, chain
import numpy as np

# Parameters from Utnes2007mos
H = 0.229
r1 = 0.203 / H
a1 = 0.076 / H

# Number of points in each direction
Nx = 201
Ny = 101

# Domain bounding box
# Note: this is larger than the final domain because Simra's grid
# generator has its own bounding box input
ymin = xmin = -10 * H
ymax = xmax = 10 * H

x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
xx, yy = np.meshgrid(x, y)

r = np.sqrt(xx**2 + yy**2) / H
zz = H * (1.04 / (1 + r**4) - 0.083/(1 + (r-r1)**2 / a1**2) - 0.03)

# SIMRA Gridgen operates with a factor 10 resolution in altitude
zz *= 10

def grouper(iterable, n):
    """Iterate over elements of an iterable in groups of n."""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=('',)*3)

with open('map.dat', 'w') as f:
    # Important: 8 characters for Nx and Ny
    f.write('{: >8}{: >8}\n'.format(Nx, Ny))

    # Important: 9 values per line
    for pts in grouper(zip(xx.flat, yy.flat, zz.flat), 3):
        for value in chain(*pts):
            # Important: 10 characters per value
            f.write('{: >10.3}'.format(value))
        f.write('\n')


# gen2.dat and gen3.dat
xmin = -6 * H
xmax = 7 * H
ymin = -5 * H
ymax = 5 * H
zmax = 7 * H
nx = ny = 89
nz = 35
rmin = -2 * H
rmax = 2 * H
strength = 3.0
c1 = 0.4

with open('gen2.dat', 'w') as f:
    f.write(f'{xmin:.3f},{ymin:.3f},{xmax-xmin:.3f},{ymax-ymin:.3f},0.0\n')
    f.write('0,0\n')
    f.write(f'{nx},{ny}\n')
    f.write('3,3,2,3,1.0\n')
    f.write('200,0.001,1,1\n')
    f.write(f'{rmin:.3f},{rmax:.3f},{rmin:.3f},{rmax:.3f},{strength:.3f}\n')

with open('gen3.dat', 'w') as f:
    f.write(f'{nz},{nz},{zmax:.3f}\n')
    f.write(f'1.12,{c1:.3f}\n')
    f.write('1\n')
    f.write('0.0,0.0\n')
