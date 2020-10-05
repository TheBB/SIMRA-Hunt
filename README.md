# Hunt hill case for SIMRA

To run:

```
python3 generate_terrain.py
gfortran ic_bc.f90 -o ic_bc
./ic_bc simra.in
simra simra.in
```
