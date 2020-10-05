# Hunt hill case for SIMRA

To run: ensure you have `simra` and `gridgen` on the path. Then

```
python3 generate_terrain.py
gridgen
gfortran ic_bc.f90 -o ic_bc
./ic_bc simra.in
simra simra.in
```
