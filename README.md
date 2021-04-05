This package provides a tool to map radio interferometric astronomical data
following the direct optimal mapping philosophy (Tegmark 1997; Dillon et al. 2015). This package is specifically developed
for the HERA project.

Installation
===========

This package can be installed via `pip`:
```
pip install .
```

Description
===========

## Short Introduction

Currently, the package contains two main classes `DataConditioning` and `OptMapping`.

The `DataConditioning` class provides necessary tools to select one frequency channel
and one polarization from the raw data. It can also perform redundant averaging and 
flagged-data removal. It will output a prepared pyuvdata object for the mapping part.

The `OptMapping` class contains the tools to calculate the mapping matrix (A-matrix) of
the instrument, along with auxilary product (including K_psf and K_facet). The class can
also calculate the point-spread matrix, aka P-matrix.

To use the classes, the python script file should first be imported:

```python
import optimal_mapping
import data_conditioning
```

To initiate one instance, one can do:

```python
ifreq = 758 #Frequency Channel No.
ipol = -5 # polarization number with -5 to -8 referring to XX, YY, XY, YX
dc = data_conditioning.DataConditioning(uv, ifreq, ipol)
uv_ready = dc.rm_flag(dc.uv_1d)
opt_map = optimal_mapping.OptMapping(uv_ready, nside, epoch='J2000')
```

where `uv` is one pyuvdata object (discussed more later),
`uv_ready` is the processed pyuvdata object with one polarizatoin and one frequency, and 
flagged data removed, `nside` is the for the healpix map. The ephoch is also specified as 
'J2000' here.

Then another line can be run:

```python
opt_map.set_k_psf(radius_deg=50, calc_k=False)
```

where it sets up the range of the PSF. Note: K_PSF is not calculated here since
the `calc_k` flag is set as `False`, which saves memory.

After the previous preparations, the A matrix can be calculated via:

```python
a_mat = opt_map.set_a_mat()
```

where the a_mat is calculated within the range of the PSF.

Then the map can be generated with the data via:

```python
hmap = np.matmul(a_mat.H, opt_map.data)
```

Please note that hmap only covers the area within the PSF.

This only gives a brief introdution, more details can be explored in the data_conditioning.py 
and optimal_mapping.py files with the help of docstrings and comments.

## uv Object
The uv object can be read in via:

```python
uv = UVData()
uv.read(filepath)
```
where `filepath` is the address to the .uvh5 file.
Then the uv object should include only 'cross' correlations and ordered
by baselines (this is because the internal calculation of the OptMapping object was set that way). 

An example is shown here:

```python
uv.select(ant_str='cross')
uv.reorder_blts('baseline')
```

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
# optimal_mapping
This repository stores code for optimal mapping within the HERA project. 
