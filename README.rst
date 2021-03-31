===============
optimal_mapping
===============


This package provides a tool to map radio interferometric astronomical data
following the optimal mapping philosophy. This package is specifically developed
for the HERA project.

Installation
===========
```pip install .
```

Description
===========

## Short Introduction
The optimal_mapping.py file contains the OptMapping class. 

To use the class, one should first import the python script file:

```python
import optimal_mapping
```

To initiate one instance, one can do:

```python
opt_map = optimal_mapping.OptMapping(uv, nside)
```

where `uv` is one pyuvdata/heradata object (discussed more later) with one polarizatoin and one frequency,
`nside` is the for the healpix map,

Then another two lines can be run sequentially:

```python
opt_map.set_k_psf(radius_deg=50, calc_k=False)
```

where the first line phase the data to the midpoint of the integration, and the
second line sets up the range of the PSF. Note: K_PSF is not calculated here since
the `calc_k` flag is set as `False`.

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

This only gives a brief introdution, more details can be explored in the optimal_mapping.py file
with the help of docstrings and comments.

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
