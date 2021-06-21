This package provides a tool to map radio interferometric astronomical data
following the direct optimal mapping philosophy (Tegmark 1997; Dillon et al. 2015). This package is specifically developed
for the HERA project.

Installation
===========

This package can be installed via `pip`. After descending into the `direct_optimal_mapping` folder, please run:
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
dc.noise_calc()
dc.rm_flag()
opt_map = optimal_mapping.OptMapping(dc.uv_1d, nside, epoch='J2000')
```

The first line initiated the `DataConditioning` object as `dc`, where `uv` is one pyuvdata object (to be discussed more later). 
The `dc` object initiation stripped the `uv` object with 
one polarizatoin and one frequency.
The `noise_calc()` function calculates the visibility noise from the autocorrelations, the result is saved as a pyuvdata
object in the `dc` attribute `dc.uvn`. 
Then the the `rm_flag()` function removes flagged data considering both the flagged
data in the original pyuvdata object and the noise pyuvdata object. 
The last line initated the `OptimalMapping` object as `opt_map`, where `dc.uv_1d` is the processed pyuvdata object with one polarizatoin and one frequency, and 
flagged data removed, `nside` is the for the healpix map. In this example, the ephoch is also specified as 'J2000'.

Then another line can be run:

```python
opt_map.set_k_psf(radius_deg=50, calc_k=False)
```

where it sets up the range of the PSF. Note: K_PSF is not calculated here since
the `calc_k` flag is set as `False`, which saves memory.

After the previous preparations, the A matrix can be calculated via:

```python
opt_map.set_a_mat()
opt_map.set_inv_noise_mat(dc.uvn)
```

where the `set_a_mat()` function adds a `.a_mat` attribute as the A matrix; the `set_inv_noise_mat()` function uses the 
noise information in the `.uvn` object and adds a `.inv_noise_mat` 
attribute as the inverse N matrix. The A matrix is calculated within the range of the PSF.

Then the map can be generated with the data via:

```python
hmap = np.matmul(np.matrix(opt_map.a_mat.H), np.matmul(opt_map.inv_noise_mat, np.matrix(opt_map.data)))
```

Please note that hmap only covers the area within the PSF. The calculated pixels are a subset of the full-sky healpix
map at the corresponding nside. `opt_map.idx_psf_in` saves the index information of the hmap pixels in the full-sky healpix
map.

This only gives a brief introdution, more details can be explored in the data_conditioning.py 
and optimal_mapping.py files with the help of docstrings and comments.

## A-matrix with Point-source Pixels (upcoming...)

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
