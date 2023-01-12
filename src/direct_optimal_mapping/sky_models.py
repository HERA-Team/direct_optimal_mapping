import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import constants as const
import copy

def byrne21(freq_mhz, file='/nfs/esc/hera/sky_models/byrne182MHz_2020/diffuse_map.healfits'):
    '''Generating the map in Byrne et al. 2021 at given frequency and nside512 in a HealPix 
    fashion. The map is at 182MHz originally and will be extrapolated to a target frequency 
    with the index of -2.61. The map is in mK.
    
    Parameters
    ----------
    freq_mhz: float
        frequency in MHz
    file: str
        location of the map file
        
    Return
    ------
    hp_map: 1d array
    '''
    with fits.open(file) as contents:
        nside_rb = contents[0].header['nside']
        ordering = contents[0].header['ordering']
        signal_data = contents[0].data
        freq_rb = contents[0].header['crval2']  # Frequency in MHz
        pixel_vals = contents[1].data['hpx_inds']
    byrne21_intensity = np.zeros(hp.nside2npix(nside_rb))
    byrne21_intensity[pixel_vals] = signal_data[:, 0, 0]
    byrne21_intensity = hp.reorder(byrne21_intensity, n2r=True)
    hp_map = byrne21_intensity * (freq_mhz/182)**(-2.61) # Mozdzen+16 index
    
    return nside_rb, hp_map

def gleam_catalog(freq_mhz, nside, verbose=True,
                  folder='/nfs/esc/hera/sky_models/gleam'):
    '''
    flux density is calculated from the GLEAM catalogs
    '''
    # Point source set up

    ## GLEAM egc point source selection
    hdul = fits.open(folder+'/GLEAM_EGC_v2.fits')
    hdr = hdul[0].header
    data = hdul[0].data
    hdr=hdul[1].header
    source_list = hdul[1].data
    hdul.close()

    ra_egc = np.radians(source_list['RAJ2000'])
    dec_egc = np.radians(source_list['DEJ2000'])

    int_flux200 = source_list['int_flux_fit_200']
    alpha = source_list['alpha']

    flux_jy_egc = int_flux200*(freq_mhz/200.)**(alpha)

    ## GLEAM galactic plane point source selection
    hdul = fits.open(folder+'/GLEAM_GAL.fits')
    hdr = hdul[0].header
    data = hdul[0].data
    hdr=hdul[1].header
    source_list = hdul[1].data
    hdul.close()

    ra_gal = np.radians(source_list['RAJ2000'])
    dec_gal = np.radians(source_list['DEJ2000'])

    int_flux200 = source_list['int_flux_fit_200']
    alpha = source_list['alpha']

    flux_jy_gal = int_flux200*(freq_mhz/200.)**(alpha)

    ## Bright sources
    t_src = Table.read(folder+'/A-team_bright_sources.txt', format='ascii')
    ra_bs = []
    dec_bs = []
    flux_jy_bs = []
    for i, source in enumerate(t_src['source_name']):
        flux_t = t_src['flux_jy'][i]*(freq_mhz/t_src['freq0_mhz'][i])**(t_src['index'][i])
        ra_t = np.radians(t_src['ra_deg'].data[i])
        dec_t = np.radians(t_src['dec_deg'].data[i])
        ra_bs.append(ra_t)
        dec_bs.append(dec_t)
        flux_jy_bs.append(flux_t)

    ra_all = np.concatenate((ra_egc, ra_gal, ra_bs))
    dec_all = np.concatenate((dec_egc, dec_gal, dec_bs))
    flux_jy_all = np.concatenate((flux_jy_egc, flux_jy_gal, flux_jy_bs))

    idx_sel = np.where(~np.isnan(flux_jy_all))[0]
    ra_sel = ra_all[idx_sel]
    dec_sel = dec_all[idx_sel]
    flux_sel = flux_jy_all[idx_sel]
    if verbose:
        print('%d valid point sources in the GLEAM + A team catalog@%.2fMHz.'%(len(idx_sel), freq_mhz))
    
    gleam_flux = np.zeros(hp.nside2npix(nside))
    source_idx = hp.ang2pix(nside, np.pi/2 - dec_sel, ra_sel)
    flux_jy_cp = copy.deepcopy(flux_sel)
    while len(source_idx) > 0:
        hp_no, idx_t  = np.unique(source_idx, return_index=True)
        gleam_flux[hp_no] += flux_jy_cp[idx_t]
        source_idx = np.delete(source_idx, idx_t)
        flux_jy_cp = np.delete(flux_jy_cp, idx_t)
    return {'ra': ra_sel, 'dec': dec_sel, 'flux': flux_sel}, gleam_flux