import healpy as hp
import numpy as np

hera_dec = -30.721526120689507

def set_facet_idx_disc(NSIDE, cen_pos, radius):
    '''Calculate the healpix indices of a circular facet centred at a given position

    Input:
    ------
    NSIDE: int
        The nside of the Healpix map
    cen_pos: int, 2-tuple, or 3-tuple
        The central position of the disc, can be a healpix index (int), longitude and latitude in degree (2-tuple), or vector coordinates defining the point on the sphere (3-tuple).
    radius: float (in degree)
        radius to be included in the facet

    Output:
    ------
    idx_facet_in: 1d array (int)
        healpix map indices within the facet
    '''
    try:
        _l = len(cen_pos)
        if _l == 3:
            cen_vec = cen_pos
        elif _l == 2:
            cen_vec = hp.ang2vec(cen_pos[0], cen_pos[1], lonlat = True)
        else:
            raise ValueError("Invalid argument length for cen_pos")
    except TypeError:
        assert cen_pos <= hp.nside2npix(NSIDE), "Invalid pixel number"
        cen_vec = hp.pix2vec(NSIDE, cen_pos)

    return hp.query_disc(NSIDE, cen_vec, np.deg2rad(radius), inclusive = True)

def set_psf_idx(NSIDE, start_lst, end_lst, radius = 90):
    '''Calculate the healpix indices of the combined PSF region for a given integation time
    ***Note: is only exact when the radius is set to be 90 deg

    Input:
    ------
    NSIDE: int
        The nside of the Healpix map
    start_lst: float (in radian)
        The starting local sidereal time
    end_lst: float (in radian)
        The ending local sidereal time
    radius: float (in degree)
        radius to be included in the PSF

    Output:
    ------
    idx_psf_in: 1d array (int)
        healpix map indices within the combined PSF
    '''
    start_psf = hp.query_disc(NSIDE, 
                              hp.ang2vec(np.rad2deg(start_lst), hera_dec, lonlat=True), 
                              np.deg2rad(radius), 
                              inclusive = True
    )
    end_psf = hp.query_disc(NSIDE, 
                            hp.ang2vec(np.rad2deg(end_lst), hera_dec, lonlat=True), 
                            np.deg2rad(radius), 
                            inclusive = True
    )
    return np.unique( np.concatenate((start_psf, end_psf)) )

def set_sky_coverage(nside, lst_min, lst_max, radius, res=1/60.):
    '''Calculating the sky coverage and return the healpix no.
    
    Input
    ------
    nside: int
        nside of the HEALpix
    lst_min, lst_max: float
        min and max of the LST range for the sky coverage
    radius: float
        Radius around the center line (in radians)
    res: float
        resolution of advancing disks in the calculation (in deg)
    
    Output
    ------
    hp_idx: array
        idx of the healpix for the sky coverage
    '''
    res = np.radians(res)
    lst_ls = np.arange(lst_min, lst_max, res)
    hp_sky_cover = np.zeros(hp.nside2npix(nside), dtype=np.bool8)
    for lst_t in lst_ls:
        ctr_vec = hp.ang2vec(np.degrees(lst_t),
                             hera_dec,
                             lonlat=True,)
        hp_idx_t = hp.query_disc(nside, ctr_vec, radius, inclusive=True)
        hp_sky_cover[hp_idx_t] = True
    return np.where(hp_sky_cover)[0]