import healpy as hp
import numpy as np

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

hera_dec = -30.721526120689507
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
