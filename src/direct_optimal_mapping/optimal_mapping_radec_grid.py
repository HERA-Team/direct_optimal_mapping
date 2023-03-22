import numpy as np
import numexpr as ne
import healpy as hp
from astropy.table import Table
from astropy.time import Time
from astropy import constants
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, TETE
from astropy.modeling.functional_models import AiryDisk2D

import copy
import healpy as hp
from pyuvdata import UVData, UVBeam
import multiprocessing

from scipy.interpolate import SmoothSphereBivariateSpline as SSBS
from scipy.interpolate import RectSphereBivariateSpline as RSBS
from scipy.interpolate import RectBivariateSpline as RBS

class SkyPx:
    '''Sky pixel object defining the sky pixels in
    ra/dec regular grid
    
    ...
    Attributes
    ----------
    
    Methods
    -------
    calc_radec_pix(ra_ctr_deg, ra_rng_deg, n_ra,
            dec_ctr_deg, dec_rng_deg, n_dec):
        Calculate the location and solid angle of ra/dec grids
    
    calc_healpix(nside, ra_ctr_deg, dec_ctr_deg, radius_deg)
        Calculate the the location and solid angle of the healpix
            
    '''
    def __init__(self):
        return
    
    def calc_radec_pix(self, ra_ctr_deg, ra_rng_deg, n_ra,
                dec_ctr_deg, dec_rng_deg, n_dec):
        '''Initiating the SkyPix given center location, range, and pixel number
        along RA and Dec
        Ideally, n_ra and n_dec are power of 2 for later FFT
        
        Parameters
        ----------
        ra_ctr_deg: float
            ra central location of the sky patch, in degrees
        ra_rng_deg: float
            ra range of the sky patch, in degrees
        n_ra: int
            number of grids along the RA axis, ideally power of 2
        dec_ctr_deg: float
            dec central location of the sky patch, in degrees
        dec_rng_deg: float
            dec range of the sky patch, in degrees
        n_dec: int
            number of grids along the Dec axis, ideally power of 2
        
        Return
        ------
        px_dic: dictionary
            Containing the ra/dec locations and the solid angle of
            the pixels
        '''
        edge_ra, edge_dec = np.mgrid[ra_ctr_deg - ra_rng_deg/2.:ra_ctr_deg + ra_rng_deg/2.:(n_ra+1)*1j,
                                     dec_ctr_deg - dec_rng_deg/2.:dec_ctr_deg + dec_rng_deg/2.:(n_dec+1)*1j]
        delta_phi = np.radians(edge_ra[1:, :-1]) - np.radians(edge_ra[:-1, :-1])
        edge_dec_shift = edge_dec - dec_ctr_deg #Moving the horizon to the dec center to minize the curve-sky effect
        delta_sin_theta = np.sin(np.radians(edge_dec_shift[:-1, 1:])) - np.sin(np.radians(edge_dec_shift[:-1, :-1]))
        delta_sa = delta_phi*delta_sin_theta
        # pixel center
        ra_res_deg = ra_rng_deg/n_ra
        dec_res_deg = dec_rng_deg/n_dec
        ctr_ra_deg = edge_ra[:-1, :-1] + ra_res_deg/2.
        ctr_dec_deg = edge_dec[:-1, :-1] + dec_res_deg/2.
        ra_deg = ctr_ra_deg.flatten()
        dec_deg = ctr_dec_deg.flatten()        
        px_id = np.array(['%.2f,%.2f'%(ra_deg[i], dec_deg[i]) for i in range(len(ra_deg))])
        px_dic = {'ra_deg': ctr_ra_deg, 'dec_deg': ctr_dec_deg, 'sa_sr': delta_sa, 
                  'px_id':px_id.reshape(ctr_ra_deg.shape)}

        return px_dic
    
    def calc_healpix(self, nside, obstime, site, radius_deg, epoch='J2000'):
        '''Initiating the SkyPix given nside, center location, and radius in healpixels
        
        Parameters
        ----------
        nside: int
            nside of the healpixels
        obstime: Astropy Time Object
            in form of JD
        site: Astropy Earthlocation Object
        radius_deg: float
            radius of the sky patch, in degrees
        epoch: str
            epoch of the coordinates, can be either 'J2000' or 'Current'
        
        Return
        ------
        px_dic: dictionary
            Containing the healpix locations and the solid angle of
            the pixels
        '''
        ra_deg, dec_deg = hp.pix2ang(nside, range(hp.nside2npix(nside)), lonlat=True)
        ra = np.radians(ra_deg)
        dec = np.radians(dec_deg)
        
        aa = AltAz(location=site, obstime=obstime)
        if epoch == 'J2000':
            c = SkyCoord(ra=ra, dec=dec, unit='radian', frame=TETE(obstime=epoch))
        elif epoch == 'Current':
            c = SkyCoord(ra=ra, dec=dec, unit='radian', frame=TETE(obstime=obstime))
        else:
            print('Please provide a proper epoch: either J2000 or Current')
        az = np.radians(c.transform_to(aa).az.value)
        alt = np.radians(c.transform_to(aa).alt.value)
        
        hp_idx_t = np.where(alt > np.radians(90-radius_deg))[0]
        sa_sr = np.array([hp.nside2pixarea(nside)] * len(hp_idx_t))
        px_id = np.array(['%.2f,%.2f'%(ra_deg[i], dec_deg[i]) for i in hp_idx_t])
        px_dic = {'ra_deg': ra_deg[hp_idx_t], 'dec_deg': dec_deg[hp_idx_t], 'sa_sr': sa_sr, 
                  'px_id': px_id, 'hp_idx': hp_idx_t}

        return px_dic       

class OptMapping:
    '''Optimal Mapping Object
    
    '''
    
    def __init__(self, uv, px_dic_outer, px_dic_inner=None, epoch='J2000', feed=None,
                 beam_file=None,
                 beam_folder='/nfs/esc/hera/zhileixu/git_beam/HERA-Beams/NicolasFagnoniBeams'):
        '''Init function for basic setup
         
        Parameters
        ----------
        uv: pyuvdata object
            UVData data in the pyuvdata format, data_array only has the blt dimension
        px_dic_outer: dictionary
            pixel dictionary with ra/dec and solid angle information of the outer sky coverage
        px_dic_inner: dictionary
            Default: None, then same as the pix_dic_outer. Otherwise, it defines a smaller sky
            coverage for the map facet
        epoch: str
            epoch of the map, can be either 'J2000' or 'Current'
        feed: str
            feed type 'dipole' or 'vivaldi'. Default is None, and feed type is determined by
            the observation date
        beam_file: str or None
            beam file address, should be in the pyuvbeam fits format describing the efield
            if None (default), look for the beam files in the beam folder
            if Airy, using an analytical Airy beam
        beam_folder: str
            folder of the simulated primary beam files
            only in action when beam_file is None
            
        Return
        ------
        None        
        
        '''
        self.hera_site = EarthLocation(lat=uv.telescope_location_lat_lon_alt_degrees[0]*u.deg,
                                       lon=uv.telescope_location_lat_lon_alt_degrees[1]*u.deg,
                                       height=uv.telescope_location_lat_lon_alt_degrees[2]*u.m)
        
        self.uv = uv
        self.hera_dec = self.uv.telescope_location_lat_lon_alt[0]
        self.lsts = np.unique(self.uv.lst_array)
        self.times = np.unique(uv.time_array)
        self.equinox = epoch
        if beam_file is None:
            self.beam_folder = beam_folder
            if feed is None:
                if np.mean(self.times) < 2458362: #2018-09-01
                    self.feed_type = 'dipole'
                    self.beam_file = self.beam_folder+'/NF_HERA_Dipole_efield_beam_high-precision.fits'
                else:
                    self.feed_type = 'vivaldi'
                    self.beam_file = self.beam_folder+'/NF_HERA_Vivaldi_efield_beam.fits'
            else:
                self.feed_type = feed            
        elif beam_file == 'Airy':
            self.beam_file = 'Airy'
        else:
            self.beam_file = beam_file
            
        if px_dic_inner is None:
            px_dic_inner = px_dic_outer
            self.idx_inner_in_outer = np.ones(len(px_dic_inner['px_id'].flatten()))
        else:                                          
            self.idx_inner_in_outer = np.array([np.where(px_dic_outer['px_id'].flatten() == px_dic_inner['px_id'].flatten()[i])[0][0] 
                                                for i in range(len(px_dic_inner['px_id'].flatten()))])

        self.ra = np.radians(px_dic_outer['ra_deg']).flatten()
        self.dec = np.radians(px_dic_outer['dec_deg']).flatten()
        self.px_sa = px_dic_outer['sa_sr'].flatten()
        self.npx = len(self.px_sa)
        
        az, alt = self._radec2azalt(self.ra, self.dec,
                                    np.mean(self.times))
        self.az = az
        self.alt = alt
        
        self.frequency = np.squeeze(self.uv.freq_array)
        self.wavelength = constants.c.value/self.frequency
                
        data = np.squeeze(self.uv.data_array, axis=(1, 2, 3))
        flag = np.squeeze(self.uv.flag_array, axis=(1, 2, 3))
        self.data = np.expand_dims(data, axis=1)
        self.flag = np.expand_dims(flag, axis=1)
        self.nvis = len(data)

        return

    def _radec2azalt(self, ra, dec, time):
        '''Convert ra/dec to az/alt at the given obs_time and assuming the site
        as HERA
        
        Parameters
        ----------
        ra: 1d array (float)
            array of the ra coordintes (in radians)
        dec, 1d array (float)
            array of the dec coordintes (in radians)
        time: float
            observation time (in the format of JD)
            
        Output
        ------
        az, alt: 1d array (float)
            arrays containing the converted az, alt values (in radians)
        '''
        obstime = Time(time, format='jd')
        aa = AltAz(location=self.hera_site, obstime=obstime)
        if self.equinox == 'J2000':
            c = SkyCoord(ra=ra, dec=dec, unit='radian', frame=TETE(obstime=self.equinox))
        elif self.equinox == 'Current':
            c = SkyCoord(ra=ra, dec=dec, unit='radian', frame=TETE(obstime=obstime))
        else:
            print('Please provide a proper epoch: either J2000 or Current')
        az = np.radians(c.transform_to(aa).az.value)
        alt = np.radians(c.transform_to(aa).alt.value)
        
        return az, alt
    
    def set_pyuvbeam(self, beam_file):
        '''Set up the pyuvbeam from simulation for interpolation
        
        Parameters
        ----------
        beam_file: str 
            address of the beam file
            
        Return
        ------
        None
        
        Attribute
        ---------
        .pyuvbeam: UVBeam Object
            UVBeam Object for beam interpolation 
        '''

        pyuvbeam = UVBeam()
        pyuvbeam.read_beamfits(beam_file)
        if pyuvbeam.beam_type == 'efield':
            pyuvbeam.efield_to_power()
        pyuvbeam.select(polarizations=self.uv.polarization_array)
        pyuvbeam.peak_normalize()
        pyuvbeam.interpolation_function = 'az_za_simple'
        pyuvbeam.freq_interp_kind = 'cubic'
        
        # attribute assignment
        self.pyuvbeam = pyuvbeam
        return
    
    def airy_beam(self, az, alt, freq, dish_dia=12):
        '''Calculating the circularly symmetric Airy beam
        
        Parameters
        ----------
        az, alt: azimuth and altitude of the sky position, in radians
        freq: frequency, in Hz
        dish_dia: dish diameter, in meters
            Default value is 12m for HERA dishes
        
        Return
        ------
        Airy beam values
        '''
        airy = AiryDisk2D()
        wv = constants.c.value/freq
        airy_radius = 1.22*wv/dish_dia
        peak_amp = 1
        # airy.evaluate(x, y, amplitude, x_0, y_0, radius)
        return airy.evaluate(np.cos(alt)*np.cos(az),
                             np.cos(alt)*np.sin(az), 
                             peak_amp, 0, 0, airy_radius)
    
    def set_a_mat(self, uvw_sign=1, apply_beam=True):
        '''Calculating A matrix, covering the range defined by the px_dic object
        
        Parameters
        ----------
        uvw_sign: 1 or -1
            uvw sign for the baseline calculation
        apply_beam: boolean
            Whether apply beam to the a matrix elements, default:true
        
        Attribute
        ---------
        .a_mat: 2d matrix (complex128)
            a_matrix (Nvis X Npsf) from the given observation
        .beam_mat: 2d matrix (float64)
            a_matrix with only the beam term considered (Nvis X Npsf)
        '''
        self.phase_mat = np.zeros((self.nvis, self.npx), dtype='float64')
        self.beam_mat = np.zeros(self.phase_mat.shape, dtype='float64')
        self.sa_mat = np.zeros(self.phase_mat.shape, dtype='float64')
        if self.beam_file != 'Airy':
            self.set_pyuvbeam(beam_file=self.beam_file)
        freq_array = np.array([self.frequency,])
        for time_t in np.unique(self.uv.time_array):
            az_t, alt_t = self._radec2azalt(self.ra, self.dec, time_t)
            lmn_t = np.array([np.cos(alt_t)*np.sin(az_t), 
                              np.cos(alt_t)*np.cos(az_t), 
                              np.sin(alt_t)])
            
            if self.beam_file == 'Airy':
                print('Airy beam.')
                beam_map_t = self.airy_beam(az_t, alt_t, freq_array[0])
            else:
                pyuvbeam_interp,_ = self.pyuvbeam.interp(az_array=np.mod(np.pi/2. - az_t, 2*np.pi), 
                                                         za_array=np.pi/2. - alt_t, 
                                                         az_za_grid=False, freq_array= freq_array,
                                                         reuse_spline=True, check_azza_domain=False)
                beam_map_t = pyuvbeam_interp[0, 0, 0, 0].real
            
            idx_time = np.where(self.uv.time_array == time_t)[0]
            self.phase_mat[idx_time] = uvw_sign*2*np.pi/self.wavelength*(self.uv.uvw_array[idx_time]@lmn_t)
            self.beam_mat[idx_time] = np.tile(beam_map_t, idx_time.size).reshape(idx_time.size, -1)
            self.sa_mat[idx_time] = np.tile(self.px_sa, idx_time.size).reshape(idx_time.size, -1)
            
        self.a_mat = ne.evaluate('exp(A * 1j)', global_dict={'A':self.phase_mat})
        
        if apply_beam:
            self.beam_mat[self.flag.flatten()] = 0
            self.a_mat = np.multiply(self.a_mat, self.beam_mat)
        
        self.a_mat = np.multiply(self.a_mat, self.sa_mat)

        return 
    
    def set_inv_noise_mat(self, uvn, matrix=True, norm=False):
        '''Calculating the inverse noise matrix with auto-correlations
        
        Parameters
        ----------
        uvn: pyuvdata
            pyuvdata object with estimated noise information
        matrix: boolean
            'True' means the return matrix is a matrix; 'False' saves only
            the diagonal elements of the matrix, igonoring covariance
        norm: boolean
            whether normalize the sum of N^-1 diagonal terms
        '''
        if matrix:
            inv_noise_mat = np.diag(np.squeeze(uvn.data_array, axis=(1, 2, 3)).real**(-2))
            self.norm_factor = np.sum(np.diag(inv_noise_mat))
        else:
            inv_noise_mat = np.squeeze(uvn.data_array, axis=(1, 2, 3)).real**(-2)          
            self.norm_factor = np.sum(inv_noise_mat)
        if norm:
            inv_noise_mat = inv_noise_mat/self.norm_factor
        self.inv_noise_mat = inv_noise_mat
       
        return inv_noise_mat
    
    def set_p_mat(self):
        '''Calculating P matrix
        
        Parameters
        ----------
        None
        
        Return
        ------
        None

        Attribute
        ---------
        .p_mat: 2d matrix (complex128)
            p_matrix from the given observation as an attribute
        .p_diag: 1d array (complex128)
            normalization array for the map within the facet
        .p_square: 2d matrix (complex128)
            square p matrix containing only the facet pixels on 
            both dimensions
        '''
        #p_matrix set up
        if not hasattr(self, 'a_mat'):
            raise AttributeError('A matrix is not set up.')

        p_mat1 = np.conjugate(self.a_mat[:, self.idx_inner_in_outer].T)
        p_mat2 = np.diag(self.inv_noise_mat)[:, None]*self.a_mat 
        #Equivalent to inv_noise_mat@a_mat, assuming diagonal noise matrix

        self.p_mat = np.real(np.matmul(p_mat1, p_mat2))
        del p_mat1, p_mat2
        
        return
