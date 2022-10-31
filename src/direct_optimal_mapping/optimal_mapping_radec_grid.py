import numpy as np
import numexpr as ne
import healpy as hp
from astropy.table import Table
from astropy.time import Time
from astropy import constants
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, TETE
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
    calc_px(ra_ctr_deg, ra_rng_deg, n_ra,
            dec_ctr_deg, dec_rng_deg, n_dec):
        Calculate the location and solid angle of ra/dec grids
    
    '''
    def __init__(self):
        return
    
    def calc_px(self, ra_ctr_deg, ra_rng_deg, n_ra,
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
            
        Attritute
        ---------
        self.px_dic: as px_dic
        
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
        
        self.px_dic = {'ra_deg': ctr_ra_deg, 'dec_deg': ctr_dec_deg, 'sa_sr': delta_sa}

        return self.px_dic

class OptMapping:
    '''Optimal Mapping Object
    
    '''
    
    def __init__(self, uv, px_dic, epoch='J2000', feed=None, 
                 beam_folder='/nfs/esc/hera/zhileixu/git_beam/HERA-Beams/NicolasFagnoniBeams'):
        '''Init function for basic setup
         
        Parameters
        ----------
        uv: pyuvdata object
            UVData data in the pyuvdata format, data_array only has the blt dimension
        px_dic: dictionary
            pixel dictionary with ra/dec and solid angle information
        epoch: str
            epoch of the map, can be either 'J2000' or 'Current'
        feed: str
            feed type 'dipole' or 'vivaldi'. Default is None, and feed type is determined by
            the observation date
        beam_folder: str
            folder of the simulated primary beam files

        Return
        ------
        None        
        
        '''
        self.hera_site = EarthLocation(lat=uv.telescope_location_lat_lon_alt_degrees[0]*u.deg,
                                       lon=uv.telescope_location_lat_lon_alt_degrees[1]*u.deg,
                                       height=uv.telescope_location_lat_lon_alt_degrees[2]*u.m)
        
        self.uv = uv
#         self.nside = nside
#         self.npix = hp.nside2npix(nside)
        self.hera_dec = self.uv.telescope_location_lat_lon_alt[0]
        self.lsts = np.unique(self.uv.lst_array)
        self.times = np.unique(uv.time_array)
        self.equinox = epoch
        self.beam_folder = beam_folder
        if feed is None:
            if np.mean(self.times) < 2458362: #2018-09-01
                self.feed_type = 'dipole'
            else:
                self.feed_type = 'vivaldi'
        else:
            self.feed_type = feed
        #print('RA/DEC in the epoch of %s, with %s beam used.'%(self.equinox, self.feed_type))

#         theta, phi = hp.pix2ang(nside, range(self.npix))
        self.ra = np.radians(px_dic['ra_deg']).flatten()
        self.dec = np.radians(px_dic['dec_deg']).flatten()
        self.px_sa = px_dic['sa_sr'].flatten()
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
        obs_time = Time(time, format='jd')
        aa = AltAz(location=self.hera_site, obstime=obs_time)
        if self.equinox == 'J2000':
            c = SkyCoord(ra=ra, dec=dec, unit='radian', frame=TETE(obstime=self.equinox))
            #c = SkyCoord(ra=ra, dec=dec, unit='radian', frame='icrs')
            #print('ICRS')
        elif self.equinox == 'Current':
            c = SkyCoord(ra=ra, dec=dec, unit='radian', frame=TETE(obstime=obs_time))
        else:
            print('Please provide a proper epoch: either J2000 or Current')
        az = np.radians(c.transform_to(aa).az.value)
        alt = np.radians(c.transform_to(aa).alt.value)
        
        return az, alt
    
    def set_pyuvbeam(self, beam_model):
        '''Set up the pyuvbeam from simulation for interpolation
        
        Parameters
        ----------
        beam_model: str ('vivaldi' or 'dipole')
            beam model used for interpolation
            
        Return
        ------
        None
        
        Attribute
        ---------
        .pyuvbeam: UVBeam Object
            UVBeam Object for beam interpolation 
        '''
        # loading the beamfits file
        if beam_model == 'vivaldi':
            beamfits_file = self.beam_folder+\
                            '/efield_farfield_Vivaldi_pos_0.0_0.0_0.0_0.0_0.0_160_180MHz_high_precision_0.125MHz_simplified_model.beamfits'
            print('Vivaldi beam simulation file is not set up yet.')
        elif beam_model == 'dipole':
            beamfits_file = self.beam_folder+'/NF_HERA_Dipole_efield_beam_high-precision.fits'
            #beamfits_file = '/nfs/esc/hera/zhileixu/git_beam/cst_beam_files/fagnoni_high_precision_dipole/H19/'+\
            #                'E-farfield-100ohm-50-250MHz-high-acc-ind-H19-port21/efield_dipole_H19-port21_high-precision_peak-norm.fits'     
        else:
            print('Please provide correct beam model (either vivaldi or dipole)')
        #print('Beam file:', beamfits_file)
        pyuvbeam = UVBeam()
        pyuvbeam.read_beamfits(beamfits_file)        
        pyuvbeam.efield_to_power()
        pyuvbeam.select(polarizations=self.uv.polarization_array)
        #print(pyuvbeam.polarization_array)
        pyuvbeam.peak_normalize()
        pyuvbeam.interpolation_function = 'az_za_simple'
        pyuvbeam.freq_interp_kind = 'cubic'
        
        # attribute assignment
        self.pyuvbeam = pyuvbeam
        return
    
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
        self.set_pyuvbeam(beam_model=self.feed_type)
#         freq_array = np.array([self.frequency,])
        freq_array = np.array([169e6,])
        for time_t in np.unique(self.uv.time_array):
            az_t, alt_t = self._radec2azalt(self.ra, self.dec, time_t)
            lmn_t = np.array([np.cos(alt_t)*np.sin(az_t), 
                              np.cos(alt_t)*np.cos(az_t), 
                              np.sin(alt_t)])
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

        p_mat1 = np.conjugate(self.a_mat.T)
        p_mat2 = np.diag(self.inv_noise_mat)[:, None]*self.a_mat 
        #Equivalent to inv_noise_mat@a_mat, assuming diagonal noise matrix

        self.p_mat = np.real(np.matmul(p_mat1, p_mat2))
        del p_mat1, p_mat2
        
        return
