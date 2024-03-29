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

from . import pixel_selection

class OptMapping:
    '''Optimal Mapping Object
    '''
    
    def __init__(self, uv, nside, epoch='J2000', feed=None, 
                 beam_file = None,
                 beam_folder='/nfs/esc/hera/zhileixu/git_beam/HERA-Beams/NicolasFagnoniBeams'):
        '''Init function for basic setup
         
        Input
        ------
        uv: pyuvdata object
            UVData data in the pyuvdata format, data_array only has the blt dimension
        nside: int
            nside of the healpix map
        epoch: str
            epoch of the map, can be either 'J2000' or 'Current'
        feed: str
            feed type 'dipole' or 'vivaldi'. Default is None, and feed type is determined by
            the observation date
        beam_file: str or None
            beam file address, should be in the pyuvbeam fits format describing the efield
            if None (default), look for the beam files in the beam folder
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
        self.nside = nside
        self.npix = hp.nside2npix(nside)
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
        else:
            self.beam_file = beam_file
        
        #print('RA/DEC in the epoch of %s, with %s beam used.'%(self.equinox, self.feed_type))

        theta, phi = hp.pix2ang(nside, range(self.npix))
        self.ra = phi
        self.dec = np.pi/2. - theta
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
        
        Input:
        ------
        ra: 1d array (float)
            array of the ra coordintes (in radians)
        dec, 1d array (float)
            array of the dec coordintes (in radians)
        time: float
            observation time (in the format of JD)
            
        Output:
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
        
    def set_k_psf(self, radius_deg, calc_k=False):
        '''Function to set up the K_psf matrix. K_psf selects
        healpix from the entire sky to the regions within a 
        certain radius away from the phase center
        
        Input:
        ------
        radius_deg: float (in degrees)
            radius to be included in the K_psf matrix
        calc_k: boolean
            whether calculating K_psf
            
        Output:
        ------
        k_psf: 2d array (boolean) (if calc_k=True)
            Npsf X Npix array 
            
        Attributes:
        ------
        .k_psf_in: 1d array (int)
            healpix map indices within the PSF
        .k_psf_out: 1d array (int)
            healpix map indices outside of the PSF
        .k_psf: 2d array (bool), if calc_k=True
            matrix turning the full map into psf-included map
        '''
        psf_radius = np.radians(radius_deg)
        self.idx_psf_out = np.where((np.pi/2 - self.alt) > psf_radius)[0]
        self.idx_psf_in = np.where((np.pi/2 - self.alt) < psf_radius)[0]
        if calc_k:
            k_full = np.diag(np.ones(self.npix, dtype=bool))
            k_psf = np.delete(k_full, idx_psf_out, axis=0).T
            del k_full
            self.k_psf = k_psf
            return k_psf
        else:
            return

    def set_psf_by_idx(self, idx_psf_in=None, calc_k=False):
        '''Set up the K_psf matrix by passing idices. 
        
        Input:
        ------
        idx_psf_idx: array-like int
            Healpix idices of the psf region for the map. Default is the union of horizon during
            the entire observation according to uv
        calc_k: boolean
            whether calculating K_psf
            
        Output:
        ------
        k_psf: 2d array (boolean) (if calc_k=True)
            Npsf X Npix array 
            
        Attributes:
        ------
        .k_psf_in: 1d array (int)
            healpix map indices within the PSF
        .k_psf_out: 1d array (int)
            healpix map indices outside of the PSF
        .k_psf: 2d array (bool), if calc_k=True
            matrix turning the full map into psf-included map
        '''
        if idx_psf_in is None:
            self.idx_psf_in = pixel_selection.set_psf_idx(nside, self.lsts.min(), self.lsts.max(), radius=90)
        else:
            assert idx_psf_in.max() <= self.npix, "PSF indices out of range."
            self.idx_psf_in = idx_psf_in
        self.idx_psf_out = np.arange(self.npix)[~np.in1d(np.arange(self.npix), self.idx_psf_in)]
        if calc_k:
            k_full = np.diag(np.ones(self.npix, dtype=bool))
            k_psf = np.delete(k_full, idx_psf_out, axis=0).T
            del k_full
            self.k_psf = k_psf
            return k_psf
        else:
            return
    
    def set_pyuvbeam(self, beam_file):
        '''Set up the pyuvbeam from simulation for interpolation
        Args
        ------
        beam_file: str 
            address of the beam file
            
        Output:
        ------
        None
        
        Attribute:
        .pyuvbeam: UVBeam Object
            UVBeam Object for beam interpolation 
        '''
        
        pyuvbeam = UVBeam()
        pyuvbeam.read_beamfits(beam_file)        
        if pyuvbeam.beam_type == 'efield':
            pyuvbeam.efield_to_power()
        pyuvbeam.select(polarizations=self.uv.polarization_array)
        #print(pyuvbeam.polarization_array)
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
    
    def set_a_mat(self, uvw_sign=1, apply_beam=True, beam_model='cst'):
        '''Calculating A matrix, covering the range defined by K_psf
        
        Input:
        ------
        uvw_sign: 1 or -1
            uvw sign for the baseline calculation
        apply_beam: boolean
            Whether apply beam to the a matrix elements, default:true
        beam_model: str
            string of the beam model, can be 'cst' (default) or 'airy'
        
        Attribute:
        ------
        ..a_mat: 2d matrix (complex128)
            a_matrix (Nvis X Npsf) from the given observation
        .beam_mat: 2d matrix (float64)
            a_matrix with only the beam term considered (Nvis X Npsf)
        '''
        self.a_mat = np.zeros((len(self.data), len(self.idx_psf_in)), dtype='float64')
        self.beam_mat = np.zeros(self.a_mat.shape, dtype='float64')
        self.set_pyuvbeam(beam_file=self.beam_file)
        freq_array = np.array([self.frequency,])
        for time_t in np.unique(self.uv.time_array):
            az_t, alt_t = self._radec2azalt(self.ra[self.idx_psf_in],
                                            self.dec[self.idx_psf_in],
                                            time_t)
            lmn_t = np.array([np.cos(alt_t)*np.sin(az_t), 
                              np.cos(alt_t)*np.cos(az_t), 
                              np.sin(alt_t)])
            if beam_model == 'cst':
                pyuvbeam_interp,_ = self.pyuvbeam.interp(az_array=np.mod(np.pi/2. - az_t, 2*np.pi), 
                                                         za_array=np.pi/2. - alt_t, 
                                                         az_za_grid=False, freq_array= freq_array,
                                                         reuse_spline=True, check_azza_domain=False)
                beam_map_t = pyuvbeam_interp[0, 0, 0, 0].real
            elif beam_model == 'airy':
                print('Airy beam.')
                beam_map_t = self.airy_beam(az_t, alt_t, freq_array[0])
            else:
                print('Please provide correct beam_model.')
            idx_time = np.where(self.uv.time_array == time_t)[0]
            self.a_mat[idx_time] = uvw_sign*2*np.pi/self.wavelength*(self.uv.uvw_array[idx_time]@lmn_t)
            self.beam_mat[idx_time] = np.tile(beam_map_t, idx_time.size).reshape(idx_time.size, -1)
            
        self.a_mat = ne.evaluate('exp(A * 1j)', global_dict={'A':self.a_mat})
        if apply_beam:
            self.beam_mat[self.flag.flatten()] = 0
            self.a_mat = np.multiply(self.a_mat, self.beam_mat)

        return 
    
    def beam_interp_onecore(self, time, pix):
        '''Calculating the phase for the pixels within PSF at a given time
        Input
        ------
        time: float
            JD of the observation time
        pix: str
            'hp' or 'hp+ps' meaning 'healpix' or 'healpix + point sources'
        
        Output
        ------
        beam_dic: dictionary
            with the 'time' variable as the key and the interpolated beam 
            as the content
        '''
        
        if pix == 'hp':
            ra_arr = self.ra[self.idx_psf_in]
            dec_arr = self.dec[self.idx_psf_in]
        elif pix == 'hp+ps':
            ra_arr = np.concatenate((self.ra[self.idx_psf_in], self.ra_ps))
            dec_arr = np.concatenate((self.dec[self.idx_psf_in], self.dec_ps))
        else:
            print('Please provide a correct pix kind: hp or hp+ps.')
        az_t, alt_t = self._radec2azalt(ra_arr, dec_arr, time)
        lmn_t = np.array([np.cos(alt_t)*np.sin(az_t), 
                          np.cos(alt_t)*np.cos(az_t), 
                          np.sin(alt_t)])
        pyuvbeam_interp, _ = self.pyuvbeam.interp(az_array=np.mod(np.pi/2. - az_t, 2*np.pi), 
                                                  za_array=np.pi/2. - alt_t, 
                                                  az_za_grid=False, freq_array= np.array([self.frequency,]),
                                                  reuse_spline=True)
        beam_map_t = pyuvbeam_interp[0, 0, 0, 0].real
        return {time: beam_map_t}
    
    def set_beam_interp(self, pix, ncores=10):
        '''Run the beam interpolation in parallel and store the result in a dictionary
        Input
        ------
        pix: str
            'hp', or 'hp+ps'
        ncores: int
            Number of cores for the parallelization
        
        Output
        ------
        None
        '''
        print(pix)
        self.set_pyuvbeam(beam_file=self.beam_file)
        pool = multiprocessing.Pool(processes=ncores)
        args = []
        for time_t in np.unique(self.uv.time_array):
            args.append([time_t, pix])
        results = pool.starmap(self.beam_interp_onecore, args)
        pool.close()
        pool.join()
        self.beam_dic = {}
        for dic_t in results:
            self.beam_dic.update(dic_t)
        return 
        
    def set_a_mat_ps(self, ps_radec, uvw_sign=1, apply_beam=True):
        '''Calculating A matrix, covering the range defined by K_psf
        + the point sources given in the ps_radec arguement
        
        Input:
        ------
        ps_radec: 2d array
            with shape as n_source X 2, it saves the ra,dec of all 
            the point sources (in radians)
        uvw_sign: 1 or -1
            uvw sign for the baseline calculation
        apply_beam: boolean
            Whether apply beam to the a matrix elements, default:true
        
        Attribute:
        ------
        .a_mat_ps: 2d matrix (complex128)
            a_matrix (Nvis X (Npsf+Nps)) from the given observation
        .a_mat: 2d matrix (complex128)
            a_matrix (Nvis X Npsf) from the given observation
        .beam_mat: 2d matrix (float64)
            a_matrix_ps with only the beam term considered (Nvis X (Npsf+Nps))
        '''
        self.a_mat_ps = np.zeros((len(self.data), len(self.idx_psf_in)+ps_radec.shape[0]), dtype='float64')
        self.beam_mat = np.zeros(self.a_mat_ps.shape, dtype='float64')
        self.set_pyuvbeam(beam_file=self.beam_file)
        freq_array = np.array([self.frequency,])
        self.ra_ps = ps_radec[:, 0]
        self.dec_ps = ps_radec[:, 1]
        for time_t in np.unique(self.uv.time_array):
            az_t, alt_t = self._radec2azalt(np.concatenate((self.ra[self.idx_psf_in], self.ra_ps)),
                                            np.concatenate((self.dec[self.idx_psf_in], self.dec_ps)),
                                            time_t)
            lmn_t = np.array([np.cos(alt_t)*np.sin(az_t), 
                              np.cos(alt_t)*np.cos(az_t), 
                              np.sin(alt_t)])
            pyuvbeam_interp,_ = self.pyuvbeam.interp(az_array=np.mod(np.pi/2. - az_t, 2*np.pi), 
                                                     za_array=np.pi/2. - alt_t, 
                                                     az_za_grid=False, freq_array= freq_array,
                                                     reuse_spline=True, check_azza_domain=False)
            print('check_azza_domain=False.')
            beam_map_t = pyuvbeam_interp[0, 0, 0, 0].real
            beam_map_t = pyuvbeam_interp[0, 0, 0, 0].real
            idx_time = np.where(self.uv.time_array == time_t)[0]
            self.a_mat_ps[idx_time] = uvw_sign*2*np.pi/self.wavelength*(self.uv.uvw_array[idx_time]@lmn_t)
            self.beam_mat[idx_time] = np.tile(beam_map_t, idx_time.size).reshape(idx_time.size, -1)

        self.a_mat_ps = ne.evaluate('exp(A * 1j)', global_dict={'A':self.a_mat_ps})
        if apply_beam:
            self.beam_mat[self.flag.flatten()] = 0
            self.a_mat_ps = np.multiply(self.a_mat_ps, self.beam_mat)
        self.a_mat = self.a_mat_ps[:, :len(self.idx_psf_in)]
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
    
    def set_p_mat(self, facet_radius_deg=7, facet_idx=None):
        '''Calculating P matrix, covering the range defined by K_psf,
        projectin to the range defined by K_facet
        
        Input:
        ------
        facet_radius_deg: Size of the circular facet. Facet will be 
            located around the zenith at the mean integration time
            of the given pyuvdata object. Default radius is 7 deg.
        facet_idx: User specified facet index. Can be obtained using
            the pixel_selection module.
        
        Output:
        ------
        None

        Attribute:
        ------
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

        if facet_idx is None:
            self.set_k_facet(radius_deg=facet_radius_deg, calc_k=False)
        else:
            self.idx_facet_in = facet_idx

        _idx = np.searchsorted(self.idx_psf_in, self.idx_facet_in) #Equivalent to Finding K_facet
        p_mat1 = np.conjugate(self.a_mat.T)[_idx] #Equivalent to K_facet@a_mat.H
        p_mat2 = np.diag(self.inv_noise_mat)[:, None]*self.a_mat 
        #Equivalent to inv_noise_mat@a_mat, assuming diagonal noise matrix

        self.p_mat = np.real(np.matmul(p_mat1, p_mat2))
        del p_mat1, p_mat2

        self.p_square = self.p_mat[:, _idx]
        self.p_diag = np.diag(self.p_square)
        
        return
    
    def set_p_mat_ps(self, facet_radius_deg=7, facet_idx=None):
        '''Calculating P matrix with stand-alone point sources, 
        covering the range defined by K_psf + point source pixels,
        projectin to the range defined by K_facet
        
        Input:
        ------
        facet_radius_deg: Size of the circular facet. Facet will be 
            located around the zenith at the mean integration time
            of the given pyuvdata object. Default radius is 7 deg.
        facet_idx: User specified facet index. Can be obtained using
            the pixel_selection module.
        
        Output:
        ------
        None
        
        Attribute:
        ------
        .p_mat_ps: 2d matrix (complex128)
            p_matrix_ps from the given observation as an attribute
        .p_diag_ps: 1d array (complex128)
            normalization array for the map within the facet
        .p_square_ps: 2d matrix (complex128)
            square p matrix containing only the facet pixels on 
            both dimensions
        '''
        if not hasattr(self, 'a_mat_ps'):
            raise AttributeError('A matrix with point sources pixel is not set up.')

        if facet_idx is None:
            self.set_k_facet(radius_deg=facet_radius_deg, calc_k=False)
        else:
            self.idx_facet_in = facet_idx

        _idx = np.searchsorted(self.idx_psf_in, self.idx_facet_in) #Equivalent to Finding K_facet
        p_mat1 = np.conjugate(self.a_mat.T)[_idx] 
        #Equivalent to K_facet@a_mat.H. Note that discard the point source part
        p_mat2 = np.diag(self.inv_noise_mat)[:, None]*self.a_mat_ps
        #Equivalent to inv_noise_mat@a_mat, assuming diagonal noise matrix

        self.p_mat_ps = np.real(np.matmul(p_mat1, p_mat2))
        del p_mat1, p_mat2

        self.p_square_ps = self.p_mat_ps[:, _idx]
        self.p_diag_ps = np.diag(self.p_square_ps)
        
        return
        
    def set_k_facet(self, radius_deg, calc_k=False):
        '''Calculating the K_facet matrix
        
        Input:
        ------
        radius: float (in degrees)
            radius to be included in the K_facet matrix
            
        Output:
        ------
        k_facet: 2d array (boolean)
            Nfacet X Npsf array 
            
        Attributes:
        ------
        .k_facet_in: 1d array (int)
            healpix map indices within the facet
        .k_facet_out: 1d array (int)
            healpix map indices outside of the facet
        .k_facet: 2d array (bool), if calc_k=True
            matrix turning the full map into facet-included map
        '''
        facet_radius = np.radians(radius_deg)
        self.idx_facet_in = np.where((np.pi/2. - self.alt) < facet_radius)[0]
        self.idx_facet_out = np.where((np.pi/2. - self.alt) > facet_radius)[0]       
        
        if calc_k:
            k_full = np.diag(np.ones(len(self.idx_psf_in), dtype=bool))
            idx_facet_out_psf = np.where((np.pi/2. - self.alt[self.idx_psf_in]) > facet_radius)[0]
            k_facet = np.delete(k_full, idx_facet_out_psf, axis=0)
            del k_full
            self.k_facet = k_facet
            return k_facet
        else:
            return    
