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

class OptMapping:
    '''Optimal Mapping Object
    '''
    
    def __init__(self, uv, nside, epoch='J2000', feed=None):
        '''Init function for basic setup
         
        Input
        ------
        uv: pyuvdata object
            UVData data in the pyuvdata format, data_array only has the blt dimension
        nside: integar
            nside of the healpix map
        epoch: str
            epoch of the map, can be either 'J2000' or 'Current'
        feed: str
            feed type 'dipole' or 'vivaldi'. Default is None, and feed type is determined by
            the observation date
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
        if feed is None:
            if np.mean(self.times) < 2458362: #2018-09-01
                self.feed_type = 'dipole'
            else:
                self.feed_type = 'vivaldi'
        else:
            self.feed_type = feed

        print('RA/DEC in the epoch of %s, with %s beam used.'%(self.equinox, self.feed_type))
        
        theta, phi = hp.pix2ang(nside, range(self.npix))
        self.ra = phi
        self.dec = np.pi/2. - theta
        az, alt = self._radec2azalt(self.ra, self.dec,
                                    np.mean(self.times))
        self.az = az
        self.alt = alt
        
        self.frequency = np.squeeze(self.uv.freq_array)
        self.wavelength = constants.c.value/self.frequency
                
        data = np.squeeze(self.uv.data_array)
        flag = np.squeeze(self.uv.flag_array)
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
    
    def set_pyuvbeam(self, beam_model):
        '''Set up the pyuvbeam from simulation for interpolation
        Args
        ------
        beam_model: str ('vivaldi' or 'dipole')
            beam model used for interpolation
            
        Output:
        ------
        None
        
        Attribute:
        .pyuvbeam: UVBeam Object
            UVBeam Object for beam interpolation 
        '''
        # loading the beamfits file
        if beam_model == 'vivaldi':
            beamfits_file = '/nfs/esc/hera/HERA_beams/high_precision_runs/outputs/'+\
            'cst_vivaldi_time_solver_simplified_master_Apr2021/uvbeam/'+\
            'efield_farfield_Vivaldi_pos_0.0_0.0_0.0_0.0_0.0_160_180MHz_high_precision_0.125MHz_simplified_model.beamfits'
            #print('Vivaldi beam simulation file is not set up yet.')
        elif beam_model == 'dipole':
            beamfits_file = '/nfs/ger/home/zhileixu/data/git_beam/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Dipole_efield_beam_high-precision.fits'
        else:
            print('Please provide correct beam model (either vivaldi or dipole)')
        pyuvbeam = UVBeam()
        pyuvbeam.read_beamfits(beamfits_file)        
        pyuvbeam.efield_to_power()
        #pyuvbeam.select(polarizations=self.uv.polarization_array)
        pyuvbeam.select(polarizations=[-6,])
        #print(pyuvbeam.polarization_array)
        pyuvbeam.peak_normalize()
        pyuvbeam.interpolation_function = 'az_za_simple'
        pyuvbeam.freq_interp_kind = 'cubic'
        
        # attribute assignment
        self.pyuvbeam = pyuvbeam
        return
    
    def pyuvbeam_efield_to_power(self, efield_data, basis_vector_array,
                                 calc_cross_pols=True):
    
        Nfeeds = efield_data.shape[0]
        Nfreqs = efield_data.shape[3]
        Nsources = efield_data.shape[4]

        feed_pol_order = [(0, 0)]
        if Nfeeds > 1:
            feed_pol_order.append((1, 1))

        if calc_cross_pols:
            Npols = Nfeeds ** 2
            if Nfeeds > 1:
                feed_pol_order.extend([(0, 1), (1, 0)])
        else:
            Npols = Nfeeds


        power_data = np.zeros((1, 1, Npols, Nfreqs, Nsources), dtype=np.complex128)


        for pol_i, pair in enumerate(feed_pol_order):
            for comp_i in range(2):
                power_data[0, :, pol_i] += (
                    (
                        efield_data[0, :, pair[0]]
                        * np.conj(efield_data[0, :, pair[1]])
                    )
                    * basis_vector_array[0, comp_i] ** 2
                    + (
                        efield_data[1, :, pair[0]]
                        * np.conj(efield_data[1, :, pair[1]])
                    )
                    * basis_vector_array[1, comp_i] ** 2
                    + (
                        efield_data[0, :, pair[0]]
                        * np.conj(efield_data[1, :, pair[1]])
                        + efield_data[1, :, pair[0]]
                        * np.conj(efield_data[0, :, pair[1]])
                    )
                    * (
                        basis_vector_array[0, comp_i]
                        * basis_vector_array[1, comp_i]
                    )
                )

        power_data = np.real_if_close(power_data, tol=10)

        return power_data
        
    def set_beam_model(self, beam_model, interp_method='grid'):
        '''Beam interpolation model set up with RectSphereBivariantSpline
        beam power is used as sqrt(col4**2 + col6**2)
        
        Input:
        ------
        beam_model: str ('vivaldi' or 'dipole')
            beam model used for interpolation
        interp_method: str ('grid' or 'sphere')
            Method used for interpolating the beam
            'grid' -> RectBivariateSpline
            'sphere' -> RectSphereBivariateSpline
        
        Output:
        ------
        None
        
        Attribute:
        .beam_model: function
            interpolation function for the beam
        '''
        # loading the beam file
        if beam_model == 'vivaldi':
            beam_file_folder = '/nfs/eor-14/d1/hera/beams/Vivaldi_1.8m-detailed_mecha_design-E-field-100ohm_load-Pol_X'
        elif beam_model == 'dipole':
            beam_file_folder = '/nfs/ger/proj/hera/beams/dipole_beams_Efield/HERA 4.9m - E-field'
        else:
            print('Please provide correct beam model (either vivaldi or dipole)')
        ifreq = int(np.round(self.frequency/1e6))
        beam_file = beam_file_folder+'/farfield (f=%d) [1].txt'%ifreq
        beam_table = Table.read(beam_file, format='ascii', data_start=2)
        #print(beam_model, 'is selected with', interp_method, 'interpolation method.')
        beam_theta = np.radians(np.unique(beam_table['col1']))
        beam_phi = np.radians(np.unique(beam_table['col2']))
        power = beam_table['col4']**2 + beam_table['col6']**2
        beam_data = power.reshape(len(beam_phi), len(beam_theta)).T
        beam_data = beam_data/beam_data.max()
        if interp_method == 'sphere':
            epsilon = 1e-5
            beam_theta[0] += epsilon
            beam_theta[-1] -= epsilon
            beam_model = RSBS(beam_theta, beam_phi, beam_data)
        elif interp_method == 'grid':
            beam_model = RBS(beam_theta, beam_phi, beam_data)
        else:
            print('Please provide a proper interpolation method, either sphere or grid.')
        # Attribute assignment
        self.beam_model = beam_model
        
        return

    def set_a_mat(self, uvw_sign=1, apply_beam=True):
        '''Calculating A matrix, covering the range defined by K_psf
        
        Input:
        ------
        uvw_sign: 1 or -1
            uvw sign for the baseline calculation
        apply_beam: boolean
            Whether apply beam to the a matrix elements, default:true
        
        Output:
        ------
        a_mat: 2d matrix (complex64)
            a_matrix (Nvis X Npsf) from the given observation

        Attribute:
        ------
        .a_mat: 2d matrix (complex64)
            a_matrix added in the attribute
        '''
        a_mat = np.zeros((len(self.data), len(self.idx_psf_in)), dtype='float64')
        beam_mat = np.zeros(a_mat.shape, dtype='float64')
        #self.set_beam_model(beam_model=self.feed_type)
        self.set_pyuvbeam(beam_model=self.feed_type)
        #print('Pyuvdata readin.')
        freq_array = np.array([self.frequency,])
        #self.set_beam_interp('hp')
        for time_t in np.unique(self.uv.time_array):
            az_t, alt_t = self._radec2azalt(self.ra[self.idx_psf_in],
                                            self.dec[self.idx_psf_in],
                                            time_t)
            lmn_t = np.array([np.cos(alt_t)*np.sin(az_t), 
                              np.cos(alt_t)*np.cos(az_t), 
                              np.sin(alt_t)])
            #beam_map_t = self.beam_model(np.pi/2. - alt_t, az_t, grid=False)
            pyuvbeam_interp,_ = self.pyuvbeam.interp(az_array=az_t, za_array=np.pi/2. - alt_t, 
                                                     az_za_grid=False, freq_array= freq_array,
                                                     reuse_spline=True) 
            #print('efield interpolation...')
            #pyuvbeam_interp_e, vectors = self.pyuvbeam.interp(az_array=az_t, za_array=np.pi/2. - alt_t, 
            #                                                  az_za_grid=False, freq_array= freq_array,
            #                                                  reuse_spline=True)
            #pyuvbeam_interp = self.pyuvbeam_efield_to_power(pyuvbeam_interp_e, vectors)
            #ipol = 0
            #print(ipol)
            beam_map_t = pyuvbeam_interp[0, 0, 0, 0].real
            #beam_map_t = self.beam_dic[time_t]
            idx_time = np.where(self.uv.time_array == time_t)[0]
            for i in range(len(idx_time)):
                irow = idx_time[i]
                a_mat[irow] = uvw_sign*2*np.pi/self.wavelength*np.matmul(np.matrix(self.uv.uvw_array[irow].astype(np.float32)),
                                                                         np.matrix(lmn_t.astype(np.float32)))
                if self.flag[irow] == False:
                    beam_mat[irow] = beam_map_t.astype(np.float32)
                elif self.flag[irow] == True:
                    beam_mat[irow] = np.zeros(beam_mat.shape[1])
                    print('%dth visibility is flagged.'%irow)
                else:
                    print('Flag on the %dth visibility is not recognized.'%irow)
        a_mat = ne.evaluate('exp(a_mat * 1j)')
        a_mat = a_mat.astype('complex128')
        a_mat = np.matrix(a_mat)
        if apply_beam:
            a_mat = np.matrix(np.multiply(a_mat, beam_mat))
        self.a_mat = a_mat
        return a_mat
    
    def beam_interp_onecore(self, time, pix):
        '''Calculating the phase for the pixels within PSF at a given time
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
        #beam_map_t = self.beam_model(np.pi/2. - alt_t, az_t, grid=False)
        #pyuvbeam_interp,_ = self.pyuvbeam.interp(az_array=az_t, za_array=np.pi/2. - alt_t, 
        #                                         az_za_grid=False, freq_array= freq_array,
        #                                         reuse_spline=True)
        print(time, 'efield interpolation')
        #pyuvbeam = self.set_pyuvbeam(beam_model=self.feed_type)
        pyuvbeam_interp_e, vectors = self.pyuvbeam.interp(az_array=az_t, za_array=np.pi/2. - alt_t, 
                                                          az_za_grid=False, freq_array= np.array([self.frequency,]),
                                                          reuse_spline=True)
        pyuvbeam_interp = self.pyuvbeam_efield_to_power(pyuvbeam_interp_e, vectors)
        ipol = 1
        beam_map_t = pyuvbeam_interp[0, 0, ipol, 0].real
        return {time: beam_map_t}
    
    def set_beam_interp(self, pix, ncores=10):
        '''Run the beam interpolation in parallel and store the result in a dictionary
        
        pix: str
            'hp', or 'hp+ps'
        
        '''
        print(pix)
        self.set_pyuvbeam(beam_model=self.feed_type)
        pool = multiprocessing.Pool(processes=ncores)
        args = []
        for time_t in np.unique(self.uv.time_array):
            args.append([time_t, pix])
        results = pool.starmap(self.beam_interp_onecore, args)
        pool.close()
        pool.join()
        beam_dic = {}
        for dic_t in results:
            beam_dic.update(dic_t)
        self.beam_dic = beam_dic
        return beam_dic
        
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

        Output:
        ------
        a_mat_ps: 2d matrix (complex64)
            a_matrix (Nvis X (Npsf+Nps)) from the given observation
        
        Attribute:
        ------
        .a_mat_ps: 2d matrix (complex64)
            a_matrix_ps added in the attribute
        '''
        a_mat = np.zeros((len(self.data), len(self.idx_psf_in)+ps_radec.shape[0]), dtype='float64')
        beam_mat = np.zeros(a_mat.shape, dtype='float64')
        #self.set_beam_model(beam_model=self.feed_type)
        self.set_pyuvbeam(beam_model=self.feed_type)
        #print('Pyuvdata readin.')
        freq_array = np.array([self.frequency,])
        self.ra_ps = ps_radec[:, 0]
        self.dec_ps = ps_radec[:, 1]
        #self.set_beam_interp('hp+ps')
        for time_t in np.unique(self.uv.time_array):
            az_t, alt_t = self._radec2azalt(np.concatenate((self.ra[self.idx_psf_in], self.ra_ps)),
                                            np.concatenate((self.dec[self.idx_psf_in], self.dec_ps)),
                                            time_t)
            lmn_t = np.array([np.cos(alt_t)*np.sin(az_t), 
                              np.cos(alt_t)*np.cos(az_t), 
                              np.sin(alt_t)])
            #beam_map_t = self.beam_model(np.pi/2. - alt_t, az_t, grid=False)
            pyuvbeam_interp,_ = self.pyuvbeam.interp(az_array=az_t, za_array=np.pi/2. - alt_t, 
                                                     az_za_grid=False, freq_array= freq_array,
                                                     reuse_spline=True)
            #print('efield interpolation')
            #pyuvbeam_interp_e, vectors = self.pyuvbeam.interp(az_array=az_t, za_array=np.pi/2. - alt_t, 
            #                                                  az_za_grid=False, freq_array= freq_array,
            #                                                  reuse_spline=True)
            #pyuvbeam_interp = self.pyuvbeam_efield_to_power(pyuvbeam_interp_e, vectors)
            #ipol = 0
            #print(ipol)
            beam_map_t = pyuvbeam_interp[0, 0, 0, 0].real
            #beam_map_t = self.beam_dic[time_t]
            idx_time = np.where(self.uv.time_array == time_t)[0]
            for i in range(len(idx_time)):
                irow = idx_time[i]
                a_mat[irow] = uvw_sign*2*np.pi/self.wavelength*np.matmul(np.matrix(self.uv.uvw_array[irow].astype(np.float32)),
                                                                         np.matrix(lmn_t.astype(np.float32)))
                if self.flag[irow] == False:
                    beam_mat[irow] = beam_map_t.astype(np.float32)
                elif self.flag[irow] == True:
                    beam_mat[irow] = np.zeros(beam_mat.shape[1])
                    print('%dth visibility is flagged.'%irow)
                else:
                    print('Flag on the %dth visibility is not recognized.'%irow)
        a_mat = ne.evaluate('exp(a_mat * 1j)')
        a_mat = a_mat.astype('complex128')
        a_mat = np.matrix(a_mat)
        if apply_beam:
            a_mat = np.matrix(np.multiply(a_mat, beam_mat))
        self.a_mat_ps = a_mat
        return a_mat
    
    def set_inv_noise_mat(self, uvn):
        '''Calculating the inverse noise matrix with auto-correlations
        Args:
        ------
        uvn: pyuvdata
            pyuvdata object with estimated noise information
        '''
        inv_noise_mat = np.diag(np.squeeze(uvn.data_array).real**(-2))
        self.inv_noise_mat = inv_noise_mat
        self.norm_factor = np.sum(np.diag(inv_noise_mat))

        return inv_noise_mat
    
    def set_p_mat(self, facet_radius_deg=7):
        '''Calculating P matrix, covering the range defined by K_psf,
        projectin to the range defined by K_facet
        
        Input:
        ------
        None
        
        Output:
        ------
        p_mat: 2d matrix (complex64) n_k_facet X n_k_psf
            p_matrix from the given observation
        p_diag: 1d array (complex64)
            normalization array for the map within the facet
        
        Attribute:
        ------
        .p_mat: 2d matrix (complex64)
            p_matrix from the given observation as an attribute
        .p_diag: 1d array (complex64)
            normalization array for the map within the facet
        .p_square: 2d matrix (complex64)
            square p matrix containing only the facet pixels on 
            both dimensions
        '''
        #p_matrix set up
        k_facet = np.matrix(self.set_k_facet(radius_deg=facet_radius_deg, calc_k=True))
        p_mat1 = np.matmul(k_facet, self.a_mat.H)
        p_mat2 = np.matmul(self.inv_noise_mat, self.a_mat)
        p_mat = np.matmul(p_mat1, p_mat2)
        p_mat = np.matrix(np.real(p_mat))
        p_mat = p_mat/self.norm_factor
        #normalizatoin factor set up
        k_facet_transpose = np.matrix(k_facet.T)
        p_square = np.matmul(p_mat, k_facet_transpose) 
        p_diag = np.diag(p_square)
        #del inv_noise_mat, k_facet, p_mat1, p_mat2
        del k_facet, p_mat1, p_mat2
        
        #attribute assignment
        self.p_mat = p_mat
        self.p_diag = p_diag
        self.p_square = p_square
        return p_mat, p_diag, p_square
    
    def set_p_mat_ps(self, facet_radius_deg=7):
        '''Calculating P matrix with stand-alone point sources, 
        covering the range defined by K_psf + point source pixels,
        projectin to the range defined by K_facet
        
        Input:
        ------
        None
        
        Output:
        ------
        p_mat_ps: 2d matrix (complex64) n_k_facet X (n_k_psf + n_ps)
            p_matrix_ps from the given observation
        
        Attribute:
        ------
        .p_mat_ps: 2d matrix (complex64)
            p_matrix_ps from the given observation as an attribute
        .p_diag_ps: 1d array (complex64)
            normalization array for the map within the facet
        .p_square_ps: 2d matrix (complex64)
            square p matrix containing only the facet pixels on 
            both dimensions
        '''
        if not hasattr(self, 'a_mat_ps'):
            print('A matrix with point sources pixel is not set up, returning None.')
            return
        #p_matrix_ps set up
        k_facet = np.matrix(self.set_k_facet(radius_deg=facet_radius_deg, calc_k=True))
        p_mat1 = np.matmul(k_facet, self.a_mat.H)
        p_mat2 = np.matmul(self.inv_noise_mat, self.a_mat_ps)
        #p_mat2 = self.a_mat_ps
        p_mat_ps = np.matmul(p_mat1, p_mat2)
        p_mat_ps = np.matrix(np.real(p_mat_ps))
        p_mat_ps = p_mat_ps/self.norm_factor
        #normalizatoin factor set up
        k_facet_transpose = np.matrix(k_facet.T)
        p_square = np.matmul(p_mat_ps[:, :len(self.idx_psf_in)], k_facet_transpose) 
        p_diag_ps = np.diag(p_square)
        #del inv_noise_mat, k_facet, k_facet_transpose, p_mat1, p_mat2
        del k_facet, k_facet_transpose, p_mat1, p_mat2
        
        #attribute assignment
        self.p_mat_ps = p_mat_ps
        self.p_diag_ps = p_diag_ps
        self.p_square_ps = p_square
        return p_mat_ps, p_diag_ps
        
    
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
