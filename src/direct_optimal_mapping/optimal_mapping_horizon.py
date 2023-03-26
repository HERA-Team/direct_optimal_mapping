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

from . import pixel_selection

class OptMappingHorizon:
    '''Optimal Mapping Object for  horizon coordinate mapping
    '''
    
    def __init__(self, uv, cellsize, epoch='J2000', feed=None,
                 beam_file = None,
                 beam_folder='/nfs/esc/hera/zhileixu/git_beam/HERA-Beams/NicolasFagnoniBeams'):
        '''Init function for basic setup
         
        Input
        ------
        uv: pyuvdata object
            UVData data in the pyuvdata format, data_array only has the blt dimension
        cellsize: flt
            cellsize in arcseconds on the horizon and in declination
            a close value that is commensurate with LST spacing of data will be calculated
            actual cellsize will be the LST spacing of the data or smaller
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
        
        print('This is the test version of optimal_mapping_horizon.py')
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
        else:
            self.beam_file = beam_file

        # calculate time spacing and actual cell size
        # cell size cannot be larger than the LST spacing
        deltat_sec=(self.times[1]-self.times[0])*24.*3600.
        # cannot do requested cell size - change it and report
        if (cellsize > deltat_sec*15.):
            cellsize_asec = deltat_sec*15.
            print('Map cell size set to LST spacing of data: ',cellsize_asec,' arc seconds')
        else:
            cellsize_asec=(deltat_sec*15.)/int(deltat_sec*15./cellsize)
        self.cellsize_asec=cellsize_asec
    
        # create RA dec meshgrid wiith cellsize_asec spacing
        # align the first RA cell with the first LST (is this necessary?)
        # create az alt meshgrid for use on each time stamp
        numra=int(24.*15./(cellsize_asec/3600.))
        #print('deltat_sec, cellsize_asec, num for ra', deltat_sec, cellsize_asec,numra) 
        x1=np.linspace(0.,360.,numra,endpoint=False)
        x1=np.mod(x1+np.degrees(self.lsts[0]),360.)
        numdec=int(180./(cellsize_asec/3600.))
        #print('num for declination is ', numdec)
        self.npix = numra*numdec
        x2=np.linspace(-90.,90.,numdec,endpoint=False)
        ra,dec = np.meshgrid(x1,x2)
        ra=np.radians(np.matrix.flatten(ra))
        dec=np.radians(np.matrix.flatten(dec))
        az, alt = self._radec2azalt(ra, dec, self.times[0])
        self.ra = ra
        self.dec = dec
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


    def set_k_psf_horizon(self, radius_deg, calc_k=False):
        '''Function to set up the K_psf matrix. Original K_psf selects
        healpix from the entire sky to the regions within a
        certain radius away from the phase center
        For the horizon version, indices are not healpix indices
        but rather the indices of the flattened map array.
        The code is the same as set_k_psf because the code does
        not care what kind of indices it is working with.

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
            map indices within the PSF
        .k_psf_out: 1d array (int)
            map indices outside of the PSF
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
        #pyuvbeam.select(polarizations=[-6,])
        #print(pyuvbeam.polarization_array)
        pyuvbeam.peak_normalize()
        pyuvbeam.interpolation_function = 'az_za_simple'
        pyuvbeam.freq_interp_kind = 'cubic'

        # attribute assignment
        self.pyuvbeam = pyuvbeam
        return


    def set_a_mat_horizon(self, uvw_sign=1, apply_beam=True):
        '''Calculating A matrix for the hemisphere overhead
        
        Input:
        ------
        uvw_sign: 1 or -1
            uvw sign for the baseline calculation
        apply_beam: boolean
            Whether apply beam to the a matrix elements, default:true
        
        Attribute:
        ------
        ..a_mat_horizon: 2d matrix (complex128)
            a_matrix (Nvis X Npsf) from the given observation
        .beam_mat_horizon: 2d matrix (float64)
            a_matrix with only the beam term considered (Nvis X Npsf)
        '''
        self.set_pyuvbeam(beam_file=self.beam_file)
        freq_array = np.array([self.frequency,])

        self.az_mat = self.az[self.idx_psf_in]
        self.alt_mat = self.alt[self.idx_psf_in]
        print('az shape after psf limit: ',self.az_mat.shape)
        print('alt shape after psf limit: ',self.alt_mat.shape)
        utimes=np.unique(self.uv.time_array)
        idx_t=np.where(self.uv.time_array==utimes[0])[0]
        lmn = np.array([np.cos(self.alt_mat)*np.sin(self.az_mat), 
                        np.cos(self.alt_mat)*np.cos(self.az_mat), 
                        np.sin(self.alt_mat)])
        pyuvbeam_interp,_ = self.pyuvbeam.interp(az_array=np.mod(np.pi/2. - self.az_mat, 2*np.pi), 
                                                 za_array=np.pi/2. - self.alt_mat, 
                                                 az_za_grid=False, freq_array= freq_array,
                                                 reuse_spline=True) 
        beam_map_t = pyuvbeam_interp[0, 0, 0, 0].real
        self.a_mat_horizon = uvw_sign*2*np.pi/self.wavelength*(self.uv.uvw_array[idx_t]@lmn)
        self.beam_mat_horizon = np.tile(beam_map_t, idx_t.size).reshape(idx_t.size,-1)
        self.a_mat_horizon = ne.evaluate('exp(A * 1j)', global_dict={'A':self.a_mat_horizon})
        if apply_beam:
            # indices no longer involve time so this doesn't work
            # self.beam_mat_horizon[self.flag.flatten()] = 0
            self.a_mat_horizon = np.multiply(self.a_mat_horizon, self.beam_mat_horizon)

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

    def set_p_mat_horizon(self, facet_radius_deg=7, facet_idx=None):
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
        if not hasattr(self, 'a_mat_horizon'):
            raise AttributeError('A matrix is not set up.')

        if facet_idx is None:
            self.set_k_facet_horizon(radius_deg=facet_radius_deg, calc_k=False)
        else:
            self.idx_facet_in = facet_idx

        _idx = np.searchsorted(self.idx_psf_in, self.idx_facet_in) #Equivalent to Finding K_facet
        p_mat1 = np.conjugate(self.a_mat_horizon.T)[_idx] #Equivalent to K_facet@a_mat.H
        p_mat2 = np.diag(self.inv_noise_mat)[:, None]*self.a_mat_horizon
        #Equivalent to inv_noise_mat@a_mat, assuming diagonal noise matrix

        self.p_mat = np.real(np.matmul(p_mat1, p_mat2))
        del p_mat1, p_mat2

        self.p_square = self.p_mat[:, _idx]
        self.p_diag = np.diag(self.p_square)
        
        return

    def set_k_facet_horizon(self, radius_deg, calc_k=False):
        '''Calculating the K_facet matrix
        This code is same as non-horizon version because it
        doesn't matter what kind of indices you use.
        
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
