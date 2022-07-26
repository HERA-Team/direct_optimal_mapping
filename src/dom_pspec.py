import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
import hera_pspec.conversions as conversions
from uvtools import dspec

class PS_Calc:
    '''Class to calculate power spectrum from direct opitmal
    mapping data cubes
    '''
    def __init__(self, data_cube_dic):
        '''Initialization of the class
        
        Parameters
        ----------
        data_cube_dic: dictionary
            dictionary containing the data cube brightness (mK)
            and the coordinates (ra/dec/freqency)
        
        Returns
        -------
        
        '''
        self.freq_mhz = data_cube_dic['freq_mhz']
        self.px_dic = data_cube_dic['px_dic']
        self.data_cube = data_cube['data_cube']
        
        return
    
    def set_cosmo_grid(self, beta=1., buffer=200e-9):
        '''Calculating cosmological coordinates from the measurement
        coordinates and set the comological grid
        
        Parameters
        ----------
        beta: float
            fraction of the hemisphere to consider for wedge calculation 
            Default: 1. (full hemisphere)
        buffer: float
            wedge buffer in unit of seconds
        
        Returns
        -------
        '''
        self.redshift = 1420.406/self.freq_mhz - 1
        self.dist_cm = cosmo.comoving_distance(self.redshift)
        self.dist_cm_h = dist_cm.to(u.Mpc/u.littleh, equivalencies=u.with_H0(cosmo.H0))

        z_avg = np.mean(self.redshift)
        freq_avg = np.mean(freq_mhz_arr)*1e6
        cos_conversion = conversions.Cosmo_Conversions(Om_L=cosmo.Ode0, Om_b=cosmo.Ob0, Om_c=cosmo.Odm0, H0=cosmo.H0.value)
        slope = beta*cos_conversion.dRperp_dtheta(z_avg)/(cos_conversion.dRpara_df(z_avg)*np.mean(freq_avg))
        y_intercept = buffer*cos_conversion.tau_to_kpara(z_avg)
        
        self.nx, self.ny, self.nz = self.data_cube.shape
        self.n_voxel = nx*ny*nz

        dist_cm_h_avg = np.mean(self.dist_cm_h)

        self.res_x_deg = np.mean(np.diff(self.px_dic['ra_deg'][:, 0]))
        self.res_x_mpch = np.radians(self.res_x_deg)*dist_cm_h_avg.value

        self.res_y_deg = np.mean(np.diff(self.px_dic['dec_deg'][0, :]))
        self.res_y_mpch = np.radians(self.res_y_deg)*dist_cm_h_avg.value

        self.res_z_mpch = np.mean(np.abs(np.diff(self.dist_cm_h))).value
        self.voxel_volume = self.res_x_mpch * self.res_y_mpch * self.res_z_mpch
        
        return
    
    def calc_fft(self, window='bh7'):
        '''Calculating the fft of the data cube with the window applied
        along the frequency direction.
        '''
        z_window = dspec.gen_window(window, self.nz)
        data_cube_tapered = self.data_cube * z_window[np.newaxis, np.newaxis, :]
        
        self.fft3d = self.voxel_volume*np.fft.fftn(data_cube_tapered)

        return
    
    def set_k_space(self):
        '''Setting the k-space grid from the cosmological grid
        '''
        self.kx = np.fft.fftfreq(self.nx, d=self.res_x_mpch)*2*np.pi
        self.ky = np.fft.fftfreq(self.ny, d=self.res_y_mpch)*2*np.pi
        self.kz = np.fft.fftfreq(self.nz, d=self.res_z_mpch)*2*np.pi

        self.k_xx, self.k_yy, self.k_zz = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij')

        self.k_perp = np.sqrt(np.average(self.k_xx, axis=2)**2 + 
                              np.average(self.k_yy, axis=2)**2)
        n_perp = max(nx//2, ny//2)
        k_perp_edge = np.linspace(0, np.max(self.k_perp), n_perp+1)
        self.ps2d = np.zeros((n_perp, self.nz))
        for i in range(len(k_perp_edge)-1):
            idx_t = np.where((self.k_perp > k_perp_edge[i]) & (self.k_perp < k_perp_edge[i+1]))
            self.ps2d[i] = np.average(np.abs(self.fft3d[idx_t])**2, axis=0)/(self.n_voxel*self.voxel_volume)
        self.ps2d = self.ps2d[:, :self.nz//2]
        
        return
        
        
        
        
    
    