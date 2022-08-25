import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy.cosmology.units import littleh, with_H0
import hera_pspec.conversions as conversions
from uvtools import dspec

class PS_Calc:
    '''Class to calculate power spectrum from direct opitmal
    mapping data cubes
    '''
    def __init__(self, data_cube_dic1, data_cube_dic2=None):
        '''Initialization of the class
        
        Parameters
        ----------
        data_cube_dic1: dictionary
            dictionary containing the data cube brightness (mK)
            and the coordinates (ra/dec/freqency)
        data_cube_dic2: dictionary or None
            identical to data_cube_dic1 with independent noise propertie. 
            If None, data_cube_dic2 = data_cube_dic1
        
        Returns
        -------
        
        '''
        if data_cube_dic2 == None:
            data_cube_dic2 = data_cube_dic1
        self.freq_mhz = data_cube_dic1['freq_mhz']
        self.px_dic = data_cube_dic1['px_dic']
        self.data_cube1 = data_cube_dic1['data_cube_I']
        self.data_cube2 = data_cube_dic2['data_cube_I']
        
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
        self.dist_cm_h = self.dist_cm.to(u.Mpc/littleh, equivalencies=with_H0(cosmo.H0))

        z_avg = np.mean(self.redshift)
        freq_avg = np.mean(self.freq_mhz)*1e6
        cos_conversion = conversions.Cosmo_Conversions(Om_L=cosmo.Ode0, Om_b=cosmo.Ob0, Om_c=cosmo.Odm0, H0=cosmo.H0.value)
        self.slope = beta*cos_conversion.dRperp_dtheta(z_avg)/(cos_conversion.dRpara_df(z_avg)*np.mean(freq_avg))
        self.y_intercept = buffer*cos_conversion.tau_to_kpara(z_avg)
        
        self.nx, self.ny, self.nz = self.data_cube1.shape
        self.n_voxel = self.nx*self.ny*self.nz

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
        data_cube1_tapered = self.data_cube1 * z_window[np.newaxis, np.newaxis, :]
        data_cube2_tapered = self.data_cube2 * z_window[np.newaxis, np.newaxis, :]
        
        self.fft3d1 = self.voxel_volume*np.fft.fftn(data_cube1_tapered)
        self.fft3d2 = self.voxel_volume*np.fft.fftn(data_cube2_tapered)
        self.ps3d = self.fft3d1.conjugate() * self.fft3d2 / (self.n_voxel*self.voxel_volume)
        self.ps3d = self.ps3d.real
        
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
        self.k_para = self.kz[:self.nz//2]
        n_perp = max(self.nx//2, self.ny//2)
        self.k_perp_edge = np.linspace(0, np.max(self.k_perp), n_perp+1)
        self.ps2d = np.zeros((n_perp, self.nz))
        for i in range(len(self.k_perp_edge)-1):
            idx_t = np.where((self.k_perp > self.k_perp_edge[i]) & (self.k_perp < self.k_perp_edge[i+1]))
            self.ps2d[i] = np.average(self.ps3d[idx_t], axis=0)
        self.ps2d = self.ps2d[:, :self.nz//2]
        self.k_perp = self.k_perp_edge[:-1]
        
        return
    
    def calc_ps1d(self, nbin=None, avoid_fg=True):
        '''Calculating 1d PS from the 3d PS
        Parameters
        ----------
        nbin: int or None
            Number of 1d k-bin numbers. Default: None, using the bin number from nz
        avoid_fg: bool
            Avoid foreground or not, avoid the wedge + buffer (defined in set_cosmo_grid)
        
        '''
        if nbin == None:
            nbin = self.nz//2
        kr = np.sqrt(self.k_xx**2 + self.k_yy**2 + self.k_zz**2)
        if avoid_fg == True:
            k_perp = np.sqrt(self.k_xx**2 + self.k_yy**2)
            fg_flag = self.k_zz > k_perp * self.slope + self.y_intercept
            k_min = self.y_intercept
        else:
            fg_flag = np.ones(kr.shape, dtype='bool')
            k_min = 0
        kr_edge = np.linspace(k_min, kr.max(), nbin+1)
        ps1d = []
        ps1d_var = []
        for i in range(nbin):
            idx_t = np.where((kr > kr_edge[i]) & (kr < kr_edge[i+1]) & fg_flag)
            ps1d.append(np.average(self.ps3d[idx_t]).value)
            ps1d_var.append(np.mean(self.ps3d[idx_t].value**2) - np.mean(self.ps3d[idx_t].value)**2)
        self.kr = kr_edge[:-1] + np.mean(np.diff(kr_edge))/2.
        self.ps1d = np.array(ps1d)
        self.ps1d_std = np.sqrt(ps1d_var)
        
        return
        