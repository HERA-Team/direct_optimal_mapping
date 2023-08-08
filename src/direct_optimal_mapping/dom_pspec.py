import numpy as np
import scipy
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy.cosmology.units import littleh, with_H0
import hera_pspec.conversions as conversions
from uvtools import dspec

class PS_Calc:
    '''Class to calculate power spectrum from direct opitmal
    mapping data cubes
    '''
    def __init__(self, data_cube_dic1, data_cube_dic2=None, 
                 par_taper='bh', per_taper='tukey'):
        '''Initialization of the class
        
        Parameters
        ----------
        data_cube_dic1: dictionary
            dictionary containing the data cube brightness (mK)
            and the coordinates (ra/dec/freqency)
        data_cube_dic2: dictionary or None
            identical to data_cube_dic1 with independent noise propertie. 
            If None, data_cube_dic2 = data_cube_dic1
        par_taper, per_taper: str
            Taper function type along the frequency axis and across the sky-plane
            accepted inputs are 'tophat', 'hann', 'tukey', 'bh4', 'bh7', 'cs9', and 'cs11'
            No tapering when None is given

        Returns
        -------
        
        '''
        if data_cube_dic2 == None:
            data_cube_dic2 = data_cube_dic1
        self.freq_mhz = data_cube_dic1['freq_mhz']
        self.px_dic = data_cube_dic1['px_dic']
        self.data_cube1 = data_cube_dic1['data_cube_I']
        self.data_cube2 = data_cube_dic2['data_cube_I']
        self.beam_pwr_corr = data_cube_dic1['beam_pwr_corr']
        self.syn_beam_sr = data_cube_dic1['syn_beam_sr']
        self.par_taper = par_taper
        self.per_taper = per_taper
        self.nz, self.nx, self.ny = self.data_cube1.shape
        
        # Setting up the tapering function in image cube
        if self.par_taper is None:
            z_taper = np.ones(self.nz)
        else:
            z_taper = dspec.gen_window(self.par_taper, self.nz)
        if self.per_taper is None:
            x_taper = np.ones(self.nx)
            y_taper = np.ones(self.ny)
        else:
            x_taper = dspec.gen_window(self.per_taper, self.nx)
            y_taper = dspec.gen_window(self.per_taper, self.ny)
        
        self.taper_3d = np.ones((self.nz, self.nx, self.ny))
        self.taper_3d *= z_taper[:, np.newaxis, np.newaxis]
        self.taper_3d *= x_taper[np.newaxis, :, np.newaxis]
        self.taper_3d *= y_taper[np.newaxis, np.newaxis, :]
        
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
        
        self.n_voxel = self.nx*self.ny*self.nz

        dist_cm_h_avg = np.mean(self.dist_cm_h)

        self.res_x_deg = np.mean(np.diff(self.px_dic['ra_deg'][:, 0]))
        self.res_x_mpch = np.radians(self.res_x_deg)*dist_cm_h_avg.value

        self.res_y_deg = np.mean(np.diff(self.px_dic['dec_deg'][0, :]))
        self.res_y_mpch = np.radians(self.res_y_deg)*dist_cm_h_avg.value

        self.res_z_mpch = np.mean(np.abs(np.diff(self.dist_cm_h))).value
        self.voxel_volume = self.res_x_mpch * self.res_y_mpch * self.res_z_mpch
        
        self.syn_beam_mpch = np.average(self.syn_beam_sr) * dist_cm_h_avg.value
        
        return
    
    def calc_fft(self, volume='original'):
        '''Calculating the fft of the data cube with the taper applied
        along the frequency direction.
        
        Parameters
        ----------
        volume: str
            'original' or 'effective', referring to the original size
            of the image cube or the ones attenuated by the primary beam
            and the tapering
        
        '''
        
#         if self.taper_type == None:
#             data_cube1_tapered = self.data_cube1
#             data_cube2_tapered = self.data_cube2
#             norm_factor = 1.
#         else:
#             z_taper = dspec.gen_window(self.taper_type, self.nz)
#             self.taper_3d = z_taper[:, np.newaxis, np.newaxis]
#             data_cube1_tapered = self.data_cube1 * self.taper_3d
#             data_cube2_tapered = self.data_cube2 * self.taper_3d
#             norm_factor = self.nz/np.sum(self.taper_3d**2)
#             self.kpara_taper = np.fft.fftn(z_taper)
#         if self.perp_apdz:
#             x_window = dspec.gen_window(self.taper_type, self.nx)
#             data_cube1_tapered = data_cube1_tapered * x_window[np.newaxis, :, np.newaxis]
#             data_cube2_tapered = data_cube2_tapered * x_window[np.newaxis, :, np.newaxis]

#             y_window = dspec.gen_window(self.taper_type, self.ny)
#             data_cube1_tapered = data_cube1_tapered * y_window[np.newaxis, np.newaxis, :]
#             data_cube2_tapered = data_cube2_tapered * y_window[np.newaxis, np.newaxis, :]
        data_cube1_tapered = self.data_cube1 * self.taper_3d
        data_cube2_tapered = self.data_cube2 * self.taper_3d
        
#         self.fft3d1 = (self.voxel_volume) * np.fft.fftn(data_cube1_tapered, norm='ortho')
#         self.fft3d2 = (self.voxel_volume) * np.fft.fftn(data_cube2_tapered, norm='ortho')
#         self.ps3d = self.fft3d1.conjugate() * self.fft3d2 * self.voxel_volume
        self.fft3d1 = np.fft.fftn(data_cube1_tapered, norm='ortho')
        self.fft3d2 = np.fft.fftn(data_cube2_tapered, norm='ortho')
        self.ps3d = self.fft3d1.conjugate() * self.fft3d2 * self.voxel_volume
        if volume == 'original':
            self.ps3d = self.ps3d
        elif volume == 'effective':
            tp_corr = np.sum(self.taper_3d**2)/len(self.taper_3d.flatten())
            corr = tp_corr * self.beam_pwr_corr
            self.ps3d = self.ps3d/corr
        self.ps3d = self.ps3d.real
#         if norm:
#             self.ps3d = self.ps3d * norm_factor
                
        return
    
    def norm_calc(self, p_dic, norm_data=True):
        '''Calculating the normalization factor for the 3D power
        spectrum
        Parameters
        ----------
        p_dic: dictionary
            containing the p matrices
        norm_data: boolean
            whether normalize the ps3d data in place
        Return
        ------
        None
        '''
        nx, ny = p_dic['px_dic']['ra_deg'].shape
        nz, _, _ = p_dic['p_mat_I'].shape
        taper_3d_p = self.taper_3d.reshape((nz, nx * ny))
        rp_mat = p_dic['p_mat_I'] * np.expand_dims(taper_3d_p, axis=2)
        
        n_px = nx * ny
        n_vx = nz * nx * ny
        m_diag = np.zeros(n_vx)
        for i in range(n_vx):
            ifreq = i//n_px
            ip = i%n_px
            col_t = np.zeros(n_vx)
            col_t[n_px*ifreq:n_px*(ifreq+1)] = rp_mat[ifreq, :, ip]
            col_t_reshape = col_t.reshape((nz, nx, ny))
            col_t_tilda = np.fft.fftn(col_t_reshape, norm='ortho').flatten()
            m_diag += np.abs(col_t_tilda)**2
        self.m_diag = m_diag
        if norm_data is True:
            self.ps3d = self.ps3d/m_diag.reshape((nz, nx, ny))        
        
        return
        
    
    def set_k_space(self, ps3d=None, binning='lin', n_perp=None):
        '''Setting the k-space grid from the cosmological grid
        Parameters
        ----------
        ps3d: array
            power spectrum in 3d. Default is None, using self.ps3d
        binning: str
            binning can be 'lin' or 'log', meaning binning k_perp evenly in
            linear or log space
        n_perp: int
            number of k_perp bins
        Return
        ------
        '''
        
        if ps3d is None:
            ps3d = self.ps3d
        
        self.kx = np.fft.fftfreq(self.nx, d=self.res_x_mpch)*2*np.pi
        self.ky = np.fft.fftfreq(self.ny, d=self.res_y_mpch)*2*np.pi
        self.kz = np.fft.fftfreq(self.nz, d=self.res_z_mpch)*2*np.pi
        kx_res = np.mean(np.diff(self.kx[:self.nx//2]))
        ky_res = np.mean(np.diff(self.ky[:self.ny//2]))
        self.syn_beam_k = 2*np.pi*(1/self.syn_beam_mpch)

        self.k_zz, self.k_xx, self.k_yy = np.meshgrid(self.kz, self.kx, self.ky, indexing='ij')
        self.mask_3d = np.sqrt(self.k_xx**2 + self.k_yy**2) < self.syn_beam_k

        self.k_perp = np.sqrt(np.average(self.k_xx, axis=0)**2 + 
                              np.average(self.k_yy, axis=0)**2)
        if n_perp is None:
            n_perp = max(self.nx//2, self.ny//2)
        n_para = self.nz//2
        self.k_para = self.kz[:n_para]
        if binning == 'lin':
            self.k_perp_edge = np.linspace(np.sqrt(kx_res * ky_res), np.max(self.k_perp), n_perp+1)
        elif binning == 'log':
            self.k_perp_edge = np.geomspace(np.min(self.k_perp[self.k_perp>0])/2., 
                                            np.max(self.k_perp), n_perp+1)
        else:
            raise RuntimeError('Wrong binning input.')
        self.ps2d = np.zeros((n_perp, self.nz))
        self.ps2d_se = np.zeros((n_perp, self.nz))
        for i in range(n_perp):
            idx_t = np.where((self.k_perp >= self.k_perp_edge[i]) & (self.k_perp < self.k_perp_edge[i+1]))
            self.ps2d[i] = np.average(ps3d[:, idx_t[0], idx_t[1]], axis=1)
            self.ps2d_se[i] = np.std(ps3d[:, idx_t[0], idx_t[1]], axis=1)/np.sqrt(len(idx_t[0]))
        self.ps2d = self.ps2d[:, :n_para]
        self.k_perp = self.k_perp_edge[:-1] + np.mean(np.diff(self.k_perp_edge))/2.
        
        return

    
    def calc_p_tilda(self, p_dic, normalize=True):
        '''FFT of the 3d p_mat

        Parameter
        ---------
        p_dic: dictioinary
            storing the p matrix for each frequencc, 
            N_freq X N_pix X N_pix
        normalize: Boolean
            whether normalize ps3d with the h_sum3d

        Return
        ------
        '''
#         z_taper = dspec.gen_window(self.taper_type, self.nz)
        shape = [self.nz, self.nx, self.ny]
#         p_mat_I_tapered = p_dic['p_mat_I'] * z_taper[:, np.newaxis, np.newaxis]       
#         if self.perp_apdz:
#             x_window = dspec.gen_window(self.taper_type, self.nx)
#             y_window = dspec.gen_window(self.taper_type, self.ny)
#             p_mat_I_tapered = p_mat_I_tapered * x_window[np.newaxis, :, np.newaxis]
#             p_mat_I_tapered = p_mat_I_tapered * y_window[np.newaxis, np.newaxis, :]
            
        self.taper_3d_p = self.taper_3d.reshape((self.nz, self.nx * self.ny))
        p_mat_I_tapered = p_dic['p_mat_I'] * np.expand_dims(self.taper_3d_p, axis=2)
#         p_mat_I_tapered *= np.expand_dims(self.taper_3d_p, axis=2)
    
        p_3d = scipy.linalg.block_diag(*p_mat_I_tapered)
        p_tilda1 = np.zeros(p_3d.shape, dtype='complex128')
        for i in range(p_tilda1.shape[1]):
            p_col = p_3d[:, i]
            p_col_reshape = p_col.reshape(shape)
            p_col_tilda = np.fft.fftn(p_col_reshape, norm='ortho').flatten()
            p_tilda1[:, i] = p_col_tilda# * self.voxel_volume
        p_tilda = np.zeros(p_tilda1.shape, dtype='complex128')
        for i in range(p_tilda.shape[0]):
            p_row = p_tilda1[i, :]
            p_row_reshape = p_row.reshape(shape)
            p_row_tilda = np.fft.ifftn(p_row_reshape, norm='ortho').flatten()
#             p_row_tilda = np.conjugate(p_row_tilda)
            p_tilda[i, :] = p_row_tilda# * self.voxel_volume
#         p_tilda = p_tilda * self.voxel_volume       
        self.h_mat = np.abs(p_tilda)**2
        self.p_tilda = p_tilda
        self.h_mat_masked = self.h_mat * self.mask_3d.flatten()[np.newaxis, :]
        self.h_sum3d = np.sum(self.h_mat_masked, axis=1).reshape(shape)
        if normalize:
            self.ps3d = self.ps3d/self.h_sum3d
        
        return
    
    def calc_norm_fft(self):
        '''Calculating the normalized FFT and PS using
        p_tilda
        '''
        if not hasattr(self, 'p_tilda'):
            print('calc_p_tilda should have been run first.')
            return
        if not hasattr(self, 'fft3d1'):
            print('calc_fft should have been run first.')
            return
        fft3d1_norm = np.matmul(self.p_tilda, self.fft3d1)
        fft3d2_norm = np.matmul(self.p_tilda, self.fft3d2)
        self.ps3d_norm = fft3d1_norm.conjugate() * fft3d2_norm * self.voxel_volume
#         self.ps3d = self.ps3d.real
        return
    
    def h_mat_binning(self):
        '''Binning the 3d h_mat into k_para and k_perp
        '''
        k_perp = np.sqrt(self.k_xx**2 +self.k_yy**2)

        n_para_bin = len(self.kz)
        n_perp_bin = len(self.k_perp_edge) - 1

        h_mat_2d_partial = np.zeros((self.h_mat.shape[0], n_para_bin*n_perp_bin))
        for k in range(self.h_mat.shape[0]):
            h_mat_row_t = self.h_mat[k]
            h_mat_row_t_reshaped = h_mat_row_t.reshape([n_para_bin, len(self.kx), len(self.ky)])
            h_mat_row_t_2d = np.zeros([n_para_bin, n_perp_bin], dtype='complex128')
            for j in range(n_para_bin):
                p_perp_t = np.zeros(n_perp_bin, dtype='complex128')
                for i in range(n_perp_bin):
                    idx_t = np.where((k_perp[j, :, :] >= self.k_perp_edge[i]) & 
                                     (k_perp[j, :, :] < self.k_perp_edge[i+1]))
                    p_perp_t[i] = np.average(h_mat_row_t_reshaped[j][idx_t])
                h_mat_row_t_2d[j] = p_perp_t
            h_mat_2d_partial[k] = h_mat_row_t_2d.flatten()

        h_mat_2d = np.zeros((n_para_bin*n_perp_bin, n_para_bin*n_perp_bin))
        for k in range(h_mat_2d_partial.shape[1]):
            h_mat_row_t = h_mat_2d_partial[:, k]
            h_mat_row_t_reshaped = h_mat_row_t.reshape([n_para_bin, len(self.kx), len(self.ky)])
            h_mat_row_t_2d = np.zeros([n_para_bin, n_perp_bin], dtype='complex128')
            for j in range(n_para_bin):
                p_perp_t = np.zeros(n_perp_bin, dtype='complex128')
                for i in range(n_perp_bin):
                    idx_t = np.where((k_perp[j, :, :] > self.k_perp_edge[i]) & 
                                     (k_perp[j, :, :] < self.k_perp_edge[i+1]))
                    p_perp_t[i] = np.average(h_mat_row_t_reshaped[j][idx_t])
                h_mat_row_t_2d[j] = p_perp_t
            h_mat_2d[:, k] = h_mat_row_t_2d.flatten()
            
        self.h_mat_2d = h_mat_2d.real
        
        h_sum = np.nansum(self.h_mat_2d[:, :], axis=1).reshape(n_para_bin, n_perp_bin)
        self.window = 1/h_sum
        
        return
    
    
    def calc_ps1d(self, nbin=None, avoid_fg=True, max_kperp=None):
        '''Calculating 1d PS from the 3d PS
        Parameters
        ----------
        nbin: int or None
            Number of 1d k-bin numbers. Default: None, using the bin number from nz
        avoid_fg: bool
            Avoid foreground or not, avoid the wedge + buffer (defined in set_cosmo_grid)
        max_kperp: float
            Max k_perp to include given the instrument max baseline (unit: h/Mpc)
        
        '''
        if nbin == None:
            nbin = self.nz//2
        if max_kperp == None:
            max_kperp = self.syn_beam_k
        kr = np.sqrt(self.k_xx**2 + self.k_yy**2 + self.k_zz**2)
        k_perp = np.sqrt(self.k_xx**2 + self.k_yy**2)
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
        n_sample = []
        for i in range(nbin):
            idx_t = np.where((kr > kr_edge[i]) & (kr < kr_edge[i+1]) & fg_flag & (k_perp < max_kperp))
            ps1d.append(np.average(self.ps3d[idx_t]))
            ps1d_var.append(np.mean(self.ps3d[idx_t]**2) - np.mean(self.ps3d[idx_t])**2)
            n_sample.append(len(idx_t[0]))
        self.kr = kr_edge[:-1] + np.mean(np.diff(kr_edge))/2.
        self.ps1d = np.array(ps1d)
        self.ps1d_std = np.sqrt(ps1d_var)
        self.ps1d_nsample = np.array(n_sample)
        
        return
        
