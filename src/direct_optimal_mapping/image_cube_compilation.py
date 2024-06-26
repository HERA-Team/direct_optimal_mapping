import numpy as np
import pickle
from astropy import constants as const
import re
from scipy import interpolate

class ImgCube:
    '''Converting individual maps into an image cube,
    also convert the image cube into mK units
    '''
    def __init__(self, files_n5, files_n6):
        '''Initialize the object
        '''
        self.files_n5 = files_n5
        self.files_n6 = files_n6
#         with open(syn_sa_file, 'rb') as f_t:
#             self.syn_sa_dic = pickle.load(f_t)            
#         self.sa_interp = interpolate.interp1d(self.syn_sa_dic['freq_mhz'], self.syn_sa_dic['sa'],
#                                               bounds_error=False, fill_value='extrapolate')
        return
        
    def image_cube_calc(self, aper_dia=14, units='unnormalized'):
        '''Image cube compilation
        '''
        assert len(self.files_n5) == len(self.files_n6), 'Pol-5 and Pol-6 do not have same number of maps.'
        for i in range(len(self.files_n5)):
            file_n5_t = self.files_n5[i]
            file_n6_t = self.files_n6[i]

            with open(file_n5_t, 'rb') as f_t:
                map_dic_n5 = pickle.load(f_t)
            with open(file_n6_t, 'rb') as f_t:
                map_dic_n6 = pickle.load(f_t)

#             freq_mhz = float(re.search('_(......)MHz', file_n5_t).group(1))
#             freq_mhz = np.linspace(100, 200, 1024, endpoint=False)[515:695][i]
            
            freq_mhz = map_dic_n5['freq']/1e6
            wv = const.c.value/map_dic_n5['freq']
            aper_area = np.pi*(aper_dia/2)**2
            beam_sa = wv**2/aper_area
        
            bl_max = map_dic_n5['bl_max']
            radius2ctr = map_dic_n5['radius2ctr']
            syn_beam_size = 1.22*const.c.value/map_dic_n5['freq']/(bl_max * np.cos(radius2ctr))
            syn_beam_sa = syn_beam_size.flatten()**2
#             beam_dilution = syn_beam_size**2/map_dic_n5['px_dic']['sa_sr']
#             beam_dilution = beam_dilution.flatten()

#             syn_sa = self.sa_interp(freq_mhz)
#             print(i, freq_mhz, 'MHz', end=',')
            
            if units=='unnormalized':
                # normalization d calculation
                d_diag = 1/map_dic_n5['beam_weight_sum']# * np.square(map_dic_n5['px_dic']['sa_sr']).flatten()) # vis -> Jy/sr
                self.px_sa = map_dic_n5['px_dic']['sa_sr'].flatten()
#             d_diag = 1/(map_dic_n5['beam_weight_sum'] * map_dic_n5['px_dic']['sa_sr'].flatten() * syn_sa) # vis -> Jy/sr
                jy2mKsr = 1e-26*const.c.value**2/2/(1e6*freq_mhz)**2/const.k_B.value*1e3
#             d_diag = d_diag * jysr2mk.value # Jy/sr -> mK
                corr = np.sum(map_dic_n5['beam_sq_weight_sum']/map_dic_n5['n_vis'])/len(map_dic_n5['beam_sq_weight_sum'])
                map_n5_t = map_dic_n5['map_sum'].squeeze() * d_diag * jy2mKsr / beam_sa #* syn_beam_sa) # / beam_dilution
                map_n6_t = map_dic_n6['map_sum'].squeeze() * d_diag * jy2mKsr / beam_sa #* syn_beam_sa) # / beam_dilution
            else:
                map_n5_t = map_dic_n5['map_sum'].squeeze() 
                map_n6_t = map_dic_n6['map_sum'].squeeze() 


            
            if i == 0:
                data_dic = {'px_dic':map_dic_n5['px_dic']}
                img_cube_n5 = map_n5_t
                img_cube_n6 = map_n6_t
                freq_mhz_arr = np.array([freq_mhz,])
                if units=='unnormalized':
                    self.d_diag = d_diag
                    self.jy2mKsr = jy2mKsr
                    self.beam_sa = beam_sa
                    self.syn_beam_sa = syn_beam_sa
                    self.beam_pwr_corr = corr
            else:
                img_cube_n5 = np.vstack((img_cube_n5, map_n5_t))
                img_cube_n6 = np.vstack((img_cube_n6, map_n5_t))
                freq_mhz_arr = np.append(freq_mhz_arr, freq_mhz)
                if units=='unnormalized':
                    self.d_diag = np.vstack((self.d_diag, d_diag))
                    self.jy2mKsr = np.vstack((self.jy2mKsr, jy2mKsr))
                    self.beam_sa = np.vstack((self.beam_sa, beam_sa))
                    self.syn_beam_sa = np.vstack((self.syn_beam_sa, syn_beam_sa))
                    self.beam_pwr_corr = np.vstack((self.beam_pwr_corr, corr))
                
        img_cube_n5 = img_cube_n5.squeeze().reshape(((-1, *map_dic_n5['px_dic']['ra_deg'].shape)))
        img_cube_n6 = img_cube_n6.squeeze().reshape(((-1, *map_dic_n6['px_dic']['ra_deg'].shape)))

        data_dic['data_cube_pol-5'] = img_cube_n5
        data_dic['data_cube_pol-6'] = img_cube_n6
        data_dic['data_cube_I'] = 0.5*(img_cube_n5 + img_cube_n6)
        data_dic['freq_mhz'] = freq_mhz_arr
        if units=='unnormalized':
            data_dic['beam_pwr_corr'] = np.mean(self.beam_pwr_corr)
            data_dic['syn_beam_sr'] = np.sqrt(self.syn_beam_sa)
        self.data_dic = data_dic
        
        return self.data_dic
    
    def p_mat_calc(self, norm=True):
        '''Calculating p matrices
        
        Parameters
        ----------
        norm: boolean
            whether multiply D on the left of A^dagger N^-1 A
        Returns
        -------
        
        '''
        for i in range(len(self.files_n5)):
            file_n5_t = self.files_n5[i]
            file_n6_t = self.files_n6[i]

            with open(file_n5_t, 'rb') as f_t:
                map_dic_n5 = pickle.load(f_t)
            with open(file_n6_t, 'rb') as f_t:
                map_dic_n6 = pickle.load(f_t)

#             freq_mhz = float(re.search('_(......)MHz', file_n5_t).group(1))
            freq_mhz = map_dic_n5['freq']/1e6
            sa_sr = map_dic_n5['px_dic']['sa_sr']
            jy2mKsr = 1e-26*const.c.value**2/2/(1e6*freq_mhz)**2/const.k_B.value*1e3
#             print(i, freq_mhz, 'MHz', end=',')
#             norm_t = self.d_diag[i] / (self.px_sa * self.beam_sa[i]) #* self.syn_beam_sa[i]) #/ self.beam_dilution[i]
            norm_t = self.d_diag[i] * jy2mKsr / self.beam_sa[i]
            if norm:
                p_mat_n5_t = map_dic_n5['p_sum']*norm_t[:, np.newaxis]
                p_mat_n6_t = map_dic_n6['p_sum']*norm_t[:, np.newaxis]
                p_mat_n5_t *= sa_sr.flatten()[np.newaxis, :]
                p_mat_n6_t *= sa_sr.flatten()[np.newaxis, :]
            else:
                p_mat_n5_t = map_dic_n5['p_sum']
                p_mat_n6_t = map_dic_n6['p_sum']
            
            if i == 0:
                p_dic = {'px_dic':map_dic_n5['px_dic']}
                p_mat_n5 = p_mat_n5_t[np.newaxis, ...]
                p_mat_n6 = p_mat_n6_t[np.newaxis, ...]
                freq_mhz_arr = np.array([freq_mhz,])
            else:
                p_mat_n5 = np.concatenate((p_mat_n5, p_mat_n5_t[np.newaxis, ...]), axis=0)
                p_mat_n6 = np.concatenate((p_mat_n6, p_mat_n6_t[np.newaxis, ...]), axis=0)
                freq_mhz_arr = np.append(freq_mhz_arr, freq_mhz)
        
        p_dic['p_mat_pol-5'] = p_mat_n5
        p_dic['p_mat_pol-6'] = p_mat_n6
        p_dic['p_mat_I'] = 0.5 * (p_mat_n5 + p_mat_n6)
        p_dic['freq_mhz'] = freq_mhz_arr
        self.p_dic = p_dic
        
        return p_dic
    
    def cov_mat_calc(self, diag_only=False):
        '''Calculating covarince matrices
        
        Args:
        -----
        diag_only: Bool
            Whether calculating the diagonal elements only, default: False
        
        '''
        for i in range(len(self.files_n5)):
            file_n5_t = self.files_n5[i]
            file_n6_t = self.files_n6[i]

            with open(file_n5_t, 'rb') as f_t:
                map_dic_n5 = pickle.load(f_t)
            with open(file_n6_t, 'rb') as f_t:
                map_dic_n6 = pickle.load(f_t)

#             freq_mhz = float(re.search('_(......)MHz', file_n5_t).group(1))
            freq_mhz = map_dic_n5['freq']/1e6
            print(i, freq_mhz, 'MHz', end=',')
        
            d_diag = self.d_diag[i]
            
            cov_mat_n5_t = map_dic_n5['p_sum']*d_diag[:, np.newaxis]*d_diag[np.newaxis, :]
            cov_mat_n6_t = map_dic_n6['p_sum']*d_diag[:, np.newaxis]*d_diag[np.newaxis, :]
            
            if diag_only:
                cov_mat_n5_t = np.diag(cov_mat_n5_t)
                cov_mat_n6_t = np.diag(cov_mat_n6_t)
            
            if i == 0:
                cov_dic = {'px_dic':map_dic_n5['px_dic']}
                cov_mat_n5 = cov_mat_n5_t[np.newaxis,...]
                cov_mat_n6 = cov_mat_n6_t[np.newaxis,...]
                freq_mhz_arr = np.array([freq_mhz,])
            else:
                cov_mat_n5 = np.concatenate((cov_mat_n5, cov_mat_n5_t[np.newaxis,...]), axis=0)
                cov_mat_n6 = np.concatenate((cov_mat_n6, cov_mat_n6_t[np.newaxis,...]), axis=0)
                freq_mhz_arr = np.append(freq_mhz_arr, freq_mhz)
        
        cov_dic['cov_mat_pol-5'] = cov_mat_n5
        cov_dic['cov_mat_pol-6'] = cov_mat_n6
        cov_dic['cov_mat_I'] = 0.5**2 * (cov_mat_n5 + cov_mat_n6)
        cov_dic['freq_mhz'] = freq_mhz_arr
        
        return cov_dic
        
