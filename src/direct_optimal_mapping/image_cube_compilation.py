import numpy as np
import pickle
from astropy import constants as const
import re

class ImgCube:
    '''Converting individual maps into an image cube,
    also convert the image cube into mK units
    '''
    def __init__(self, files_n5, files_n6, syn_sa_file):
        '''Initialize the object
        '''
        self.files_n5 = files_n5
        self.files_n6 = files_n6
        with open(syn_sa_file, 'rb') as f_t:
            self.syn_sa_dic = pickle.load(f_t)
        return
        
    def image_cube_calc(self):
        '''Image cube compilation
        '''
        assert len(self.files_n5) == len(self.files_n6), 'Pol-5 and Pol-6 do not have same number of maps.'
        assert len(self.files_n5) == len(self.syn_sa_dic['sa']), 'File number and syn solid angle number do not match.'
        for i in range(len(self.files_n5)):
            file_n5_t = self.files_n5[i]
            file_n6_t = self.files_n6[i]

            with open(file_n5_t, 'rb') as f_t:
                map_dic_n5 = pickle.load(f_t)
            with open(file_n6_t, 'rb') as f_t:
                map_dic_n6 = pickle.load(f_t)

            freq_mhz = float(re.search('_(......)MHz', file_n5_t).group(1))
            print(i, freq_mhz, 'MHz', end=',')
            
            # normalization d calculation
            d_diag = 1/(map_dic_n5['beam_weight_sum'] * map_dic_n5['px_dic']['sa_sr'].flatten()) # vis -> Jy/beam
            d_diag = d_diag/self.syn_sa_dic['sa'][i] # Jy/beam -> Jy/sr
#             d_diag = d_diag/map_dic_n5['px_dic']['sa_sr'].flatten() # Jy/beam -> Jy/sr
#             print(self.syn_sa_dic['sa'][i], map_dic_n5['px_dic']['sa_sr'])
            jysr2mk = 1e-26*const.c**2/2/(1e6*freq_mhz)**2/const.k_B*1e3
            d_diag = d_diag * jysr2mk # Jy/sr -> mK

            map_n5_t = map_dic_n5['map_sum'].squeeze() * d_diag
            map_n6_t = map_dic_n6['map_sum'].squeeze() * d_diag
            
            if i == 0:
                data_dic = {'px_dic':map_dic_n5['px_dic']}
                img_cube_n5 = map_n5_t
                img_cube_n6 = map_n6_t
                freq_mhz_arr = np.array([freq_mhz,])
                self.d_diag = d_diag
            else:
                img_cube_n5 = np.vstack((img_cube_n5, map_n5_t))
                img_cube_n6 = np.vstack((img_cube_n6, map_n5_t))
                freq_mhz_arr = np.append(freq_mhz_arr, freq_mhz)
                self.d_diag = np.vstack((self.d_diag, d_diag))
        img_cube_n5 = img_cube_n5.squeeze().reshape(((-1, *map_dic_n5['px_dic']['ra_deg'].shape)))
        img_cube_n6 = img_cube_n6.squeeze().reshape(((-1, *map_dic_n6['px_dic']['ra_deg'].shape)))
        img_cube_n5 = np.moveaxis(img_cube_n5, 0, -1)
        img_cube_n6 = np.moveaxis(img_cube_n6, 0, -1)

        data_dic['data_cube_pol-5'] = img_cube_n5
        data_dic['data_cube_pol-6'] = img_cube_n6
        data_dic['data_cube_I'] = 0.5*(img_cube_n5 + img_cube_n6)
        data_dic['freq_mhz'] = freq_mhz_arr
        
        return data_dic
    
    def p_mat_calc(self):
        '''Calculating p matrices
        
        Parameters
        ----------

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

            freq_mhz = float(re.search('_(......)MHz', file_n5_t).group(1))
            print(i, freq_mhz, 'MHz', end=',')
            
            d_diag = self.d_diag[i]
            
            p_mat_n5_t = map_dic_n5['p_sum']*d_diag[:, np.newaxis]
            p_mat_n6_t = map_dic_n6['p_sum']*d_diag[:, np.newaxis]
            
            if i == 0:
                p_dic = {'px_dic':map_dic_n5['px_dic']}
                p_mat_n5 = p_mat_n5_t[np.newaxis,...]
                p_mat_n6 = p_mat_n6_t[np.newaxis,...]
                freq_mhz_arr = np.array([freq_mhz,])
            else:
                p_mat_n5 = np.concatenate((p_mat_n5, p_mat_n5_t[np.newaxis,...]), axis=0)
                p_mat_n6 = np.concatenate((p_mat_n6, p_mat_n6_t[np.newaxis,...]), axis=0)
                freq_mhz_arr = np.append(freq_mhz_arr, freq_mhz)
        
        p_dic['p_mat_pol-5'] = p_mat_n5
        p_dic['p_mat_pol-6'] = p_mat_n6
        p_dic['p_mat_I'] = 0.5**2 * (p_mat_n5 + p_mat_n6)
        p_dic['freq_mhz'] = freq_mhz_arr
        
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

            freq_mhz = float(re.search('_(......)MHz', file_n5_t).group(1))
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
        
