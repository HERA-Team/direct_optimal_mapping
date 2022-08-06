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

            map_n5_t = map_dic_n5['map_sum'].squeeze()/map_dic_n5['beam_sq_weight_sum']/map_dic_n5['px_dic']['sa_sr'].flatten() #Jy/beam
            map_n6_t = map_dic_n6['map_sum'].squeeze()/map_dic_n6['beam_sq_weight_sum']/map_dic_n6['px_dic']['sa_sr'].flatten() #Jy/beam
            
            # Jy/beam -> Jy/sr
            map_n5_t = map_n5_t/self.syn_sa_dic['sa'][i]
            map_n6_t = map_n6_t/self.syn_sa_dic['sa'][i]

            # Jy/sr -> mK
            jysr2mk = 1e-26*const.c**2/2/(1e6*freq_mhz)**2/const.k_B*1e3
            map_n5_t = map_n5_t * jysr2mk
            map_n6_t = map_n6_t * jysr2mk
            if i == 0:
                data_dic = {'px_dic':map_dic_n5['px_dic']
                    }
                img_cube_n5 = map_n5_t
                img_cube_n6 = map_n6_t
                freq_mhz_arr = np.array([freq_mhz,])
            else:
                img_cube_n5 = np.vstack((img_cube_n5, map_n5_t))
                img_cube_n6 = np.vstack((img_cube_n6, map_n5_t))
                freq_mhz_arr = np.append(freq_mhz_arr, freq_mhz)
        img_cube_n5 = img_cube_n5.squeeze().reshape(((-1, *map_dic_n5['px_dic']['ra_deg'].shape)))
        img_cube_n6 = img_cube_n6.squeeze().reshape(((-1, *map_dic_n6['px_dic']['ra_deg'].shape)))
        img_cube_n5 = np.moveaxis(img_cube_n5, 0, -1)
        img_cube_n6 = np.moveaxis(img_cube_n6, 0, -1)

        data_dic['data_cube_pol-5'] = img_cube_n5
        data_dic['data_cube_pol-6'] = img_cube_n6
        data_dic['freq_mhz'] = freq_mhz_arr
        
        return data_dic
