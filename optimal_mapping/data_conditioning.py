import numpy as np
from pyuvdata import UVData
import copy

class DataConditioning:
    '''This class takes in the pyuvdata UVData object and perform
    conditioning before the optimal mapping procedure, including
    redundant baseline grouping, redundant baseline averaging,
    frequency-domain flagging, time-domain flagging, antenna
    selection, etc.
    '''

    def __init__(self, uv, ifreq, ipol):
        '''Setting up initial parameters for the object
        
        Args
        ------
        uv: pyuvdata object
            main pyuvdata object
        ifreq: integer [0 - 1023]
            select the frequency band
        ipol: integer [-5 - -8]
            select the linear polarization (-5:-8 (XX, YY, XY, YX))
        '''
        self.ifreq = ifreq
        self.ipol = ipol
        uv_cp = copy.deepcopy(uv)
        uv_cross = uv_cp.select(ant_str='cross', inplace=False)
        self.uv_1d = uv_cross.select(freq_chans=ifreq, polarizations=ipol, 
                                     inplace=False, keep_all_metadata=False)
        uv_auto = uv_cp.select(ant_str='auto', inplace=False)
        self.uv_auto = uv_auto.select(freq_chans=ifreq, polarizations=ipol,
                                      inplace=False, keep_all_metadata=False)   

    def rm_flag(self, uv=None):
        '''Remove flagged data visibilities, keeping the original
        data object untouched
        
        Args
        ------
        uv: UVData object
            input UVData object to be flag-removed
            The UVData.data_array must only have one dimension along blt
            If None, self.uv_1d is used by default
        Output
        ------
        uv_flag_rm: UVData object
            flag-removed UVData object
        '''
        if uv is None:
            uv = self.uv_1d
        if np.all(uv.flag_array):
            print('All data are flagged. Returning None.')
            return None
        idx_t = np.where(uv.flag_array==False)[0]
        uv_flag_rm = uv.select(blt_inds=idx_t, 
                               inplace=False, keep_all_metadata=False)
        if hasattr(self.uv_1d, 'noise_array'):
            uv_flag_rm.noise_array = self.uv_1d.noise_array[idx_t]
        
        return uv_flag_rm
        
    def redundant_avg(self, uv, tol=1.0):
        '''Averaging within each redundant group, keeping the original
        data object untouched
        
        Input
        ------
        uv: UVData object
            input UVData object to be flag-removed
            The UVData.data_array must only have one dimension along blt
            
        tol: float
            tolerace of the redundant baseline grouping (in meters)
        
        Output
        ------
        uv_red_avg: UVData object
            redundant-baseline averaged UVData object
        
        '''
        # redundant baseline averaging
        uv_red_avg = uv.compress_by_redundancy(tol=tol,
                                               method='average',
                                               inplace=False,
                                               keep_all_metadata=False,)
        return uv_red_avg
        
    def redundant_grouping(self, tol=1.0):
        '''Grouping the baselines within the UVData

        Input
        ------
        tol: float
            tolerace of the redundant baseline grouping (in meters)

        Output
        ------
        bl_groups: list of lists (int)
            listed redundant baseline numbers

        uvw_groups: list of 2d np.array (float)
            center value of each redundant baseline group
            
        uv_red_avg: UVData obj
            averaged UVData
            
        Attributes
        ------
        .bl_groups: list of lists (int)
            listed redundant baseline numbers

        .uvw_groups: list of 2d np.array (float)
            center value of each redundant baseline group
            
        .uv_red_avg: UVData obj
            averaged UVData
        '''
        # redundant baseline grouping
        bl_groups, uvw_groups, _ = self.uv.get_redundancies(tol=self.tol, 
                                                            use_antpos=False, 
                                                            include_conjugates=False, 
                                                            include_autos=False)
        # redundant baseline averaging
        uv_red_avg = self.uv.compress_by_redundancy(tol=self.tol,
                                                    method='average',
                                                    inplace=False,
                                                    keep_all_metadata=False,)
        # Attribute assignment
        self.bl_groups = bl_groups
        self.uvw_groups = uvw_groups
        self.uv_red_avg = uv_red_avg
        
        return bl_groups, uvw_groups, uv_red_avg
