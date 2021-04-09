import numpy as np
from pyuvdata import UVData

class DataConditioning:
    '''This class takes in the pyuvdata UVData object and perform
    conditioning before the optimal mapping procedure, including
    redundant baseline grouping, redundant baseline averaging,
    frequency-domain flagging, time-domain flagging, antenna
    selection, etc.
    '''

    def __init__(self, uv, ifreq, ipol):
        '''Setting up initial parameters for the object
        
        Input
        ------
        uv: UVData Object

        ifreq: integer [0 - 1023]
            select the frequency band
        ipol: integer [-5 - -8]
            select the linear polarization (-5:-8 (XX, YY, XY, YX))           

        '''
        uv_1d = uv.select(freq_chans=ifreq, polarizations=ipol, 
                          inplace=False, keep_all_metadata=False)
        self.uv_1d = uv_1d

    def rm_flag(self, uv):
        '''Remove flagged data visibilities, keeping the original
        data object untouched
        
        Input
        ------
        uv: UVData object
            input UVData object to be flag-removed
            The UVData.data_array must only have one dimension along blt
            
        Output
        ------
        uv_flag_rm: UVData object
            flag-removed UVData object
        '''
        if all(uv.flag_array):
            print('All data are flagged. Returning None.')
            return None
        idx_t = np.where(uv.flag_array==False)[0]
        uv_flag_rm = uv.select(blt_inds=idx_t, 
                               inplace=False, keep_all_metadata=False)
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
