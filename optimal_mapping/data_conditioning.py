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

    def __init__(self, uv, ifreq, ipol, uv_noise=None):
        '''Setting up initial parameters for the object
        
        Args
        ------
        uv: pyuvdata object
            main pyuvdata object
        ifreq: integer [0 - 1023]
            select the frequency band
        ipol: integer [-5 - -8]
            select the linear polarization (-5:-8 (XX, YY, XY, YX))
        uv_noise: pyuvdata object
            noise information is stored in the pyuvdata format (std. dev of the full
            complex visibility), .nsample_array in the uv object is assigned with the 
            noise info in the .data_array attribute
            if uv_noise is None, the .nsample_array is assigned to be one as uniform
            noise across visibilities
        '''
        self.ifreq = ifreq
        self.ipol = ipol
        uv_cp = copy.deepcopy(uv)
        if uv_noise is None:
            uv_cp.nsample_array = np.ones(uv_cp.nsample_array.shape)
        else:
            uv_cp.nsample_array[:,:,:,0] = np.real(uv_noise.data_array[:, :, :, 0])
            uv_cp.nsample_array[:,:,:,3] = np.real(uv_noise.data_array[:, :, :, 1])
        uv_1d = uv_cp.select(freq_chans=ifreq, polarizations=ipol, 
                             inplace=False, keep_all_metadata=False)
        self.uv_1d = uv_1d
        
    def add_noise(self, uv_noise):
        '''Extract the noise value from the noise file and report
        them at given frequqncy channel and polarization, the noise
        value is added into the uv_1d object under an attribute 
        .noise_array
        
        Args
        ------
        uv_noise: UVData Object
        
        '''
        uv_noise_1d = uv_noise.select(freq_chans=self.ifreq, 
                                      polarizations=self.ipol, 
                                      inplace=False, 
                                      keep_all_metadata=False)
        self.uv_1d.noise_array = uv_noise_1d.data_array.astype(np.float32)
        

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
        if all(uv.flag_array):
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
