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
        uv_cross = uv.select(ant_str='cross', inplace=False)
        self.uv_1d = uv_cross.select(freq_chans=ifreq, polarizations=ipol, 
                                     inplace=False, keep_all_metadata=False)
            
        uv_auto = uv.select(ant_str='auto', inplace=False)
        self.uv_auto = uv_auto.select(freq_chans=ifreq, polarizations=ipol,
                                      inplace=False, keep_all_metadata=False)
        self.log = ['Init., freq. and pol. selected.',]

    def bl_selection(self, ew_proj=14.):
        '''Selecting the baselines with ew projection greater than a certain value.
        This selection is necessary because the cross-talk suppression does not work
        for those baselines. Details can be found in Kern et al. 2019, 2020
        
        Parameters
        ----------
        ew_proj: float
            baseline selection criteria, regarding to the projected EW distance.
            Default: 14; Unit: meter
        
        Return
        ------
        None
        '''
        idx_sel = np.where(np.abs(self.uv_1d.uvw_array[:, 0]) > ew_proj)[0]
        if len(idx_sel) == 0:
            return None
        self.uv_1d.select(blt_inds=idx_sel, inplace=True, keep_all_metadata=False)
        self.log.append('bl selected.')
        if 'Noise calculated.' in self.log:
            raise RuntimeError('bl selection should happen before noise calculation.')
        return
        
    def noise_calc(self):
        '''Calculating noise from the autocorrelations
        '''
        uvn = copy.deepcopy(self.uv_1d)
        for bl in uvn.get_antpairs():
            radiometer = np.sqrt(uvn.channel_width * uvn.get_nsamples(bl, squeeze='none') * np.mean(uvn.integration_time))
            # get indices of this baseline
            inds = uvn.antpair2ind(bl)
            # insert uv_ready for this cross-corr
            uvn.data_array[inds] = np.sqrt(self.uv_auto.get_data((bl[0], bl[0]), squeeze='none').real * \
                                           self.uv_auto.get_data((bl[1], bl[1]), squeeze='none').real) / radiometer
            # OR all flags
            neg_auto_flag = (self.uv_auto.get_data((bl[0], bl[0]), squeeze='none') < 0) +\
                            (self.uv_auto.get_data((bl[1], bl[1]), squeeze='none') < 0)
            auto_flag = self.uv_auto.get_flags((bl[0], bl[0]), squeeze='none') +\
                        self.uv_auto.get_flags((bl[1], bl[1]), squeeze='none')
            uvn.flag_array[inds] += auto_flag + neg_auto_flag
        self.uvn = uvn
        self.log.append('Noise calculated.')
        return uvn
    
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
        idx_t = np.where((self.uv_1d.flag_array==False) \
                         & (self.uvn.flag_array==False) \
                         & (self.uv_1d.nsample_array!=0))[0]
        if len(idx_t) == 0:
            #print('All data are flagged. Returning None.')
            return None
        self.uv_1d = uv.select(blt_inds=idx_t, 
                               inplace=False, keep_all_metadata=False)
        self.uvn = self.uvn.select(blt_inds=idx_t, 
                                   inplace=False, keep_all_metadata=False)
        idx_t = np.where(self.uv_auto.flag_array==False)[0]
        self.uv_auto = self.uv_auto.select(blt_inds=idx_t, 
                                           inplace=False, keep_all_metadata=False)
        self.log.append('Flag removed.')
        return self.uv_1d
        
    def redundant_avg(self, tol=1.0):
        '''Averaging within each redundant group, keeping the original
        data object untouched, updating the uv_1d and uvn attribute with
        the redundant averaged ones
        
        Args:
        ------            
        tol: float
            tolerace of the redundant baseline grouping (in meters)
        
        Return:
        ------
        None
        
        '''
        # redundant baseline averaging
        self.uv_1d.compress_by_redundancy(tol=tol,
                                          method='average',
                                          inplace=True,
                                          keep_all_metadata=False,)
        # noise redundant averaging
        bl_grp,_,_ = self.uvn.get_redundancies(tol=tol)
        n_bl = np.array([len(grp_t) for grp_t in bl_grp])
        self.uvn.compress_by_redundancy(tol=tol,
                                        method='average',
                                        inplace=True,
                                        keep_all_metadata=False)
        for time_t in np.unique(self.uvn.time_array):
            idx_t = np.where(self.uvn.time_array == time_t)[0]
            self.uvn.data_array[idx_t, 0, 0, 0] /= np.sqrt(n_bl)
            
        self.log.append('Redundant averaged.')
        return
        
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
