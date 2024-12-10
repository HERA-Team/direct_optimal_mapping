import numpy as np
from pyuvdata import UVData
from direct_optimal_mapping import optimal_mapping
from direct_optimal_mapping import data_conditioning
import copy
import sys

class HorizonMap:
    '''This class takes the optimal_mapping data conditioning object
       and computes the normalized map in horizon coordinates, using 
       a single-time-stamp matrix and rotations to speed up the calculation.
       The maps returned in the map dictionary are the normalized sky map 
       (unmap divided by the beam weights map), and the beam weight maps 
       if requested.  There is also the option to return the P-matrix.
       Warning:  P-matrix computation is very resource-intensive.
       The pmatrix_factor is not yet implemented, but it should be -
       pmatrix_factor=1 is not large enough for most applications.
    '''

    def __init__(self, dc, ra_ctr_deg, ra_rng_deg, dec_ctr_deg, dec_rng_deg,
                 wts='optimal', norm='one-beam', epoch_map='J2000', uvw_sign=1,
                 buffer=True, return_b1map=False, return_b2map=False,
                 return_pmatrix=False, pmatrix_factor=1):

        '''
        Parameters
        ----------
        dc: data conditioning object (one dimensional: one frequency, one polarization)
        ra_ctr_deg:  RA coordinate of the center of the map in degrees
        dec_ctr_deg:  Dec coordinate of the center of the map in degrees
        ra_rng_deg:  Width of map in coordinate degrees
        dec_rng_deg:  Height of map in degrees
        wts:  weighting given to visibility points
        norm:  normalization computed and applied to the map
        epoch_map:  epoch of the map coordinate system
        uvw_sign:  same convention as optimal mapping object
        buffer: add a buffer for weighting edge effects and numpy rolls, then trim it
        return_b1map:  include beam weights map in map dictionary
        return_b2map:  include beam squared weights map  in map dictionary
        return_pmatrix: include the PSF matrix in map dictionary
        pmatrix_factor: the linear scale factor by which the [P] sky dimension is larger than the map sky dimension

        Note: ra and dec ranges are specified in COORDINATE degrees
        Example: for a 10-degree ARC on the sky at dec=30deg, specify 10deg/0.8666
        East-West cell size will be one time integration
        North-South cell size will be set to equal the East-West cell size
        '''

        self.dc = dc
        self.ra_ctr_deg = ra_ctr_deg
        self.dec_ctr_deg = dec_ctr_deg
        self.ra_rng_deg = ra_rng_deg
        self.dec_rng_deg = dec_rng_deg
        self.wts = wts
        self.norm = norm
        self.epoch_map = epoch_map
        self.uvw_sign = uvw_sign 
        self.buffer = buffer
        self.return_b1map = return_b1map
        self.return_b2map = return_b2map
        self.return_pmatrix = return_pmatrix
        self.pmatrix_factor = pmatrix_factor

        if wts != 'optimal':
            print('Only optimal weighting is implemented so far')
        if (norm != 'one-beam') and (norm != 'two-beam'):
            print('Only one-beam and two-beam normalizations are implemented so far')
        if return_pmatrix:
            print('Warning: P-matrix computation is very preliminary')
        if pmatrix_factor!=1: 
            print('Only pmatrix_factor = 1 is implemented so far')

        return
    
    def calc_map(self):
        #
        # set up pixels
        # if buffer==True, a buffer allowing for the roll and edge 
        # effects is put in, then later removed
        #

        if self.buffer==True:
            print('Buffering the edges of the map')
        else:
            print('Not buffering the edge of the map.  Beware edge effects.')

        nra=self.ra_rng_deg*3600./(self.dc.uv_1d.integration_time[0]*15.)
        nra=int(nra)
        ndec=self.dec_rng_deg*3600./(self.dc.uv_1d.integration_time[0]*15.*np.cos(self.dec_ctr_deg*np.pi/180.))
        ndec=int(ndec)
        print('map coordinate grid is: (center,range,number)')
        print('ra: ',self.ra_ctr_deg,self.ra_rng_deg,nra,' dec: ',self.dec_ctr_deg,self.dec_rng_deg,ndec)
        temp=self.ra_rng_deg*np.cos(self.dec_ctr_deg*np.pi/180.)/nra
        print('cell sizes in arc deg based on int time are ',temp,self.dec_rng_deg/ndec,)
        #
        # compute buffer 
        # the buffer has to be an even number of pixels, or else it cannot be 
        # accommodated by the logic of the pixel object
        # The buffer needs to as long as the number of time stamps minus 1.
        #
        nbuffer=0
        buffer_deg=0.
        delra_deg=0.
        if self.buffer==True:
            nbuffer = int(len(np.unique(self.dc.uv_1d.time_array)) - 1)
            if (nbuffer%2) != 0: nbuffer=nbuffer+1
            buffer_deg=nbuffer*self.dc.uv_1d.integration_time[0]*15./3600.  # this is in coordinate degrees
            print('buffer is ',nbuffer,' pixels = ',buffer_deg,' coordinate degrees')
            print('Including buffer, nra and ndec are ',nra+nbuffer,ndec)
        #
        # set up map pixels
        #
        pixels = optimal_mapping.SkyPx()  
        pixel_dict=pixels.calc_radec_pix(self.ra_ctr_deg,self.ra_rng_deg+buffer_deg,nra+nbuffer,self.dec_ctr_deg,self.dec_rng_deg,ndec)
        #
        # compute noise matrix (assumed diagonal)
        #
        self.dc.noise_calc()
        #
        # check whether noise was calculated (noise array is 1D)
        #
        if (np.sum(self.dc.uvn.data_array)==len(self.dc.uvn.data_array)):
            print('Unable to calculate noise:  sigma-n has been set to ones')
        #
        # set reference LST at middle of time span
        #
        ulsts=np.unique(self.dc.uv_1d.lst_array)
        ref_lst_index=int(len(ulsts)/2)
        self.ref_lst_deg=ulsts[ref_lst_index]*180./np.pi
        print('Reference LST index is ',ref_lst_index)
        print('Reference LST is ',self.ref_lst_deg,' degrees')
        uvref=self.dc.uv_1d.select(lst_range=[ulsts[ref_lst_index],ulsts[ref_lst_index]],inplace=False)
        dcref = data_conditioning.DataConditioning(uvref,0,self.dc.ipol)
        del uvref
        print('Reference data array shape ',dcref.uv_1d.data_array.shape)
        #
        # create map object and compute A-matrix components and weights
        #
        print('Creating map object')
        opt_map = optimal_mapping.OptMapping(dcref.uv_1d,pixel_dict,px_dic_inner=pixel_dict,epoch=self.epoch_map)
        print('map object created')
        opt_map.set_phase_mat(uvw_sign=self.uvw_sign)
        print('Phase matrix shape is ',opt_map.phase_mat.shape)
        opt_map.set_beam_mat(uvw_sign=1)
        print('Beam matrix shape is ',opt_map.beam_mat.shape)
        # compute weights
        if self.wts=='optimal':
            opt_map.set_inv_noise_mat(self.dc.uvn,matrix=False)
            print('inv noise shape ',opt_map.inv_noise_mat.shape)
            idx=np.where(np.vectorize(np.isnan)(opt_map.inv_noise_mat)==True)[0]
            if len(idx) > 0:
                print('Found ',len(idx),' nans in nv noise array; setting to zero')
                opt_map.inv_noise_mat[idx]=0.
            idx=np.where(np.vectorize(np.isinf)(opt_map.inv_noise_mat)==True)[0]
            if len(idx) > 0:
                print('Found ',len(idx),' infs in nv noise array; setting to zero')
                opt_map.inv_noise_mat[idx]=0.
        #
        # check and sanitize the vis data 
        #
        idx=np.where(np.vectorize(np.isnan)(self.dc.uv_1d.data_array)==True)[0]
        if len(idx) > 0:
            print('Found ',len(idx),' nans in visibility array; setting weight to zero')
            opt_map.inv_noise_mat[idx]=0.
        idx=np.where(np.vectorize(np.isinf)(self.dc.uv_1d.data_array)==True)[0]
        if len(idx) > 0:
            print('Found ',len(idx),' infs in visibility array; setting weight to zero')
            opt_map.inv_noise_mat[idx]=0.
        idx=np.where(self.dc.uv_1d.flag_array==True)[0]
        if len(idx) > 0:
            print('Found ',len(idx),' flags in visibility array; setting weight to zero')
            opt_map.inv_noise_mat[idx]=0.
        #
        # compute maps (do we have sufficient precision?)
        #
        self.unmap=np.zeros((nra+nbuffer,ndec))
        if self.return_b1map or self.norm=='one-beam': self.b1map=np.zeros((nra+nbuffer,ndec))
        if self.return_b2map or self.norm=='two-beam': self.b2map=np.zeros((nra+nbuffer,ndec))
        if self.return_pmatrix: 
            psize=(nra+nbuffer)*ndec*self.pmatrix_factor**2
            self.pmatrix=np.zeros((psize,psize))
            print('Warning: pmatrix is big.  Its size is ',psize,' X ',psize)
        amatrix=opt_map.phase_mat*opt_map.beam_mat
        for itime, time_stamp in enumerate(np.unique(self.dc.uv_1d.time_array)):
            idx_roll = itime - ref_lst_index
            print('Computing map for timestamp ',itime, time_stamp,' - roll is ',idx_roll)
            idx_t=np.where(self.dc.uv_1d.time_array==time_stamp)[0]
            vis=self.dc.uv_1d.data_array[idx_t,0,0]
            wts=opt_map.inv_noise_mat[idx_t]
            if (np.sum(self.dc.uv_1d.flag_array[idx_t])==vis.shape[0]):
                print('All data are flagged for this time stamp - skip it')
            else:  
                # there is good data, so compute time-stamp map and add to accum array
                unmapt=np.real(np.matmul(np.conjugate(amatrix).T,np.multiply(wts,vis)))
                self.unmap=self.unmap+np.roll(unmapt.reshape(nra+nbuffer,ndec),idx_roll,axis=0)
                if self.return_b1map or self.norm=='one-beam':
                    b1mapt=np.real(np.matmul(np.conjugate(opt_map.beam_mat).T,np.multiply(wts,vis*0.+1.)))
                    self.b1map=self.b1map+np.roll(b1mapt.reshape(nra+nbuffer,ndec),idx_roll,axis=0)
                if self.return_b2map or self.norm=='two-beam':
                    b2mapt=np.real(np.matmul(np.conjugate(opt_map.beam_mat**2).T,np.multiply(wts,vis*0.+1.)))
                    self.b2map=self.b2map+np.roll(b2mapt.reshape(nra+nbuffer,ndec),idx_roll,axis=0)
                if self.return_pmatrix:  # this can probably be sped up with diagonal mult
                    pmt = np.real(np.matmul(np.conjugate(amatrix).T,np.matmul(np.diag(wts),amatrix)))
                    temp = pmt.reshape((nra+nbuffer)*ndec,(nra+nbuffer)*self.pmatrix_factor*ndec*self.pmatrix_factor)
                    # this is left-multiply by the rotation matrix transpose - just like the maps
                    temp = np.roll(temp,idx_roll,axis=0)
                    temp = np.roll(temp,idx_roll,axis=1)  # negative idx_roll here smears, which surprises me
                    self.pmatrix=self.pmatrix+temp
                    del temp
        #
        # Remove buffer
        #
        if self.buffer==True:
            i1=int(nbuffer/2)
            i2=int(nra+nbuffer/2)
            self.unmap=self.unmap.reshape(nra+nbuffer,ndec)[i1:i2,:]
            if self.return_b1map or self.norm=='one-beam': self.b1map=self.b1map.reshape(nra+nbuffer,ndec)[i1:i2,:]
            if self.return_b2map or self.norm=='two-beam': self.b2map=self.b2map.reshape(nra+nbuffer,ndec)[i1:i2,:]
            # still need to remove buffer on pmatrix 
        #
        # Create pixel dictionary of the map with the buffer removed
        #
        del pixels
        pixels = optimal_mapping.SkyPx()
        pixel_dict=pixels.calc_radec_pix(self.ra_ctr_deg,self.ra_rng_deg,nra,self.dec_ctr_deg,self.dec_rng_deg,ndec)
        #
        # check that the map and the pixel dictionary are consistent
        #
        if self.unmap.shape != (nra,ndec):
            print('unmap has shape ',self.unmap.shape)
            print('nra,ndec,nra*dec are ',nra,ndec,nra*ndec)
            print('nra in dictionary is ',len(pixel_dict['ra_deg']))
            print('ndec in dictionary is ',len(pixel_dict['dec_deg']))
            sys.exit()
        #
        # pack everything into the dictionary.  Don't yet have P
        #
        mapdict={'px_dic':pixel_dict}
        if self.norm=='one-beam': self.unmap=self.unmap/self.b1map
        if self.norm=='two-beam': self.unmap=self.unmap/self.b2map
        mapdict['map_sum']=self.unmap
        if self.return_b1map: mapdict['beam_weight_sum']=self.b1map
        if self.return_b2map: mapdict['beam_sq_weight_sum']=self.b2map
        if self.return_pmatrix: mapdict['pmatrix']=self.pmatrix  # key needs to be made consistent with pspec?
        mapdict['n_vis']=self.dc.uv_1d.data_array.shape[0]  # Zhilei's def.  Does not include nsamples
        mapdict['freq']=self.dc.uv_1d.freq_array[0] # in Hz
        mapdict['polarization']=self.dc.ipol
        mapdict['bl_max']=np.sqrt(np.sum(self.dc.uv_1d.uvw_array**2, axis=1)).max()
        mapdict['radius2ctr']=0.  # distance of pixels to center.  Do we really need this?

        return mapdict
