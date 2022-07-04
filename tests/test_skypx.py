import pytest
from optimal_mapping_radec_grid import SkyPx

def test_calc_px():
    sky_px = SkyPx()
    ra_center_deg = 30
    dec_center_deg = -30.7
    ra_rng_deg = 32
    n_ra = 64
    dec_rng_deg = 16
    n_dec = 32
    ra_res_deg = ra_rng_deg/n_ra
    dec_res_deg = dec_rng_deg/n_dec
    px_dic = sky_px.calc_px(ra_center_deg, ra_rng_deg, n_ra, dec_center_deg, dec_rng_deg, n_dec)
    assert px_dic['ra_deg'].shape == (n_ra, n_dec)
    assert px_dic['dec_deg'].shape == (n_ra, n_dec)
    assert px_dic['sa_sr'].shape == (n_ra, n_dec)
    assert px_dic['ra_deg'].min() == ra_center_deg - ra_rng_deg/2. + ra_res_deg/2.
    assert px_dic['ra_deg'].max() == ra_center_deg + ra_rng_deg/2. - ra_res_deg/2.
    assert px_dic['dec_deg'].min() == dec_center_deg - dec_rng_deg/2. + dec_res_deg/2.
    assert px_dic['dec_deg'].max() == dec_center_deg + dec_rng_deg/2. - dec_res_deg/2.