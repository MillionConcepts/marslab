from marslab.compat import xcam

def test_derived_cam_dict_construction():
    derived = xcam.DERIVED_CAM_DICT
    assert max(list(derived['ZCAM']['filters'].values())) == 1022
    assert derived['MCAM']['virtual_filter_mapping']['L2_R2'] == ('L2', 'R2')
    assert min(list(derived['MCAM']['filters'].values())) == 445
    assert 'L0G_R0G' in derived['ZCAM']['virtual_filters']
    assert 'L0R_R0R' in derived['MCAM']['virtual_filters']


