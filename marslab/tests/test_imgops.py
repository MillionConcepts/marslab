from marslab import imgops
import numpy as np

class TestApplyImageFilter:

    def test_apply_image_filter_1(self):
        out = imgops.apply_image_filter([[10, 10]], image_filter={'function': np.mean, 'params': {'axis': 0}})
        assert out[0]==10
        assert out[1]==10

    def test_apply_image_filter_2(self):
        out = imgops.apply_image_filter([[10, 10]])
        assert out[0][0]==10
        assert out[0][1]==10

class TestBorderCrop:

    def test_border_crop_1(self):
        img = np.ones((10,10))
        out = imgops.border_crop(img,crop=(1,1,1,1))
        assert out.shape[0]==8
        assert out.shape[1]==8

