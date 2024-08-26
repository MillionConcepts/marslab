import warnings
from itertools import product

import numpy as np

from marslab.imgops import imgutils
from marslab.tests.utilz.div0 import divide_by_zero

RNG = np.random.default_rng()


class TestApplyImageFilter:
    def test_apply_image_filter_1(self):
        out = imgutils.apply_image_filter(
            [[10, 10]],
            image_filter={"function": np.mean, "params": {"axis": 0}},
        )
        assert out[0] == 10
        assert out[1] == 10

    def test_apply_image_filter_2(self):
        out = imgutils.apply_image_filter([[10, 10]])
        assert out[0][0] == 10
        assert out[0][1] == 10


class TestCrop:
    def test_crop_1(self):
        img = np.ones((10, 10))
        out = imgutils.crop(img, bounds=(1, 1, 1, 1))
        assert out.shape[0] == 8
        assert out.shape[1] == 8


class TestEightbit:
    def test_eightbit(self):
        img = np.ones((10, 10))
        img[2] = 0
        out = imgutils.eightbit(img)
        assert out.shape[0] == 10
        assert out.shape[1] == 10
        assert out.dtype == np.uint8
        assert np.all(out[0] == 255)
        assert np.all(out[2] == 0)

    def test_eightbit_dtypes(self):
        """
        Test function for eightbit() dtype handling.
        """
        with warnings.catch_warnings():
            divide_by_zero()
            randarrays = tuple(RNG.random((128, 128)) for _ in range(5))
            dtypes = [*np.typecodes["Float"], *np.typecodes["AllInteger"]]
            for arr, dtype in product(randarrays, dtypes):
                eight = imgutils.eightbit(arr.astype(dtype))
                assert np.isin(eight.min(), (0, 255))
                assert np.isin(eight.max(), (0, 255))
                assert eight.dtype == np.uint8
                assert eight.min() <= eight.max()


def test_normalize_range():
    img = np.ones((10, 10)).astype(np.uint8)
    img[2] = 0
    out = imgutils.normalize_range(img, bounds=(-1, 1))
    assert out[0, 0] == 1
    assert out[2, 0] == -1
    assert out.dtype == np.int16
    fimg = np.random.random((10, 10)).astype(np.float16)
    out32 = imgutils.normalize_range(fimg, bounds=(0, 255))
    assert out32.dtype == np.float32
    assert (out32.min() == 0) and (abs(out32.max() - 255) < 0.01)
    out16 = imgutils.normalize_range(
        fimg, bounds=(0, 255), allow_half_float=True
    )
    assert out16.dtype == np.float16
    assert out16.min() == 0 and out16.max() == 255


def test_normalize_range_masked():
    img = np.ma.masked_array(np.full((10, 10), 2), mask=np.zeros((10, 10)))
    img[2] = 0
    img[3] = 1
    img[5] = 3
    img.mask[5] = True
    out = imgutils.normalize_range(img.astype(np.float32))
    assert out[0, 0] == 1
    assert out[2, 2] == 0
    assert out[3, 3] == 0.5
    assert out.dtype == np.float32


def test_std_clip():
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    clipped = imgutils.std_clip(arr, 1)
    assert clipped.min() == 2
    assert clipped.max() == 8
    assert clipped.mean() == 5.3
    assert abs(clipped.std() - 2.3259) < 0.01
    assert clipped[4] == arr[4]


def test_cropmask():
    mask = np.ones((10, 10))
    out = imgutils.cropmask(mask, bounds=(1, 1, 1, 1))
    assert out.shape == (10, 10)
    assert np.ma.is_masked(out[0])


def test_crop_all():
    img = np.ones((10, 10))
    imgs = [img.copy() for _ in range(5)]
    out = imgutils.crop_all(imgs, bounds=(1, 1, 1, 1))
    assert out[0].shape[0] == 8
    assert out[1].shape[1] == 8
    assert out[2].shape[0] == 8
    assert out[3].shape[1] == 8
    assert np.all(out[4] == 1)
    assert np.all(out[0] == 1)


def test_clip_unmasked():
    img = np.ma.masked_array(np.ones((10, 10)), mask=np.zeros((10, 10)))
    img[2] = 0
    img[5] = 3
    img.mask[5] = True
    out = imgutils.clip_unmasked(img)
    assert out.min() == 0 and out.max() == 1


def test_bilinear_interpolate_subgrid():
    """
    test function for bilinear_interpolate_subgrid
    """
    test_input = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=np.float32,
    )
    rows, columns = np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])
    result = imgutils.bilinear_interpolate_subgrid(
        rows, columns, test_input, (8, 8)
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            [2.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            [4.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
            [6.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
            [8.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0],
            [10.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
            [12.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0],
            [12.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)


def test_strict_reshape():
    """
    test function for strict_reshape
    """
    test_input = np.zeros((4, 4))
    test_output = imgutils.strict_reshape(test_input, 1)
    assert test_output.shape == (4, 4)
    test_output = imgutils.strict_reshape(test_input, 3)
    assert test_output.shape == (2, 8)
    test_output = imgutils.strict_reshape(test_input, 0.5)
    assert test_output.shape == (8, 2)
