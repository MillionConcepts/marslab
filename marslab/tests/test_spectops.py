from marslab import spectops
import numpy as np
import pandas as pd


class TestSpectopsErrorCalculations:
    def test_errors_in_quadrature_digit(self):
        assert spectops.addition_in_quadrature([2]) == 2

    def test_errors_in_quadrature_tuple(self):
        assert spectops.addition_in_quadrature([2, 2]) == np.sqrt(2 * 2 ** 2)

    def test_errors_in_quadrature_lists(self):
        assert spectops.addition_in_quadrature([[1, 2], [3, 4]])[0] == np.sqrt(
            1 ** 2 + 3 ** 2
        )

    def test_errors_in_quadrature_ndarray(self):
        assert spectops.addition_in_quadrature(np.array([[1, 2], [3, 4]]))[
            0
        ] == np.sqrt(1 ** 2 + 3 ** 2)

    def test_errors_in_quadrature_dataframe(self):
        assert spectops.addition_in_quadrature(pd.DataFrame([[1, 2], [3, 4]]))[
            0
        ] == np.sqrt(1 ** 2 + 3 ** 2)


class TestSpectopsRatio:
    def test_ratio_1_list(self):
        assert spectops.ratio([10, 20])[0] == 0.5

    def test_ratio_1_ndarray(self):
        assert spectops.ratio(np.array([10, 20]))[0] == 0.5

    # def test_ratio_sequence_1_series(self):
    #    assert spectops.ratio(pd.Series([10,20]))[0] == 0.5

    def test_ratio_2_list(self):
        assert spectops.ratio([[5, 20], [10, 40]])[0][0] == 0.5
        assert spectops.ratio([[5, 20], [10, 40]])[0][1] == 0.5

    def test_ratio_2_ndarray(self):
        assert spectops.ratio(np.array([[5, 20], [10, 40]]))[0][0] == 0.5
        assert spectops.ratio(np.array([[5, 20], [10, 40]]))[0][1] == 0.5

    # def test_ratio_sequence_2_dataframe(self):
    #    assert spectops.ratio(pd.DataFrame([[10,20],[30,40]]))[0][0] == 0.5
    #    assert spectops.ratio(pd.DataFrame([[10,20],[30,40]]))[0][1] == 0.75

    def test_ratio_w_errors(self):
        assert spectops.ratio([10, 20], errors=[2, 2])[0] == 0.5
        assert spectops.ratio([10, 20], errors=[2, 2])[1] == np.sqrt(
            2 * 2 ** 2
        )


class TestSpectopsBandAvg:
    def test_band_avg_1(self):
        assert spectops.band_avg([2, 4], errors=[2, 2])[0] == 3
        assert spectops.band_avg([2, 4], errors=[2, 2])[1] == np.sqrt(
            2 * 2 ** 2
        )

    def test_band_avg_2(self):
        assert spectops.band_avg([[2, 2], [3, 3]], errors=[2, 2])[0][0] == 2.5


class TestSpectopsSlope:
    def test_slope_1(self):
        assert (
            spectops.slope([1, 2], wavelengths=[0, 1], errors=[2, 2])[0] == 1
        )
        assert spectops.slope([1, 2], wavelengths=[0, 1], errors=[2, 2])[
            1
        ] == np.sqrt(2 * 2 ** 2)


class TestSpectopsBandDepth:
    def test_band_depth_1(self):
        assert (
            spectops.band_depth(
                [1, 1, 0], wavelengths=[1, 3, 2], errors=[2, 2, 2]
            )[0]
            == 1
        )
        assert spectops.band_depth(
            [1, 1, 0], wavelengths=[1, 3, 2], errors=[2, 2, 2]
        )[1] == np.sqrt(3 * 2 ** 2)


class TestSpectopsBandMin:
    def test_band_min_1(self):
        assert (
            spectops.band_min([1, 2, 3, 4], wavelengths=[1, 2, 3, 4])[0] == 1
        )

    def test_band_min_2(self):
        assert (
            spectops.band_min([[1, 2], [2, 1]], wavelengths=[1, 2])[0][0] == 1
        )
        assert (
            spectops.band_min([[1, 2], [2, 1]], wavelengths=[1, 2])[0][1] == 2
        )


class TestSpectopsBandMax:
    def test_band_min_1(self):
        assert (
            spectops.band_max([1, 2, 3, 4], wavelengths=[1, 2, 3, 4])[0] == 4
        )

    def test_band_max_2(self):
        assert (
            spectops.band_max([[1, 2], [2, 1]], wavelengths=[1, 2])[0][0] == 2
        )
        assert (
            spectops.band_max([[1, 2], [2, 1]], wavelengths=[1, 2])[0][1] == 1
        )
