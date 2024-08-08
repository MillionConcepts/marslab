from marslab.imgops import debayer
import numpy as np


class TestMakePatternMasks:
    def test_make_pattern_masks(self):
        rggb = debayer.make_pattern_masks((4, 4), debayer.RGGB_PATTERN, (2, 2))
        assert (rggb["red"][0] == np.array([0, 2, 0, 2])).all()
        assert (rggb["blue"][1] == np.array([1, 1, 3, 3])).all()
