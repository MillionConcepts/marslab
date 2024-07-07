from pathlib import Path

import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile

from marslab.imgops.imgutils import normalize_range
from marslab.imgops.render import decorrelation_stretch, make_thumbnail
from marslab.tests.utilz.utilz import normal_array

RNG = np.random.default_rng()


def test_decorrelation_stretch_1():
    for _ in range(100):
        channels = [normal_array() for _ in range(3)]
        stretched = decorrelation_stretch(channels, contrast_stretch=1)
        print(np.std(normalize_range(np.dstack(channels))), np.std(stretched))
        assert np.std(normalize_range(np.dstack(channels))) < np.std(stretched)


def test_make_thumbnail():
    """
    Test the make_thumbnail function.
    """
    try:
        image = RNG.random((1024, 1024, 3))
        jpeg_buf = make_thumbnail(image, (128, 128))
        png_buf = make_thumbnail(image, (512, 512), filetype="png")
        path = make_thumbnail(image, file_or_path_or_buffer="test.png")
        jpeg_buf.seek(0)
        png_buf.seek(0)
        assert JpegImageFile(jpeg_buf).size == (128, 128)
        assert PngImageFile(png_buf).size == (512, 512)
        assert Image.open(path).size == (256, 256)
    finally:
        Path("test.png").unlink(missing_ok=True)