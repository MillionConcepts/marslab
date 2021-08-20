from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL.ImageStat import Stat

from marslab.imgops.loaders import pil_load_rgb
from marslab.imgops.pltutils import get_mpl_image, strip_axes, attach_axis
from marslab.tests.utilz.utilz import render_in_memory
from marslab.tests.data.test_cases import test_data_path

# don't have the tests make a bunch of GUI mess
mpl.use("agg")

# TODO: more cases, clean this up


def test_with_trivial_case():
    red_square = pil_load_rgb(Path(test_data_path, "red_square.png"))
    red_square_array = np.dstack(
        [red_square["R"], red_square["G"], red_square["B"]]
    )
    fig, ax = plt.subplots()
    im = ax.imshow(red_square_array, cmap="viridis")
    cax = attach_axis(ax, size="100%", pad="0%")
    colorbar = plt.colorbar(im, cax=cax)
    rendered = render_in_memory(fig)
    # purple-black to green-yellow colorbar, red figure, ticks everywhere
    # a full range of values should be represented in each color band;
    # alpha should be full everywhere (won't be the case if you render it
    # in a Notebook instead!)
    assert rendered.getextrema() == ((0, 255), (0, 255), (0, 255), (255, 255))
    imstats = Stat(rendered)
    # should be redder than it is green
    assert imstats._getmean()[0] > imstats._getmean()[1]
    # and also more consistently red
    assert imstats._getvar()[0] < imstats._getvar()[1]
    strip_axes(ax)
    strip_axes(colorbar)
    rendered = render_in_memory(fig)
    # no ticks or spines now. lowest range will still have some red in it.
    assert rendered.getextrema() == ((30, 255), (0, 255), (0, 255), (255, 255))
    # now just get the red square
    final_red_square = get_mpl_image(fig)
    # which should be entirely red
    assert np.all(np.array(final_red_square)[:, :, 0] == 255)
