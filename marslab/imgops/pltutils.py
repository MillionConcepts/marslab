"""
utility functions for dealing with matplotlib
"""
import io
from math import ceil, sqrt
from pathlib import Path
from typing import Optional, Sequence, Union

from dustgoggles.composition import Composition
from matplotlib.colorbar import Colorbar
from matplotlib.axes import Subplot
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import PIL.Image
from skimage.transform import rescale

from marslab.imgops.imgutils import std_clip, eightbit


def get_mpl_image(fig: Figure):
    """
    tries to get the last axis from a mpl figure.
    cranky and fault-intolerant
    """
    # todo: this does not work for the general case. investigate.
    buffer = io.BytesIO()
    fig.savefig(buffer, bbox_inches="tight", pad_inches=0)
    return PIL.Image.open(buffer)


def fig2arr(fig: plt.Figure) -> np.ndarray:
    """naively convert a matplotlib figure to a numpy array"""
    fig.canvas.draw()
    # noinspection PyUnresolvedReferences
    raveled = np.frombuffer(
        fig.canvas.buffer_rgba(), dtype=np.uint8
    )
    return raveled.reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )[:, :, :3]


def remove_ticks(ax):
    """remove ticks from axis"""
    if isinstance(ax, Colorbar):
        ax.set_ticks([])
        return
    ax.set_xticks([])
    ax.set_yticks([])


def despine(ax: Subplot, edges=("top", "bottom", "left", "right")):
    # Remove axes and bounding box for a given subplot object axes
    if isinstance(ax, Colorbar):
        ax.outline.set_visible(False)
        return
    for p in edges:
        ax.spines[p].set_visible(False)


def strip_axes(ax: Subplot):
    remove_ticks(ax)
    despine(ax)


def set_colorbar_font(colorbar, colorbar_fp):
    """set font of colorbar"""
    for tick in colorbar.ax.get_yticklabels():
        tick.set_font_properties(colorbar_fp)
    colorbar.ax.get_yaxis().get_offset_text().set_font_properties(colorbar_fp)


def set_label(
    fig: Figure, text, ax_ix=0, fontproperties=None, loc="center", x_or_y="x"
):
    """
    convenience wrapper for mpl.axes._subplots.AxesSubplot.xlabel / ylabel
    """
    ax = fig.axes[ax_ix]
    if x_or_y == "x":
        method = ax.set_xlabel
    elif x_or_y == "y":
        method = ax.set_ylabel
    else:
        raise ValueError('x_or_y should be "x" or "y"')
    return method(text, loc=loc, fontproperties=fontproperties)


def attach_axis(ax: Subplot = None, where="right", size="50%", pad=0.1):
    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    return divider.append_axes(where, size=size, pad=pad)


def prefigure(func, title=None):
    def prefigured(*args, **kwargs):
        plt.figure()
        result = func(*args, **kwargs)
        if title is not None:
            plt.title(title)
        return result

    return prefigured


def clipshow_factory(title=None):
    return Composition(
        {"clip": std_clip, "plot": prefigure(plt.imshow, title)},
        inserts={"clip": {"sigma": 1}},
    )


def gridshow(
    arrays, kwarg_seq=None, *imshow_args, strip=True, **imshow_kwargs
):
    size = ceil(sqrt(len(arrays)))
    fig, grid = plt.subplots(size, size)
    kwarg_seq = [] if kwarg_seq is None else kwarg_seq
    kwarg_seq = [
        {} if i + 1 > len(kwarg_seq) else kwarg_seq[i]
        for i in range(len(arrays))
    ]
    for cell_ix in range(len(arrays)):
        x, y = (cell_ix % size, (cell_ix // size) % size)
        cell = grid[x][y]
        cell.imshow(
            arrays[cell_ix],
            *imshow_args,
            **kwarg_seq[cell_ix],
            **imshow_kwargs
        )
    if strip is True:
        tuple(map(strip_axes, grid.ravel()))
    fig.tight_layout()
    return fig, grid


def dpi_from_image(fig, ax_ix=0, im_ix=0):
    image = fig.axes[ax_ix].get_images()[im_ix]
    _, im_x_pix = image.get_size()
    dpi_transform = fig.dpi_scale_trans.inverted()
    im_x_in = image.get_window_extent().transformed(dpi_transform).bounds[2]
    return im_x_pix / im_x_in
