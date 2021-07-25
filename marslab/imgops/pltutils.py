"""
utility functions for dealing with matplotlib
"""
import io

from matplotlib.colorbar import Colorbar
from matplotlib.axes import Subplot
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PIL.Image


def get_mpl_image(fig):
    """
    tries to get the last axis from a mpl figure.
    cranky and fault-intolerant
    """
    # todo: this does not work for the general case. investigate.
    ax = fig.axes[0]
    buffer = io.BytesIO()
    # despine(ax)
    # remove_ticks(ax)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(buffer, bbox_inches=extent, pad_inches=0)
    return PIL.Image.open(buffer)


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
