"""
utility functions for dealing with matplotlib
"""

import io

import PIL.Image


def get_mpl_image(fig):
    """
    tries to get the first axis from a mpl figure.
    cranky and fault-intolerant
    """
    ax = fig.axes[0]
    buffer = io.BytesIO()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(buffer, bbox_inches=extent)
    return PIL.Image.open(buffer)


def remove_ticks(ax):
    """remove ticks from axis"""
    ax.set_xticks([])
    ax.set_yticks([])


def set_colorbar_font(colorbar, colorbar_fp):
    """set font of colorbar"""
    for tick in colorbar.ax.get_yticklabels():
        tick.set_font_properties(colorbar_fp)
    colorbar.ax.get_yaxis().get_offset_text().set_font_properties(colorbar_fp)


def set_label(
    fig, text, ax_ix=0, fontproperties=None, loc="center", x_or_y="x"
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
