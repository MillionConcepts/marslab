from itertools import product

from matplotlib.cm import register_cmap
from matplotlib.colors import ListedColormap
import numpy as np


def make_orange_teal_cmap():
    teal = (98, 252, 232)
    orange = (255, 151, 41)
    half_len = 256
    vals = np.ones((half_len * 2, 4))
    vals[0:half_len, 0] = np.linspace(orange[0] / half_len, 0, half_len)
    vals[0:half_len, 1] = np.linspace(orange[1] / half_len, 0, half_len)
    vals[0:half_len, 2] = np.linspace(orange[2] / half_len, 0, half_len)
    vals[half_len:, 0] = np.linspace(0, teal[0] / half_len, half_len)
    vals[half_len:, 1] = np.linspace(0, teal[1] / half_len, half_len)
    vals[half_len:, 2] = np.linspace(0, teal[2] / half_len, half_len)
    return ListedColormap(vals, name="orange_teal")


def make_aqua_pink_accent_cmap():
    aqua = (0, 1, 1, 1)
    pink = (1, 0, 1, 1)
    transparent = (0.5, 0.5, 0.5, 0)
    vals = np.full((16, 4), transparent)
    ramp_range = 4
    ramp = np.linspace(1, 0, ramp_range)
    for channel, value in product(range(3), range(ramp_range)):
        gray = 0.5 * ramp[ramp_range - 1 - value]
        colorness = ramp[value]
        vals[:, channel][value] = aqua[channel] * colorness + gray
        vals[:, channel][-1 - value] = pink[channel] * colorness + gray
    return ListedColormap(vals, name="aqua_pink")


register_cmap(cmap=make_orange_teal_cmap())
register_cmap(cmap=make_aqua_pink_accent_cmap())
