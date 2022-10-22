from functools import reduce

from cytoolz import keyfilter, keymap
from dustgoggles.composition import Composition
from marslab.imgops.imgutils import nanmask, ravel_valid, cut_annulus, \
    cut_rectangle
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy import stats
from scipy.ndimage import convolve, sobel


def pick_component(complex_array, component="both"):
    if component == "both":
        return complex_array
    if component == "real":
        return complex_array.real
    if component == "imag":
        return complex_array.imag
    if component in ("norm", "magnitude"):
        return np.abs(complex_array)
    if component == "phase":
        return np.arctan2(complex_array.real, complex_array.imag)
    raise ValueError(f"{component} is not a known component.")


def frequency(array, component="both"):
    freq_array = fftshift(fft2(array))
    return pick_component(freq_array, component)


def space(array, component="both"):
    space_array = ifft2(ifftshift(array))
    return pick_component(space_array, component)


def bandfilter(array, bounds, region, component):
    freq_array = frequency(array)
    annulus = cut_annulus(freq_array, bounds, region)
    annulus[annulus.mask] = 0
    return space(annulus.data, component)


def bandcut(array, bounds, component='real'):
    return bandfilter(array, bounds, 'outer', component)


def bandpass(array, bounds, component='real'):
    return bandfilter(array, bounds, 'inner', component)


def boxfilter(array, bounds, region, component, center=True):
    freq_array = frequency(array)
    rect = cut_rectangle(freq_array, bounds, region, center)
    rect[rect.mask] = 0
    return space(rect.data, component)


def recursive_cut(
    array, cut_method, boundaries, op=np.logical_and, **cut_kwargs
):
    cuts = [
        cut_method(array, bound, **cut_kwargs).mask for bound in boundaries
    ]
    mask = reduce(op, cuts)
    return np.ma.MaskedArray(array, mask=mask)


def fftplot_factory(title=None, component="norm"):
    from marslab.imgops.pltutils import clipshow_factory

    clipshow = clipshow_factory(title)
    transformer = Composition((frequency, clipshow.execute))
    transformer.add_insert(0, "component", component)


def neighbor_kernels():
    kernels = []
    for neighbor_ix in (0, 1, 2, 3, 5, 6, 7, 8):
        kernel = np.zeros((9,))
        kernel[neighbor_ix] = -1
        kernel[4] = 1
        kernels.append(kernel.reshape(3, 3))
    return kernels


def autocompare(image, copy=True):
    abs_image = np.abs(image)
    statistics = {
        "std": np.std(image),
        "mean": np.mean(image),
        "std_abs": np.std(abs_image),
        "mean_abs": np.mean(abs_image)
    }
    edge_results = []
    canvas = nanmask(image, copy)
    for kernel in neighbor_kernels():
        edge_results.append((convolve(canvas, kernel)))
    del canvas
    edge_values = np.abs(ravel_valid(np.concatenate(edge_results)))
    statistics |= {
        "edge_mean_abs": np.mean(edge_values),
        "edge_std_abs": np.std(edge_values),
    }
    sobel_values = np.abs(ravel_valid(sobel(image)))
    statistics |= {
        "sobel_mean_abs": np.mean(sobel_values),
        "sobel_std_abs": np.std(sobel_values)
    }
    return statistics


def center_and_scale(array):
    return (array - array.mean()) / array.std()


def center(array):
    return array - array.mean()


def scale(array):
    return array / array.std()


def unpack_scipy_describe(result):
    return {
        'mean': result.mean,
        'n': result.nobs,
        'min': result.minmax[0],
        'max': result.minmax[1],
        'range': result.minmax[1] - result.minmax[0],
        'skew': result.skewness,
        'kurtosis': result.kurtosis,
        'var': result.variance,
    }


# noinspection PyTypeChecker
def rvdescribe(array, rv=True):
    result = {'shape': array.shape}
    if rv is True:
        array = ravel_valid(array)
    result |= unpack_scipy_describe(stats.describe(array))
    result['std'] = np.sqrt(result['var'])
    if result['min'] >= 0:
        return result
    absresult = unpack_scipy_describe(stats.describe(np.abs(array)))
    absresult = keyfilter(lambda k: k != "n", absresult)
    absresult = keymap(lambda x: f"{x}_abs", absresult)
    absresult['std_abs'] = np.sqrt(absresult['var_abs'])
    return result | absresult


def gradient_stats(
    array, sample_distance=1, get_mag=True, return_components=True
):
    dy, dx = np.gradient(array, sample_distance)
    component_dict = {'dy': dy, 'dx': dx}
    if get_mag is True:
        component_dict['norm'] = dy ** 2 + dx ** 2
    stat_dict = {k: rvdescribe(v) for k, v in component_dict.items()}
    if return_components is True:
        return stat_dict, component_dict
    return stat_dict, {}


def fhplot(
    array, bins=128, vrange=None, ax=None, return_counts=False, **mpl_kwargs
):
    try:
        import fast_histogram as fh
        import matplotlib.pyplot as plt
    except ImportError:
        raise ValueError(
            "The fast-histogram and maplotlib libraries must be installed to "
            "use marslab.imgops.imgstats.fhplot."
        )
    if len(array.shape) > 1:
        raise ValueError("this function only plots 1D arrays.")
    if vrange is None:
        vrange = (array.min(), array.max())
    counts = fh.histogram1d(array, bins=bins, range=vrange)
    bin_positions = np.linspace(*vrange, bins + 1)
    if ax is None:
        ax = plt.gca()
    ax.hist(bin_positions[:-1], bin_positions, weights=counts, **mpl_kwargs)
    if return_counts is True:
        return ax, counts, bin_positions
    return ax
