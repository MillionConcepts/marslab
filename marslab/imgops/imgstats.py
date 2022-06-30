from functools import reduce

import numpy as np
from dustgoggles.composition import Composition
from marslab.imgops.imgutils import nanmask, ravel_valid
from marslab.imgops.pltutils import clipshow_factory
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import convolve


def pick_mask_constructors(region):
    if region == "inner":
        return np.ma.masked_outside, np.logical_or
    elif region == "outer":
        return np.ma.masked_inside, np.logical_and
    raise ValueError(f"region={region}; region must be 'inner' or 'outer'")


def centered_indices(array):
    y, x = np.indices(array.shape)
    y0, x0 = array.shape[0] // 2, array.shape[1] // 2
    return y - y0, x - x0


def radial_index(array):
    y_ix, x_ix = centered_indices(array)
    return np.sqrt(y_ix ** 2 + x_ix ** 2)


def cut_annulus(array, bounds, region="inner"):
    mask_method, _ = pick_mask_constructors(region)
    distance = radial_index(array)
    pass_min, pass_max = bounds
    pass_min = pass_min if pass_min is not None else distance.min()
    pass_max = pass_max if pass_max is not None else distance.max()
    return np.ma.MaskedArray(
        array,
        mask=mask_method(distance, pass_min, pass_max).mask
    )


def cut_rectangle(array, bounds, region="inner", center=True):
    mask_method, op = pick_mask_constructors(region)
    if center is True:
        y_dist, x_dist = centered_indices(array)
    else:
        y_dist, x_dist = np.indices(array.shape)
    masks = []
    x_bounds, y_bounds = bounds
    for bound, dist in zip((x_bounds, y_bounds), (x_dist, y_dist)):
        pass_min, pass_max = bound
        pass_min = pass_min if pass_min is not None else dist.min()
        pass_max = pass_max if pass_max is not None else dist.max()
        masks.append(mask_method(dist, pass_min, pass_max).mask)
    return np.ma.MaskedArray(array, mask=op(*masks))


def pick_component(complex_array, component="both"):
    if component == "both":
        return complex_array
    if component == "real":
        return complex_array.real
    if component == "imag":
        return complex_array.imag
    if component == "norm":
        return np.abs(complex_array)
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


def zero_mask(masked):
    masked[masked.mask] = 0
    return masked.data


def zerocut(*args, **kwargs):
    return zero_mask(cut_rectangle(*args, **kwargs))


def recursive_cut(
    array, cut_method, boundaries, op=np.logical_and, **cut_kwargs
):
    cuts = [
        cut_method(array, bound, **cut_kwargs).mask for bound in boundaries
    ]
    mask = reduce(op, cuts)
    return np.ma.MaskedArray(array, mask=mask)


def fftplot_factory(title=None, component="norm"):
    clipshow = clipshow_factory(title)
    transformer = Composition((frequency, clipshow.execute))
    transformer.add_insert(0, "component", "norm")


def neighbor_kernels():
    kernels = []
    for neighbor_ix in (0, 1, 2, 3, 5, 6, 7, 8):
        kernel = np.zeros((9,))
        kernel[neighbor_ix] = -1
        kernel[4] = 1
        kernels.append(kernel.reshape(3, 3))
    return kernels


def autocompare(image, copy=True):
    results = []
    std, mean = np.std(image), np.mean(image)
    canvas = nanmask(image, copy)
    for kernel in neighbor_kernels():
        results.append((convolve(canvas, kernel)))
    del canvas
    values = ravel_valid(np.concatenate(results))
    abs_values = np.abs(values)
    return {
        "std": std,
        "mean": mean,
        "edge_mean_abs": np.mean(abs_values),
        "edge_std_abs": np.std(abs_values),
    }