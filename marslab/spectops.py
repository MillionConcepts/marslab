"""
reference implementations of elementary spectrum operations.
overloaded to take dataframes, ndarrays, sequences, and scalars.

generic signature: (reflectance, errors, wavelength). Errors are always
optional. Wavelengths are optional (and ignored) for operations that do not
consider units of wavelength (like ratio), and required for operations that do
(like slope).

order of band shoulders is: left, right, center --
or unordered for things like average

CAUTION: everything in this module must be passed in 'last index fastest' /
'row-primary' orientation. this means that most pandas dataframes need to
be transposed to get meaningful results.
"""
from typing import Optional, Sequence, Union

import numpy as np
from numpy.linalg import norm
import pandas as pd


# TODO: there are a variety of other fairly simple operations we could
#  usefully add to this module.


Specvals = Union[
    pd.DataFrame,
    pd.Series,
    Sequence[float],
    Sequence[Sequence[float]],
    np.ndarray
]
"""
Valid types for "spectral data" arguments to many functions in this module.
"""


def preprocess_input(
    reflectance: Specvals,
    errors: Optional[Specvals] = None,
    wavelengths: Optional[Specvals] = None,
    band_arity: Optional[int] = None,
    require_wavelengths: bool = False
) -> tuple[Specvals, Specvals, Specvals]:
    """
    Did someone pass a spectop function something silly? If so, raise an
    exception. Then return all arguments cast to ndarray.
    """
    if reflectance is None:
        raise ValueError("All spectops require reflectance values.")
    if require_wavelengths is True and wavelengths is None:
        raise ValueError("This op requires wavelengths.")
    # this all looks repetitive but the cases are actually different. it is
    # permissible for `errors` and `reflectance` to be 2D arrays, but not
    # for `wavelengths`.
    if hasattr(wavelengths, "ndim") and wavelengths.ndim != 1:
        raise ValueError("Wavelengths must be 1D.")
    # TODO: these should really be None, not array([None]) when not present.
    #  need to change callers to match that signature.
    # noinspection PyTypeChecker
    ref_array, err_array, wave_array = tuple(
        [np.asarray(thing) for thing in (reflectance, errors, wavelengths)]
    )
    if band_arity is not None and ref_array.shape[0] != band_arity:
        raise ValueError(
            f"This op requires {band_arity} bands; got {ref_array.shape[0]}."
        )
    if wavelengths is not None and wave_array.shape[0] != ref_array.shape[0]:
        raise ValueError("There must be as many wavelength values as bands.")
    if errors is not None and err_array.shape != ref_array.shape:
        raise ValueError("Reflectance and errors must have the same shape.")
    return ref_array, err_array, wave_array


# TODO: this smells bad.
def reindex_or_mask(
    *object_pairs: tuple[Optional[Specvals], Optional[Specvals]]
) -> list[Specvals]:
    returning = []
    for pair in object_pairs:
        thing = pair[0]
        if isinstance(pair[1], (pd.DataFrame, pd.Series)):
            thing = pd.Series(thing, index=pair[1].columns)
        elif isinstance(pair[1], Sequence):
            masked = filter(lambda a: isinstance(a, np.ma.MaskedArray), pair[1])
            masks = [m.mask for m in masked]
            if len(masks) > 0:
                thing = np.ma.masked_array(thing, mask=sum(masks))
        returning.append(thing)
    return returning


# TODO: we remain uncertain about the validity / relevance of error
#  calculations for some operations.
def addition_in_quadrature(errors: Specvals) -> Optional[Specvals]:
    """
    take what numpy thinks is the norm. In most cases this should be equivalent
    to summing in quadrature. If `errors` contains `None` (generally meaning
    it is an array([None]) produced by `preprocess_input()`), just return None.
    """
    if None in errors:
        return None
    return norm(errors, axis=0)


def ratio(
    reflectance: Specvals,
    errors: Optional[Specvals] = None,
    _wavelengths: Optional[Specvals] = None,
) -> list[Specvals]:
    """
    just a ratio function. notionally, ratio of reflectance values between two
    bands. ignores wavelengths if passed.
    """
    ref_arr, err_arr, _wave_arr = preprocess_input(reflectance, errors, None)
    ratio_value = ref_arr[0] / ref_arr[1]
    if None not in err_arr:
        err_arr = err_arr[0:2]
    err_vals = addition_in_quadrature(err_arr)
    return reindex_or_mask((ratio_value, reflectance), (err_vals, errors))


def band_avg(
    reflectance: Specvals,
    errors: Optional[Specvals] = None,
    _wavelengths: Optional[Specvals] = None,
) -> list[Specvals]:
    """
    Just an averaging function. Notionally, average of all bands within a
    wavelength region. Note that if the caller cares that the bands are
    contiguous wrt an instrument, it is their responsibility to make sure they
    didn't skip any. Ignores `wavelengths` if passed.
    """
    ref_arr, err_arr, _wave_arr = preprocess_input(reflectance, errors, None)
    ref_mean = np.mean(ref_arr, axis=0)
    error_values = addition_in_quadrature(err_arr)
    return reindex_or_mask((ref_mean, reflectance), (error_values, errors))


def slope(
    reflectance: Specvals,
    errors: Optional[Specvals] = None,
    wavelengths: Optional[Specvals] = None
) -> list[Specvals]:
    """
    Just a slope function. notionally, slope of a line segment in the
    reflectance/wavelength plane drawn between between reflectances and
    center wavelengths oatf two bands.
    """
    ref_arr, err_arr, wave_arr = preprocess_input(
        reflectance, errors, wavelengths, 2, True
    )
    difference, distance = ref_arr[1] - ref_arr[0], wave_arr[1] - wave_arr[0]
    slope_value = difference / distance
    error_values = addition_in_quadrature(err_arr)
    if error_values is not None:
        error_values /= distance
    return reindex_or_mask((slope_value, reflectance), (error_values, errors))


def band_depth(
    reflectance: Specvals,
    errors: Optional[Specvals] = None,
    wavelengths: Specvals = None,
) -> list[Specvals]:
    """
    Standard band depth function. Requires waveleng
    wavelength space between two filters. only looks at the first
    two 'rows' you pass it, should you pass it more.
    """
    ref_arr, err_arr, wave_arr = preprocess_input(
        reflectance, errors, wavelengths, 3, True
    )    
    left, right, middle = wave_arr[:3]
    if len({left, right, middle}) != 3:
        raise ValueError(
            "band depth between a wavelength and itself is undefined"
        )
    if not (max(left, right) > middle > min(left, right)):
        raise ValueError(
            "band depth can only be calculated at a band within the "
            "chosen range."
        )
    distance = middle - left
    slope_value = slope(ref_arr[0:2], None, np.array([left, right]))[0]
    continuum_ref = ref_arr[0] + slope_value * distance
    band_depth_value = 1 - ref_arr[2] / continuum_ref
    if (err_vals := addition_in_quadrature(err_arr)) is not None:
        err_vals /= continuum_ref
    return reindex_or_mask((band_depth_value, reflectance), (err_vals, errors))


# TODO, maybe: implement band_depth_min
#  nobody actually seems to want it

# TODO: is there any meaningful way to calculate error
#  for these min and max functions?


def band_min(
    reflectance: Specvals,
    _errors: Optional[Specvals] = None,
    wavelengths: Optional[Specvals] = None,
):
    """
    Returns wavelength at which `reflectance` takes on its minimum value.
    Ignores errors if passed.
    """
    ref_arr, _err_arr, wave_arr = preprocess_input(
        reflectance, None, wavelengths, require_wavelengths=True
    )
    return reindex_or_mask(
        (wave_arr[np.argmin(ref_arr, axis=0)], reflectance), (None, None)
    )


def band_max(
    reflectance: Specvals,
    _errors: Optional[Specvals] = None,
    wavelengths: Optional[Specvals] = None,
) -> list[Specvals]:
    """
    Returns wavelength at which `reflectance` takes on its minimum value.
    Ignores errors if passed.
    """
    ref_arr, _err_arr, wave_arr = preprocess_input(
        reflectance, None, wavelengths, require_wavelengths=True
    )
    return reindex_or_mask(
        (wave_arr[np.argmax(ref_arr, axis=0)], reflectance), (None, None)
    )


SPECTOP_NAMES = (
    "band_avg",
    "band_depth",
    "band_max",
    "band_min",
    "ratio",
    "slope",
)
"""
Names of all functions in this module that follow the Spectop signature 
convention.
"""