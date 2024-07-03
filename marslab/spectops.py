"""
reference implementations of elementary spectrum operations.
overloaded to take dataframes, ndarrays, sequences, and scalars.

generic signature: (reflectance, errors, wavelength)

order of band shoulders is: left, right, center --
or unordered for things like average

Caution: everything must be passed in 'last index fastest' / 'row-primary'
orientation. this means that a lot of pandas dataframes will need to be
transposed to get meaningful results.
"""
from typing import Optional, Sequence, Union

import numpy as np
from numpy.linalg import norm
import pandas as pd


# TODO: make all kwarg signatures flatly identical
# TODO: there are a variety of other fairly simple operations we could
#  usefully add to this module.


Specvals = Union[pd.DataFrame, pd.Series, list[list[float]], np.ndarray]
"""
Valid types for "spectral data" arguments to many functions in this module.
"""


def preprocess_input(
    reflectance: Specvals,
    errors: Optional[Specvals] = None,
    wavelengths: Specvals = None
) -> list[np.ndarray]:
    """
    did someone pass this function something silly? If so, raise an exception.
    Then return all arguments cast to ndarray.
    """
    if errors is not None:
        if isinstance(reflectance, (np.ndarray, pd.DataFrame, pd.Series)):
            assert reflectance.shape == errors.shape
        else:
            assert len(reflectance) == len(errors)
    if wavelengths is not None:
        if isinstance(reflectance, (np.ndarray, pd.DataFrame, pd.Series)):
            assert reflectance.shape[0] == len(wavelengths)
        else:
            assert len(reflectance) == len(wavelengths)
    return [np.asarray(thing) for thing in (reflectance, errors, wavelengths)]


# TODO: this smells bad.
def reindex_or_mask(
    *object_pairs: Sequence[tuple[Optional[Specvals], Optional[Specvals]]]
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
def addition_in_quadrature(errors: Specvals) -> Optional[float]:
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
    errors: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray, None] = None,
    _wavelengths: Optional[Sequence] = None,
) -> list[Specvals]:
    """
    just a ratio function. notionally, ratio of reflectance
    values between two filters. looks at the first two 'rows'
    or elements of what it's passed...if you pass it more,
    it ignores them. ignores wavelength values.
    """
    reflectance_array, error_array, _wavelength_array = preprocess_input(
        reflectance, errors, _wavelengths
    )
    ratio_value = reflectance_array[0] / reflectance_array[1]
    if None not in error_array:
        error_array = error_array[0:2]
    error_values = addition_in_quadrature(error_array)
    return reindex_or_mask(
        (ratio_value, reflectance), (error_values, errors)
    )


def band_avg(
    reflectance: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray],
    errors: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray, None] = None,
    _wavelengths: Optional[Sequence] = None,
):
    """
    just an averaging function. notionally, average of all filters within a
    contiguous block of wavelengths. your responsibility to make sure
    you didn't skip any!
    """
    reflectance_array, error_array, _wavelength_array = preprocess_input(
        reflectance, errors, _wavelengths
    )

    average_of_reflectance = np.mean(reflectance_array, axis=0)
    error_values = addition_in_quadrature(error_array)
    return reindex_or_mask(
        (average_of_reflectance, reflectance), (error_values, errors)
    )


def slope(
    reflectance: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray],
    errors: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray, None] = None,
    wavelengths: Union[Sequence] = None,
):
    """
    just a slope function. notionally, slope in reflectance-
    wavelength space between two filters. only looks at the first
    two 'rows' you pass it, should you pass it more.
    """
    assert wavelengths is not None, "wavelengths must be passed to slope()"
    assert None not in wavelengths, "wavelengths must be passed to slope()"
    reflectance_array, error_array, wavelength_array = preprocess_input(
        reflectance, errors, wavelengths
    )
    difference = reflectance_array[1] - reflectance_array[0]
    distance = wavelength_array[1] - wavelength_array[0]
    slope_value = difference / distance
    error_values = addition_in_quadrature(error_array)
    if error_values is not None:
        error_values /= distance
    return reindex_or_mask(
        (slope_value, reflectance), (error_values, errors)
    )


def band_depth(
    reflectance: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray],
    errors: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray, None] = None,
    wavelengths: Union[Sequence] = None,
):
    assert (
        wavelengths is not None
    ), "wavelengths must be passed to band_depth()"
    assert (
        None not in wavelengths
    ), "wavelengths must be passed to band_depth()"
    reflectance_array, error_array, wavelength_array = preprocess_input(
        reflectance, errors, wavelengths
    )
    assert (
        np.shape(reflectance_array)[0] == 3
    ), "band depth requires exactly 3 bands"
    assert (
        np.shape(wavelength_array)[0] == 3
    ), "band depth requires exactly 3 bands"
    # just for clarity
    wave_left = wavelength_array[0]
    wave_right = wavelength_array[1]
    wave_middle = wavelength_array[2]
    if len({wave_left, wave_right, wave_middle}) != 3:
        raise ValueError(
            "band depth between a wavelength and itself is undefined"
        )
    if not (
        max(wave_left, wave_right) > wave_middle > min(wave_left, wave_right)
    ):
        raise ValueError(
            "band depth can only be calculated at a band within the "
            "chosen range."
        )
    distance = wave_middle - wave_left
    slope_value = slope(
        reflectance_array[0:2], None, np.array([wave_left, wave_right])
    )[0]
    continuum_ref = reflectance_array[0] + slope_value * distance
    band_depth_value = 1 - reflectance_array[2] / continuum_ref
    error_values = addition_in_quadrature(error_array)
    if error_values is not None:
        error_values /= continuum_ref
    return reindex_or_mask(
        (band_depth_value, reflectance), (error_values, errors)
    )


# TODO, maybe: implement band_depth_min
#  nobody actually seems to want it

# TODO: is there any meaningful way to calculate error
#  for these min and max functions?


def band_min(
    reflectance: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray],
    _errors: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray, None] = None,
    wavelengths: Optional[Sequence] = None,
):
    assert wavelengths is not None, "wavelengths must be passed to band_min()"
    assert None not in wavelengths, "wavelengths must be passed to band_min()"
    reflectance_array, _error_array, wavelength_array = preprocess_input(
        reflectance, None, wavelengths
    )
    return reindex_or_mask(
        (
            wavelength_array[np.argmin(reflectance_array, axis=0)],
            reflectance,
        ),
        (None, None),
    )


def band_max(
    reflectance: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray],
    _errors: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray, None] = None,
    wavelengths: Optional[Sequence] = None,
):
    assert wavelengths is not None, "wavelengths must be passed to band_max()"
    assert None not in wavelengths, "wavelengths must be passed to band_max()"
    reflectance_array, _error_array, wavelength_array = preprocess_input(
        reflectance, None, wavelengths
    )
    return reindex_or_mask(
        (
            wavelength_array[np.argmax(reflectance_array, axis=0)],
            reflectance,
        ),
        (None, None),
    )


SPECTOP_NAMES = (
    "band_avg",
    "band_depth",
    "band_max",
    "band_min",
    "ratio",
    "slope",
)
