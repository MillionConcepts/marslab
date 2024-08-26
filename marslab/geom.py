"""
This module contains functions for performing transformations between
coordinate systems. It assumes that the coordinate systems of interest are
defined in relation to one another by offsets and/or rotation quaternions. It
also contains utilities for preprocessing VICAR-style coordinate system
definitions from PDS3 labels in order to facilitate easy transformations from,
e.g., the SOME_INSTRUMENT frame to the SITE frame to the SOME_OTHER_INSTRUMENT
frame.
"""
from itertools import product
from functools import reduce
from numbers import Real
from operator import or_
import re
from typing import Optional, Union, Literal, Sequence

from dustgoggles.func import is_it
from dustgoggles.structures import NestingDict, get_from
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import pdr

Quaternion = Sequence[Real]
"""
Simple alias to clarify type hints. Denotes a quaternion expressed in the form
used by functions in this module. (Note that all functions in this module that
return a Quaternion return it as an ndarray, but will accept both ndarrays 
and other types of sequences as arguments.)

An argument annotated as Quaternion should always be a size-4 1D sequence of 
real numbers in which the first element represents the quaternion's scalar part
and the remaining elements represent its basis/vector part. 

In other words:

`quat: Quaternion = np.array([a, b, c, d])` represents the quaternion (a, v),
where a is a real number and v is the element [b, c, d] of ℝ3 (the real 
coordinate space of dimension 3). 

Equivalently, `quat` represents a + b * i + c * j + d * k, where (i, j, k) are 
the basis elements of a real vector space of dimension 3.
"""

UnitOfAngle = Literal["degrees", "radians", "deg", "rad"]


# TODO: this may be cruft.
def get_geometry_value(
    entity_name: str, frame_name: str, axis_name: str, data: pdr.Data
) -> Union[float, int, str]:
    """
    Fetch the value named `entity_name` for the axis named `axis_name` of the
    coordinate frame named `frame_name`, as defined in the metadata of `data`.
    """
    return get_from(
        data.metaget(f"{frame_name}_DERIVED_GEOMETRY_PARMS"),
        (f"{entity_name}_{axis_name}", "value"),
    )


# TODO: these functions feel messy and redundant.
def get_coordinate_system_properties(
    frame_name: str, data: pdr.Data
) -> Optional[dict[str, Union[np.ndarray, str, float]]]:
    """
    Fetch and format information about a VICAR-style coordinate system named
    `frame_name` from a PDS3-labeled product.
    If there is no coordinate system named `frame_name`, returns None.
    """
    syskeys = [
        k
        for k in data.metadata.fieldcounts
        if re.match(f"{frame_name}_COORD(INATE)?_SYSTEM(_PARMS)?$", k)
    ]
    if len(syskeys) == 0:
        return None
    system = data.metaget(syskeys[0])
    return {
        "name": system.get("COORDINATE_SYSTEM_NAME"),
        "reference_frame": system.get("REFERENCE_COORD_SYSTEM_NAME"),
        "quaternion": np.array(system.get("ORIGIN_ROTATION_QUATERNION")),
        "offset": np.array(system.get("ORIGIN_OFFSET_VECTOR")),
        "orientation": system.get("POSITIVE_AZIMUTH_DIRECTION"),
    }


def get_coordinate_systems(
    data: pdr.Data,
) -> dict[str, dict[str, Union[np.ndarray, str, float]]]:
    """
    Fetch and format information about all VICAR-style coordinate systems
    defined in a PDS3-labeled product.
    """
    # TODO: messy
    frame_names = [
        k.split("_")[0]
        for k in data.metadata.fieldcounts
        if re.match(r"\w+_COORD(INATE)?_SYSTEM(_PARMS)?$", k)
    ]
    return {
        name: get_coordinate_system_properties(name, data)
        for name in frame_names
    }


def _check_unit_of_angle(unit: str) -> None:
    """
    Integrity check  for angular coordinate representation conversion
    functions. Freaks out if we don't know anything about a specified unit of
    angle.
    """
    if unit not in {"degrees", "radians", "deg", "rad"}:
        raise TypeError(
            f"Unsupported angular unit {unit}. Pass 'degrees'/'deg' or "
            f"'radians'/'rad' (default is degrees)."
        )


def cart2sph(
    x0: ArrayLike, y0: ArrayLike, z0: ArrayLike, unit: UnitOfAngle = "degrees"
) -> Union[pd.DataFrame, tuple[Real, Real, Real]]:
    """
    Classic Cartesian-to-spherical coordinate representation converter.
    Returns latitude and longitude in degrees by default.
    Pass `unit="radians"` to return radians. If any of `x0`, `y0`, or `z0` are
    pandas Series/DataFrames or numpy ndarrays, returns a DataFrame with
    columns "lat", "lon", and "r". Otherwise, returns a tuple like
    (lat, lon, r).

    Notes:
        1. Assumes that latitude runs from -90 to 90 degrees.
        2. Returns strictly positive longitude (i.e., uses a 0-360 degree
           longitude system).
        3. If more than one of x0, y0, or z0 are of Series/DataFrame/ndarray
           types, all such arguments must have the same shape. (You may,
           however, mix scalars in as you please.)
    """
    _check_unit_of_angle(unit)
    radius = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    if x0 != 0:
        longitude = np.arctan2(y0, x0)
    else:
        longitude = np.pi / 2
    longitude = longitude % (np.pi * 2)
    latitude = np.arcsin(z0 / np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2))
    if unit in {"degrees", "deg"}:
        latitude = np.degrees(latitude)
        longitude = np.degrees(longitude)
    if reduce(
        or_, map(is_it(pd.DataFrame, np.ndarray, pd.Series), [x0, y0, z0])
    ):
        return pd.DataFrame({"lat": latitude, "lon": longitude, "r": radius})
    return latitude, longitude, radius


def sph2cart(
    lat: ArrayLike,
    lon: ArrayLike,
    radius: ArrayLike = 1,
    unit: UnitOfAngle = "degrees",
) -> Union[pd.DataFrame, tuple[Real, Real, Real]]:
    """
    Classic spherical-to-Cartesian coordinate representation converter. By
    default, assumes `lat` and `lon` are in degrees; pass `unit="radians"` if
    they are in radians. If any of `lat`, `lon`, or `radius` are pandas
    Series/DataFrames or numpy ndarrays, returns a DataFrame with columns "x",
    "y", and "z". Otherwise, returns a tuple like (x, y, z).

    caveats:
    1. this assumes a coordinate convention in which latitude runs from -90
        to 90 degrees.
    """
    # TODO: do there need to be other caveats?
    _check_unit_of_angle(unit)
    if unit in {"degrees", "deg"}:
        lat = np.radians(lat)
        lon = np.radians(lon)
    x0 = radius * np.cos(lat) * np.cos(lon)
    y0 = radius * np.cos(lat) * np.sin(lon)
    z0 = radius * np.sin(lat)

    if reduce(
        or_,
        map(is_it(pd.DataFrame, np.ndarray, pd.Series), [lat, lon, radius]),
    ):
        return pd.DataFrame({"x": x0, "y": y0, "z": z0})
    return x0, y0, z0


# TODO, maybe: this isn't a very good name
def quaternion_multiplication(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """
    Implements the conventional quaternion multiplication operation, i.e.,
    returns q1 ⋅ q2 for q1, q2 ∈ ℍ. See `Quaternion` docstring for format
    convention.
    """
    q1, q2 = map(np.asarray, (q1, q2))
    s1, v1, s2, v2 = q1[0], q1[1:], q2[0], q2[1:]
    scalar = s1 * s2 - np.dot(v1, v2)
    vector = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return np.array([scalar, *vector])


def invert_quaternion(quat: Quaternion) -> Quaternion:
    """
    Return the quaternion whose vector part is the additive inverse of `quat`'s
    vector part and whose scalar part is `quat`'s scalar part. For example,
    `invert_quaternion([1, 2, 3, 4])` returns `[1, -2, -3, -4]`.
    """
    return np.array([quat[0]] + [-1 * e for e in quat[1:]])


def rotate_unit_vector(
    alt: float, az: float, quat: Quaternion, clockwise: bool = True
) -> np.ndarray:
    """
    Apply a rotation expressed as a unit quaternion to a unit vector expressed
    as spherical coordinates.

    Assumes units of degrees. Also assumes left-handed coordinate systems by
    default; pass `clockwise=False` to specify right-handed system.
    """
    source_cartesian = np.array(sph2cart(alt, az))
    if clockwise is True:
        source_cartesian *= np.array([-1, -1, 1])
    zero_quaternion = np.array([0, *source_cartesian])
    inverse_rotation = invert_quaternion(quat)
    q_times_0_v = quaternion_multiplication(quat, zero_quaternion)
    v_prime = quaternion_multiplication(q_times_0_v, inverse_rotation)
    assert np.isclose(v_prime[0], 0)
    target_cartesian = v_prime[1:]
    if clockwise is True:
        target_cartesian *= np.array([-1, -1, 1])
    # TODO: it's pointless to return the third (radius) element, because
    #  the result will always be a unit vector. Will probably need to modify
    #  downstream to make this change.
    return np.array(cart2sph(*target_cartesian))


def get_coordinates(
    data: pdr.Data,
) -> dict[str, dict[str, dict[Literal["AZIMUTH", "ELEVATION"], float]]]:
    """
    Fetch and organize all VICAR-style azimuth/elevation values mentioned in
    the metadata of a pdr.Data object.

    For example, if a product's label gives INSTRUMENT and SOLAR
    azimuth/elevation values in both SITE and ROVER frames:

    >>> coordinated_data = pdr.read("coordinated.lbl")
    >>> get_coordinates(coordinated_data)

    Expected output (numeric values will of course vary depending on product):
    ```
    {'ROVER': {'INSTRUMENT': {'AZIMUTH': 351.484, 'ELEVATION': -44.5026},
    'SOLAR': {'AZIMUTH': 89.4054, 'ELEVATION': 67.2452}},
    'SITE': {'INSTRUMENT': {'AZIMUTH': 175.589, 'ELEVATION': -35.0211},
    'SOLAR': {'AZIMUTH': 293.467, 'ELEVATION': 63.6488}}}
    ```
    """
    # TODO: maybe expand this
    axes = ("AZIMUTH", "ELEVATION")
    entities = set()
    for axis in axes:
        entities.update(
            {
                k.replace(f"_{axis}", "")
                for k in data.metadata.fieldcounts
                if k.endswith(f"_{axis}")
            }
        )
    coordinates = NestingDict()
    syskeys = filter(
        lambda k: k.endswith("_GEOMETRY_PARMS"), data.metadata.fieldcounts
    )
    for key, axis, entity in product(syskeys, axes, entities):
        block = data.metaget(key)
        if block is None:
            continue
        record = block.get(f"{entity}_{axis}")
        if record is not None:
            coordinates[key.split("_")[0]][entity][axis] = record["value"]
    return coordinates.todict()


def transform_angle(
    source_frame: str, target_frame: str, entity: str, data: pdr.Data
) -> np.ndarray:
    """
    Transform an angle described in VICAR-style notation in a PDS3 label from
    one coordinate system to another. The label must also provide a quaternion
    defining the relative orientations of those coordinate systems.

    NOTE: "Angle" here is slightly misleading; it really refers to a unit
      direction vector expressed as altitude/azimuth coordinates.

    For example, assuming that "coordinated.lbl" defines both SOLAR_AZIMUTH
    and SOLAR_ELEVATION in SITE frame, and also gives a quaternion that
    rotates coordinates in SITE to coordinates in HEAD, this will return solar
    altitude and azimuth in HEAD:

    >>> coordinated_data = pdr.read("coordinated.lbl")
    >>> transform_angle("SITE", "HEAD", "SOLAR", coordinated_data)
    """
    coordinates = get_coordinates(data)
    systems = get_coordinate_systems(data)
    source_info = systems.get(source_frame)
    target_info = systems.get(target_frame)
    if source_info is None:
        if not target_info["reference_frame"].startswith(source_frame):
            raise ValueError("can't directly convert between these frames.")
        quaternion = invert_quaternion(target_info["quaternion"])
    else:
        quaternion = source_info["quaternion"]
    coord = coordinates[source_frame][entity]
    if target_info is not None:
        clockwise = "clockwise" in target_info["orientation"].lower()
    else:
        clockwise = "clockwise" in source_info["orientation"].lower()
    return rotate_unit_vector(
        coord["ELEVATION"], coord["AZIMUTH"], quaternion, clockwise
    )
