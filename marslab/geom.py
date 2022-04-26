"""
geometry utilities based on metadata in object labels.
"""
from itertools import product
from functools import reduce
from operator import or_

from dustgoggles.func import is_it
from dustgoggles.structures import NestingDict, get_from
import numpy as np
import pandas as pd
import pdr


def get_geometry_value(entity_name, frame_name, axis_name, data: pdr.Data):
    return get_from(
        data.metaget(f"{frame_name}_DERIVED_GEOMETRY_PARMS"),
        (f"{entity_name}_{axis_name}", "value")
    )


def get_coordinate_system_properties(frame_name, data):
    system = data.metaget(f"{frame_name}_COORDINATE_SYSTEM")
    if system is None:
        return None
    return {
        "name": system.get("COORDINATE_SYSTEM_NAME"),
        "reference_frame": system.get("REFERENCE_COORD_SYSTEM_NAME"),
        "quaternion": np.array(system.get("ORIGIN_ROTATION_QUATERNION")),
        "offset": np.array(system.get("ORIGIN_OFFSET_VECTOR")),
        "orientation": system.get("POSITIVE_AZIMUTH_DIRECTION")
    }


def get_coordinate_systems(data: pdr.Data):
    frame_names = [
        k.replace("_COORDINATE_SYSTEM", "")
        for k in data.metadata.fieldcounts
        if k.endswith("_COORDINATE_SYSTEM")
    ]
    return {
        name: get_coordinate_system_properties(name, data)
        for name in frame_names
    }


def cart2sph(x0, y0, z0, unit: str = "degrees"):
    """
    convert cartesian to spherical coordinates. returns degrees by default;
    pass unit="radians" to return radians. if passed any arraylike objects,
    returns a DataFrame, otherwise, returns a tuple of values.

    caveats:
    1. this assumes a coordinate convention in which latitude runs from -90
        to 90 degrees.
    2. returns longitude in strictly positive coordinates.
    """
    radius = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    if x0 != 0:
        longitude = np.arctan2(y0, x0)
    else:
        longitude = np.pi / 2
    longitude = longitude % (np.pi * 2)
    latitude = np.arcsin(z0 / np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2))
    if unit == "degrees":
        latitude = np.degrees(latitude)
        longitude = np.degrees(longitude)
    if reduce(
        or_, map(is_it(pd.DataFrame, np.ndarray, pd.Series), [x0, y0, z0])
    ):
        return pd.DataFrame({"lat": latitude, "lon": longitude, "r": radius})
    return latitude, longitude, radius


def sph2cart(lat, lon, radius=1, unit: str = "degrees"):
    """
    convert spherical to cartesian coordinates. assumes input is in degrees
    by default; pass unit="radians" to specify input in radians. if passed any
    arraylike objects, returns a DataFrame, otherwise, returns a tuple of
    values.

    caveats:
    1. this assumes a coordinate convention in which latitude runs from -90
        to 90 degrees.
    """
    if unit == "degrees":
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


def quaternion_multiplication(q1, q2):
    s1, v1 = q1[0], q1[1:]
    s2, v2 = q2[0], q2[1:]
    scalar = s1 * s2 - np.dot(v1, v2)
    vector = (s1 * v2 + s2 * v1 + np.cross(v1, v2))
    return np.array([scalar, *vector])


def invert_quaternion(quaternion):
    return np.array([quaternion[0]] + [-1 * q for q in quaternion[1:]])


def rotate_unit_vector(alt, az, rotation_quaternion, orientation="clockwise"):
    """
    assumptions:
    1. left-handed coordinate system,
    2. scalar-first quaternion representation.
    3. units are degrees.
    """
    source_cartesian = np.array(sph2cart(alt, az))
    if orientation == "clockwise":
        source_cartesian *= np.array([-1, -1, 1])
    zero_quaternion = np.array([0, *source_cartesian])
    inverse_rotation = invert_quaternion(rotation_quaternion)
    q_times_0_v = quaternion_multiplication(
        rotation_quaternion, zero_quaternion
    )
    v_prime = quaternion_multiplication(q_times_0_v, inverse_rotation)
    assert np.isclose(v_prime[0], 0)
    target_cartesian = v_prime[1:]
    if orientation == "clockwise":
        target_cartesian *= np.array([-1, -1, 1])
    return np.array(cart2sph(*target_cartesian))


def get_coordinates(data: pdr.Data):
    # TODO: maybe expand this
    axes = ("AZIMUTH", "ELEVATION")
    entities = set()
    for axis in axes:
        entities.update({
            k.replace(f"_{axis}", "")
            for k in data.metadata.fieldcounts
            if k.endswith(f"_{axis}")
        })
    coordinates = NestingDict()
    parms = {
        k for k in data.metadata.fieldcounts
        if k.endswith("_DERIVED_GEOMETRY_PARMS")
    }
    for system, axis, entity in product(parms, axes, entities):
        block = data.metaget(system)
        if block is None:
            continue
        record = block.get(f"{entity}_{axis}")
        if record is not None:
            coordinates[
                system.replace("_DERIVED_GEOMETRY_PARMS", "")
            ][entity][axis] = record['value']
    return coordinates.todict()


def transform_angle(source_frame, target_frame, entity, data):
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
        orientation = target_info['orientation'].lower()
    else:
        orientation = source_info['orientation'].lower()
    return rotate_unit_vector(
        coord['ELEVATION'], coord['AZIMUTH'], quaternion, orientation
    )
