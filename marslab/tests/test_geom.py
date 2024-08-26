from pathlib import Path

from hypothesis import given
import numpy as np
import pdr

from marslab.geom import (
    get_coordinates,
    get_coordinate_system_properties,
    invert_quaternion,
    quaternion_multiplication,
    rotate_unit_vector,
    transform_angle,
)
from marslab.tests.utilz.utilz import quaternions


def test_vicarlike_coordinate_system_transformations():
    data = pdr.read(Path(__file__).parent / "data" / "coords.lbl")
    # perform managed transformation of solar direction vector from ROVER
    # frame to SITE frame
    site_alt, site_az, _ = transform_angle("ROVER", "SITE", "SOLAR", data)
    # get the quaternion that will rotate SITE frame to ROVER frame
    reverse_rot = invert_quaternion(
        get_coordinate_system_properties("ROVER", data)["quaternion"]
    )
    # rotate the computed solar direction vector back to ROVER frame
    rov_el, rov_az, _ = rotate_unit_vector(site_alt, site_az, reverse_rot)
    # compare it to vector explicitly given in the label
    rov_coords = get_coordinates(data)["ROVER"]["SOLAR"]
    assert np.isclose(rov_el, rov_coords["ELEVATION"])
    assert np.isclose(rov_az, rov_coords["AZIMUTH"])


@given(a=quaternions, b=quaternions, c=quaternions)
def test_quaternion_multiplication_associativity(a, b, c) -> None:
    left = quaternion_multiplication(
        q1=a, q2=quaternion_multiplication(q1=b, q2=c)
    )
    right = quaternion_multiplication(
        q1=quaternion_multiplication(q1=a, q2=b), q2=c
    )
    assert np.allclose(left, right, atol=1e-5)


@given(quat=quaternions)
def test_quaternion_identity_element(quat) -> None:
    identity = [1, 0, 0, 0]
    assert (quat == quaternion_multiplication(quat, identity)).all()
    assert (quat == quaternion_multiplication(identity, quat)).all()
