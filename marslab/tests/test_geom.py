from tempfile import NamedTemporaryFile

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

COORD_LABEL = b"""GROUP                             = ROVER_COORD_SYSTEM_PARMS
  MSL:SOLUTION_ID                 = "telemetry"
  COORDINATE_SYSTEM_NAME          = "ROVER_NAV_FRAME"
  COORDINATE_SYSTEM_INDEX         = (63,2086,78,234,0,392,318,162,0,0)
  COORDINATE_SYSTEM_INDEX_NAME    = ("SITE","DRIVE","POSE","ARM","CHIMRA",
                                     "DRILL","RSM","HGA","DRT","IC")
  ORIGIN_OFFSET_VECTOR            = (-85.6879,128.59,-15.1068)
  ORIGIN_ROTATION_QUATERNION      = (0.0404744,0.081473,-0.0150167,-0.99574)
  POSITIVE_AZIMUTH_DIRECTION      = CLOCKWISE
  POSITIVE_ELEVATION_DIRECTION    = UP
  QUATERNION_MEASUREMENT_METHOD   = FINE
  REFERENCE_COORD_SYSTEM_NAME     = "SITE_FRAME"
  REFERENCE_COORD_SYSTEM_INDEX    = 63
END_GROUP                         = ROVER_COORD_SYSTEM_PARMS

GROUP                             = SITE_COORD_SYSTEM_PARMS
  MSL:SOLUTION_ID                 = "telemetry"
  COORDINATE_SYSTEM_NAME          = "SITE_FRAME"
  COORDINATE_SYSTEM_INDEX         = 63
  COORDINATE_SYSTEM_INDEX_NAME    = "SITE"
  ORIGIN_OFFSET_VECTOR            = (-289.148,93.4081,-22.25)
  ORIGIN_ROTATION_QUATERNION      = (1.0,0.0,0.0,0.0)
  POSITIVE_AZIMUTH_DIRECTION      = CLOCKWISE
  POSITIVE_ELEVATION_DIRECTION    = UP
  REFERENCE_COORD_SYSTEM_NAME     = "SITE_FRAME"
  REFERENCE_COORD_SYSTEM_INDEX    = 62
END_GROUP                         = SITE_COORD_SYSTEM_PARMS

GROUP                             = ROVER_DERIVED_GEOMETRY_PARMS
  INSTRUMENT_AZIMUTH              = 38.8131 <deg>
  INSTRUMENT_ELEVATION            = -55.8577 <deg>
  REFERENCE_COORD_SYSTEM_INDEX    = (79,294,6,0,0,0,156,54,0,0)
  REFERENCE_COORD_SYSTEM_NAME     = "ROVER_NAV_FRAME"
  SOLAR_AZIMUTH                   = 73.1638 <deg>
  SOLAR_ELEVATION                 = 81.7842 <deg>
  SUN_VIEW_DIRECTION              = (0.0413893,0.136776,-0.989737)
END_GROUP                         = ROVER_DERIVED_GEOMETRY_PARMS

/* DERIVED GEOMETRY DATA ELEMENTS: SITE FRAME */

GROUP                             = SITE_DERIVED_GEOMETRY_PARMS
  INSTRUMENT_AZIMUTH              = 161.955 <deg>
  INSTRUMENT_ELEVATION            = -32.5527 <deg>
  POSITIVE_AZIMUTH_DIRECTION      = CLOCKWISE
  REFERENCE_COORD_SYSTEM_INDEX    = 79
  REFERENCE_COORD_SYSTEM_NAME     = "SITE_FRAME"
  SOLAR_AZIMUTH                   = 300.641 <deg>
  SOLAR_ELEVATION                 = 65.4502 <deg>
END_GROUP                         = SITE_DERIVED_GEOMETRY_PARMS

END"""


def test_vicarlike_coordinate_system_transformations():
    prod = NamedTemporaryFile()
    prod.write(COORD_LABEL)
    prod.seek(0)
    data = pdr.read(prod.name)
    # perform managed transformation of solar direction vector from ROVER
    # frame to SITE frame
    site_alt, site_az, _ = transform_angle('ROVER', 'SITE', 'SOLAR', data)
    # get the quaternion that will rotate SITE frame to ROVER frame
    reverse_rot = invert_quaternion(
        get_coordinate_system_properties('ROVER', data)['quaternion']
    )
    # rotate the computed solar direction vector back to ROVER frame
    rov_el, rov_az, _ = rotate_unit_vector(site_alt, site_az, reverse_rot)
    # compare it to vector explicitly given in the label
    rov_coords = get_coordinates(data)['ROVER']['SOLAR']
    assert np.isclose(rov_el, rov_coords['ELEVATION'])
    assert np.isclose(rov_az, rov_coords['AZIMUTH'])


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
