"""
tedious symbolic version of time.py, to use as a basis for more complex
calculations.

attempting to roughly match some outputs of mars24, primarily by reimplementing
algorithms in Allison & McEwen 2000. mostly working.
references:
    1. https://www.giss.nasa.gov/tools/mars24/help/algorithm.html
    2. Allison, M. 1997. Accurate analytic representations of solar time and
        seasons on Mars with applications to the Pathfinder/Surveyor missions.
        Geophys. Res. Lett., 24, 1967-1970.
    3. Allison, M., and M. McEwen 2000. A post-Pathfinder evaluation of
    aerocentric solar coordinates with improved
    timing recipes for Mars seasonal/diurnal climate studies. Planet. Space
    Sci., 48, 215-235. (see /static/docs/Allison_2000.pdf)
"""
import datetime as dt
from typing import Union

import astropy.time as at
import dateutil.parser as dtp
import numpy as np
import sympy as sp



#  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
# ▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
# ▐░▌       ▐░▌ ▀▀▀▀█░█▀▀▀▀  ▀▀▀▀█░█▀▀▀▀ ▐░▌           ▀▀▀▀█░█▀▀▀▀  ▀▀▀▀█░█▀▀▀▀  ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀
# ▐░▌       ▐░▌     ▐░▌          ▐░▌     ▐░▌               ▐░▌          ▐░▌          ▐░▌     ▐░▌          ▐░▌
# ▐░▌       ▐░▌     ▐░▌          ▐░▌     ▐░▌               ▐░▌          ▐░▌          ▐░▌     ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄
# ▐░▌       ▐░▌     ▐░▌          ▐░▌     ▐░▌               ▐░▌          ▐░▌          ▐░▌     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
# ▐░▌       ▐░▌     ▐░▌          ▐░▌     ▐░▌               ▐░▌          ▐░▌          ▐░▌     ▐░█▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀█░▌
# ▐░▌       ▐░▌     ▐░▌          ▐░▌     ▐░▌               ▐░▌          ▐░▌          ▐░▌     ▐░▌                    ▐░▌
# ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄█░█▄▄▄▄      ▐░▌      ▄▄▄▄█░█▄▄▄▄ ▐░█▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄█░▌
# ▐░░░░░░░░░░░▌     ▐░▌     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
#  ▀▀▀▀▀▀▀▀▀▀▀       ▀       ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀       ▀       ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀


def parse_time(time: Union[dt.datetime, at.Time, str, float]) -> at.Time:
    """
    attempts to coerce any input to an astropy.Time object.
    assumes float inputs are in julian days.
    """
    if isinstance(time, (dt.datetime, at.Time)):
        return at.Time(time)
    elif isinstance(time, float):
        return at.Time(time, format="jd")
    return at.Time(dtp.parse(time))


def to_jd_tt(time: Union[dt.datetime, at.Time, str, float, None]) -> float:
    if time is None:
        time = at.Time.now()
    return parse_time(time).tt.jd


def to_j2000_jd(time: Union[dt.datetime, at.Time, str, float]) -> float:
    """
    time difference between time and J2000 epoch in julian days,
    and specifically terrestrial time
    this value (delta_j2000 and variants below) is a basic
    time unit for many common expressions in Mars horology
    """
    time = parse_time(time)
    return (at.Time(time) - at.Time('2000-01-01T12:00:00')).tt.jd


def d2r(degree_value: float) -> float:
    """degrees to radians"""
    return degree_value * np.pi / 180


def r2d(radian_value: float) -> float:
    """radians to degrees"""
    return radian_value * 180 / np.pi


def hours_to_24h_time(hours: float) -> str:
    """
    convoluted expression because of Python's...thing to transform a figure
    given in fractional hours to a string in ISO time format
    """
    return (
            dt.datetime(2001, 1, 2) + dt.timedelta(hours=hours)
    ).time().isoformat()


#  ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄
# ▐░░▌     ▐░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
# ▐░▌░▌   ▐░▐░▌▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀
# ▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌
# ▐░▌ ▐░▐░▌ ▐░▌▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄
# ▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌     ▐░▌     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
# ▐░▌   ▀   ▐░▌▐░█▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀▀▀▀▀▀█░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌          ▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌ ▄▄▄▄▄▄▄▄▄█░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░░░░░░░░░░░▌
#  ▀         ▀  ▀         ▀       ▀       ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀

# all equation / table numbers in this section of the code
# refer to Allison & McEwen 2000

# determine mars mean anomaly, fictitious mean sun angle,
# and sum of angular perturbations in longitude

# TODO: it's not incredibly useful as it stands to define these
#  as sympy expressions, but i'm anticipating further applications
#  where it may be. I don't think it's significantly hurting
#  performance _other than imports_ because we immediately lambdify
#  them for evaluation. if we are building a quick command-line utility
#  and it appears to be discernibly laggy, or people want to run these on
#  > 1000 data points at once, we may want to try slimming this down.

# this is a free symbol that will generally represent an
# output of j2000_jd
delta_j2000 = sp.S('Delta_j2000')

# eq. 16
# note that all angular values in A&M are
# in degrees; we convert them to radians
mean_anomaly = d2r(19.3871) + d2r(0.52402073) * delta_j2000

# eq 17
fictitious_mean_sun = d2r(270.3871) + d2r(0.524038496) * delta_j2000

# table 5 (truncated)
perturbation_table = np.array([
    [d2r(0.0071), 2.2353, d2r(49.409)],
    [d2r(0.0057), 2.7543, d2r(168.173)],
    [d2r(0.0039), 1.1177, d2r(191.837)],
    [d2r(0.0037), 15.7866, d2r(21.736)],
    [d2r(0.0021), 2.1354, d2r(15.704)],
    [d2r(0.0020), 2.4694, d2r(95.528)],
    [d2r(0.0018), 32.8493, d2r(49.095)]
])

# free symbols representing terms of perturbation function
# / columns of perturbation table
A, tau, phi = sp.S('A, tau, phi')

# eq 18, interior of summation
PBS_i = A * sp.cos(
    d2r(0.985626) * delta_j2000 / tau + phi
)


# evaluation function for this sum
def pbs(dt_j2000: float) -> np.ndarray:
    row_lambda = sp.lambdify([delta_j2000, A, tau, phi], PBS_i)
    return np.sum([
        row_lambda(dt_j2000, *perturbation_table[i])
        for i in range(perturbation_table.shape[0])
    ])


# terms of equation 19:
mean_terms = (d2r(10.691) + d2r(3e-7) * delta_j2000) * sp.sin(mean_anomaly) + \
             d2r(0.623) * sp.sin(2 * mean_anomaly) + \
             d2r(0.050) * sp.sin(3 * mean_anomaly) + \
             d2r(0.005) * sp.sin(4 * mean_anomaly) + \
             d2r(0.0005) * sp.sin(5 * mean_anomaly)


# evaluation function: areocentric solar longitude
def l_s(dt_j2000: float) -> float:
    return sp.lambdify(delta_j2000, fictitious_mean_sun)(dt_j2000) + \
           sp.lambdify(delta_j2000, mean_terms)(dt_j2000) + pbs(dt_j2000)


# note that areocentric solar longitude is given in Mars24;
# can compare, for instance:
# r2d(L_s(j2000_jd(at.Time.now()))) % 360

# eq 20
def eot(dt_j2000: float) -> float:
    """Allison's 'equation of time'"""
    # just expressing this one numerically for now
    return d2r(2.861) * np.sin(2 * l_s(dt_j2000)) - \
           d2r(0.071) * np.sin(4 * l_s(dt_j2000)) + \
           d2r(0.002) * np.sin(6 * l_s(dt_j2000)) + \
           sp.lambdify(delta_j2000, fictitious_mean_sun)(dt_j2000) - \
           l_s(dt_j2000)
    # these final two terms are -(v-m), additive inverse
    # of equation of center


# ******* cutting to the chase, the simplistic useful expressions *******

# NOTE 1: we _stop_ modifying to radians here, because these will be public
# quite shortly and aren't using any more transcendental functions
# NOTE 2: following Allison & McEwan, we _also_ stop using delta_j2000
# and switch to conventionally-defined jd: but still in TT!

# eq 22, modified constants (see giss.nasa.gov link, heading C-2)
def mst_hours(julian_day: float) -> float:
    days = (julian_day - 2451549.5) / 1.0274912517 + 44796.0 - 0.0009626
    return (days * 24) % 24


# eq. 23
def lmst_hours(julian_day: float, west_longitude_degrees: float) -> float:
    return mst_hours(julian_day) - west_longitude_degrees * 24 / 360


def ltst_hours(julian_day: float, west_longitude_degrees: float) -> float:
    hours = lmst_hours(julian_day, west_longitude_degrees) + \
            r2d(eot((to_j2000_jd(at.Time(julian_day, format='jd'))))) / 15
    return hours


#  ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄
# ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
# ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌ ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌ ▀▀▀▀█░█▀▀▀▀
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌     ▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░░░░░░░░░░░▌▐░▌       ▐░▌     ▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░█▀▀▀▀▀▀▀▀▀ ▐░▌       ▐░▌     ▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌          ▐░▌       ▐░▌     ▐░▌
# ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░▌          ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌
# ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌     ▐░▌          ▐░░░░░░░░░░░▌     ▐░▌
#  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀       ▀       ▀            ▀▀▀▀▀▀▀▀▀▀▀       ▀

# end-user wrapper functions


def mst(time: Union[at.Time, float, str, None, dt.datetime] = None) -> str:
    jd_tt = to_jd_tt(time)
    return hours_to_24h_time(
        mst_hours(
            jd_tt
        )
    )

def lmst(
        time: Union[at.Time, float, str, None, dt.datetime] = None,
        west_longitude: float = 0
) -> str:
    jd_tt = to_jd_tt(time)
    return hours_to_24h_time(
        lmst_hours(
            jd_tt,
            west_longitude
        )
    )

def ltst(
        time: Union[at.Time, float, str, None, dt.datetime] = None,
        west_longitude: float = 0
) -> str:
    jd_tt = to_jd_tt(time)
    return hours_to_24h_time(
        ltst_hours(
            jd_tt,
            west_longitude
        )
    )





