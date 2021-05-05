"""
fast, numerical, non-vectorized version of slowtime.py 
eschewing external dependencies other than dateutil.parser

TODO: move / copy references from that module here
"""

import curses
import datetime as dt
from itertools import starmap
from math import cos, floor, sin, pi
from typing import Callable, Union

import dateutil.parser as dtp

TZS = {
    'EST': -18000,
    'CST': -21600,
    'MST': -25200,
    'PST': -28800
}

LEAP_SECOND_THRESHOLDS = [
    dt.datetime(1972, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1972, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1973, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1974, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1975, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1976, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1977, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1978, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1979, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1981, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1982, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1983, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1985, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1987, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1989, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1990, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1992, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1993, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1994, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1995, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(1997, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(1998, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(2005, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(2008, 12, 31, tzinfo=dt.timezone.utc),
    dt.datetime(2012, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(2015, 6, 30, tzinfo=dt.timezone.utc),
    dt.datetime(2016, 12, 31, tzinfo=dt.timezone.utc)
]


# algorithm derived from Julian Date article on scienceworld.wolfram.com
# itself apparently based on Danby, J. M., Fundamentals of Celestial Mechanics
def dt_to_jd(time: dt.datetime) -> float:
    y, m, d = time.year, time.month, time.day
    h = time.hour + time.minute / 60 + time.second / 3600
    return sum([
        367 * y,
        -1 * floor(7 * (y + floor((m + 9) / 12)) / 4),
        -1 * floor(3 * (floor((y + (m - 9) / 7) / 100) + 1) / 4),
        floor(275 * m / 9) + d + 1721028.5,
        h / 24
    ])


def utc_to_tt_offset(time: dt.datetime) -> float:
    """
    return number of seconds necessary to advance UTC
    to TT. we aren't presently supporting dates prior to 1972
    because fractional leap second handling is another thing.
    """
    if time.year < 1972:
        raise ValueError("dates prior to 1972 are not currently supported")
    # this includes the horrible fractional leap seconds prior to 1972
    # and the base 32.184 s offset between TAI and TT
    offset = 42.184
    for threshold in LEAP_SECOND_THRESHOLDS:
        if time > threshold:
            offset += 1
    return offset


def to_jd_tt(time: Union[dt.datetime, str, None] = None) -> float:
    """
    transforms a UTC python datetime -- or any string representing UTC
    time in a format dateutil can parse -- into a julian day number
    in terrestrial time
    """
    if time is None:
        time = dt.datetime.utcnow()
    elif isinstance(time, str):
        time = dtp.parse(time, tzinfos=TZS)
    if time.tzinfo is None:
        time = time.replace(tzinfo=dt.timezone.utc)
    time = time.astimezone(dt.timezone.utc)
    return dt_to_jd(time) + utc_to_tt_offset(time) / 86400


def to_j2000_jd(jd: float) -> float:
    """
    time difference between time and J2000 epoch in julian days.
    """
    return jd - 2451545


def r2d(radian_value: float) -> float:
    """radians to degrees"""
    return radian_value * 180 / pi


def hours_to_24h_time(hours: float) -> str:
    """
    convoluted expression because of Python's...thing to transform a figure
    given in fractional hours to a string in ISO time format
    """
    return (
            dt.datetime(2001, 1, 2) + dt.timedelta(hours=hours)
    ).time().isoformat()


def mean_anomaly(dt_j2000: float) -> float:
    """eq. 16 a&m"""
    return 0.3383687274133927 + 0.009145887087204227 * dt_j2000


def fictitious_mean_sun(dt_j2000: float) -> float:
    """eq. 17 a&m"""
    return 4.719145149919159 + 0.009146197162399134 * dt_j2000


# table 5 (truncated)
PERTURBATION_TABLE = [
    [1.23918377e-04, 2.23530000e+00, 8.62349730e-01],
    [9.94837674e-05, 2.75430000e+00, 2.93517256e+00],
    [6.80678408e-05, 1.11770000e+00, 3.34818728e+00],
    [6.45771823e-05, 1.57866000e+01, 3.79364766e-01],
    [3.66519143e-05, 2.13540000e+00, 2.74086506e-01],
    [3.49065850e-05, 2.46940000e+00, 1.66727813e+00],
    [3.14159265e-05, 3.28493000e+01, 8.56869396e-01]
]


def perturbation_factory(dt_j2000: float) -> Callable[[float, float, float], float]:
    """
    generate the interior of the perturbation summation
    """

    def calculate_perturbation(a: float, tau: float, phi: float) -> float:
        return a * cos(
            0.01720241889326163 * dt_j2000 / tau + phi
        )

    return calculate_perturbation


def pbs(dt_j2000: float) -> float:
    """calculate the sum of perturbations"""
    return sum(starmap(perturbation_factory(dt_j2000), PERTURBATION_TABLE))


def mean_terms(dt_j2000: float) -> float:
    """terms of eq. 19"""
    # writing it in this silly way for readability
    alpha_m = mean_anomaly(dt_j2000)
    return sum([
        (0.1865931503307138 + 5.235987755982988e-09 * dt_j2000) * sin(alpha_m),
        0.010873401239924672 * sin(2 * alpha_m),
        0.0008726646259971648 * sin(3 * alpha_m),
        8.726646259971648e-05 * sin(4 * alpha_m),
        8.726646259971648e-06 * sin(5 * alpha_m)
    ])


# areocentric solar longitude
def l_s(dt_j2000: float) -> float:
    return fictitious_mean_sun(dt_j2000) + mean_terms(dt_j2000) + pbs(dt_j2000)


def eot(dt_j2000: float) -> float:
    l_s_0 = l_s(dt_j2000)
    return sum([
        0.04993386989955777 * sin(2 * l_s_0),
        -0.001239183768915974 * sin(4 * l_s_0),
        3.490658503988659e-05 * sin(6 * l_s_0),
        fictitious_mean_sun(dt_j2000),
        -1 * l_s_0
    ])


def mst_hours(julian_day: float) -> float:
    days = (julian_day - 2451549.5) / 1.0274912517 + 44796.0 - 0.0009626
    return (days * 24) % 24


def lmst_hours(julian_day: float, west_longitude_degrees: float) -> float:
    return mst_hours(julian_day) - west_longitude_degrees * 24 / 360


def ltst_hours(julian_day: float, west_longitude_degrees: float) -> float:
    hours = lmst_hours(julian_day, west_longitude_degrees) + \
            r2d(eot((to_j2000_jd(julian_day)))) / 15
    return hours


def mst(time: Union[str, None, dt.datetime] = None) -> str:
    jd_tt = to_jd_tt(time)
    return hours_to_24h_time(
        mst_hours(
            jd_tt
        )
    )


def lmst(
        time: Union[str, None, dt.datetime] = None,
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
        time: Union[str, None, dt.datetime] = None,
        west_longitude: float = 0
) -> str:
    jd_tt = to_jd_tt(time)
    return hours_to_24h_time(
        ltst_hours(
            jd_tt,
            west_longitude
        )
    )


mission_to_wlon = {
    "Phoenix": 360 - 234.250778,
    "MSL": 360 - 137.4417,
    "InSight": 360 - 135.623,
    "M2020": 360 - 77.5945,
    "AMT": 0,
}

time_type_to_function = {
    'ltst': ltst,
    'lmst': lmst
}


def drawclock(stdscr, mission: str = 'M2020', time_type='ltst'):
    k = 0
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.refresh()
    stdscr.nodelay(True)
    time_function = time_type_to_function[time_type]
    while (k != ord('q')):
        stdscr.addstr(
            1, 1,
            f'{mission} - {time_function(west_longitude=mission_to_wlon[mission])[:8]}'
        )
        stdscr.refresh()
        curses.delay_output(100)
        k = stdscr.getch()

    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()

