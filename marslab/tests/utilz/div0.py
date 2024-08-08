"""we love dividing by zero"""
import warnings

POINTLESS_WARNINGS = (
    r".*divide by zero",
    r".*invalid value enc",
    r".*All-NaN",
    r".*are nearly identical",
    r".*overflow encountered"
)


def divide_by_zero():
    for msg in POINTLESS_WARNINGS:
        warnings.filterwarnings(message=msg, action="ignore")
