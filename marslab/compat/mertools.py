"""
Constants and utility functions for compatibility with MERTools.

MERTools is an IDL software suite used for tactical data assessment and derived
product generation. It is the exclusive intellectual property of Arizona State
University (PI J. Bell) and made available on an as-needed basis.
"""

from itertools import chain

import numpy as np
import pandas as pd

from marslab.compat.xcam import (
    make_xcam_filter_dict,
    WAVELENGTH_TO_FILTER,
)

# mappings between MERspect color designations and hex codes
MERSPECT_M20_COLOR_MAPPINGS = {
    "green": "#00ff3b",  # not a creative color
    "yellow": "#eeff00",
    "blue": "#1500ff",
    "red": "#ff0000",
    "magenta": "#ff00ae",
    "cyan": "#00ffee",
    "orange": "#ffa100",
    "azure": "#0094ff",
    "purple": "#ad0bbf",
    "lime": "#9bec11",
    "rust": "#dd5b10",
    "green+2": "#7fff9d",
    "green-1": "#00b229",
    "green-2": "#007f1d",
    "yellow-2": "#777f00",
    "blue+2": "#8a7fff",
    "blue-1": "#0e00b2",
    "blue-2": "#0a007f",
    "red+2": "#ff7f7f",
    "red-1": "#b20000",
    "red-2": "#7f0000",
    "magenta+2": "#ff7fd6",
    "magenta+1": "#ff4cc6",
    "magenta-1": "#b20079",
    "magenta-2": "#7f0057",
    "magenta-3": "#4c0034",
    "cyan+2": "#a6fff9",
    "cyan+1": "#73fff6",
    "cyan-1": "#00b2a6",
    "cyan-2": "#007f77",
    "cyan-3": "#004c47",
    "orange+2": "#ffd07f",
    "orange+1": "#ffbd4c",
    "orange-1": "#b27100",
    "orange-2": "#7f5000",
    "orange-3": "#4c3000",
    "azure+2": "#7fc9ff",
    "azure+1": "#4cb4ff",
    "azure-1": "#0068b2",
    "azure-2": "#004a7f",
    "azure-3": "#002c4c",
}
# The colors below here are from the MCAM / PCAM legacy functionality

MERSPECT_MSL_COLOR_MAPPINGS = {
    "red": "#dc143c",  # duplicated from above w/ different hex code
    "light green": "#7fff00",
    "light blue": "#0000ff",
    "light cyan": "#00ffff",
    "dark green": "#128012",
    "yellow": "#ffff00",  # duplicated from above w/ different hex code
    "light purple": "#ff00ff",
    "pink": "#fa8072",
    "teal": "#008080",
    "goldenrod": "#B8860b",
    "sienna": "#a0522d",
    "dark blue": "#000080",
    "bright red": "#ff0000",
    "dark red": "#800000",
    "dark purple": "#800080",
}

MERSPECT_COLOR_MAPPINGS = {
    **MERSPECT_M20_COLOR_MAPPINGS,
    **MERSPECT_MSL_COLOR_MAPPINGS,
}

# Standard assignments of MERSpect ROI colors to feature classes.
# for the Western Washington University Reflectance Lab (PI M. Rice).
# No such mapping has yet been agreed upon for ZCAM.
COLOR_TO_FEATURE_TYPE = {
    "MCAM": {
        "light purple": "drill tailings",
        "dark purple": "dump piles",
        "light blue": "dusty rock",
        "teal": "dusty rock",
        "dark blue": "dusty rock",
        "light green": "DRT target",
        "dark green": "broken rock face",
        "bright red": "undisturbed soil",
        "dark red": "disturbed soil",
        "red": "undisturbed soil",
        "light cyan": "nodule-rich rock",
        "goldenrod": "veins",
        "sienna": None,
        "pink": None,
        "yellow": None,
    },
    "ZCAM": {
        "green": None,
        "yellow": None,
        "blue": None,
        "red": None,
        "magenta": None,
        "cyan": None,
        "orange": None,
        "azure": None,
        "purple": None,
        "lime": None,
        "rust": None,
        "green+2": None,
        "green-1": None,
        "green-2": None,
        "yellow-2": None,
        "blue+2": None,
        "blue-1": None,
        "blue-2": None,
        "red+2": None,
        "red-1": None,
        "red-2": None,
        "magenta+2": None,
        "magenta+1": None,
        "magenta-1": None,
        "magenta-2": None,
        "magenta-3": None,
        "cyan+2": None,
        "cyan+1": None,
        "cyan-1": None,
        "cyan-2": None,
        "cyan-3": None,
        "orange+2": None,
        "orange+1": None,
        "orange-1": None,
        "orange-2": None,
        "orange-3": None,
        "azure+2": None,
        "azure+1": None,
        "azure-1": None,
        "azure-2": None,
        "azure-3": None,
    },
}


def parse_merspect_fn(fn):
    # Parse the MERSpect filename for obs information
    sol = int(fn.split("/")[-1].split("_")[0][3:])
    instrument = str.upper(fn.split("/")[-1].split("_")[1][:4])
    seq_id = fn.split("/")[-1].split("_")[1]
    return {"SOL": sol, "INSTRUMENT": instrument, "SEQ_ID": seq_id}


def get_unique_colors(csv, instrument):
    # Figure out the unique color names contained in the MERSpect output file
    colors = []
    for k in csv.keys():
        keycolor = " ".join(k.strip().split(" ")[:-2])
        for c in COLOR_TO_FEATURE_TYPE[instrument].keys():
            if keycolor == c:
                colors += [c]
    return np.unique(colors)


def merspect_to_marslab(fn, write=True):
    csv = pd.read_csv(fn)
    # TODO: find a different way to get the sol, seq_id, instrument from the
    #  file because this isn't actually standardized within MERspect, only
    #  Melissa's lab
    #  -- or, make a clean version that doesn't even try to parse this
    #  because it's never going to be called directly

    # Get the sol, seq_id, instrument from the filename
    try:
        obsparams = parse_merspect_fn(fn)
    # someone has named this something weird -- try to parse the data anyway
    except (ValueError, KeyError):
        obsparams = {"SOL": "-", "INSTRUMENT": "ZCAM", "SEQ_ID": "-"}
    # Rename "# Wavelength (nm)" to "Wavelength (nm)"
    csv.rename(columns={"# Wavelength (nm)": "Wavelength"}, inplace=True)
    # Clean up column names
    [csv.rename(columns={k: k.strip()}, inplace=True) for k in csv.keys()]

    # We want the columns in order of ascending wavelength, regardless of
    # instrument
    columns = {
        "MCAM": [
            "SOL",
            "SEQ_ID",
            "INSTRUMENT",
            "COLOR",
            "FEATURE",
            "FORMATION",
            "MEMBER",
            "FLOAT",
            *list(
                chain.from_iterable(
                    [
                        (filt, filt + "_ERR")
                        for filt in make_xcam_filter_dict("MCAM").keys()
                    ]
                )
            ),
        ],
        "ZCAM": [
            "SOL",
            "SEQ_ID",
            "INSTRUMENT",
            "COLOR",
            "FEATURE",
            "FORMATION",
            "MEMBER",
            "FLOAT",
            "RMS",
            "ZOOM",
            *list(
                chain.from_iterable(
                    [
                        (filt, filt + "_ERR")
                        for filt in make_xcam_filter_dict("ZCAM").keys()
                    ]
                )
            ),
        ],
    }[obsparams["INSTRUMENT"]]

    # Init the dataframe
    data = pd.DataFrame(columns=columns)
    instrument = obsparams["INSTRUMENT"]

    # Generate the index column --- 'COLOR'
    colors = get_unique_colors(csv, instrument)
    data["COLOR"] = colors
    data = data.set_index("COLOR")

    for color in colors:
        this_color = csv[
            [
                "Eye",
                "Wavelength",
                f"{color} Mean Value",
                f"{color} Standard Deviation",
            ]
        ]
        feature = COLOR_TO_FEATURE_TYPE[instrument].get(color)
        for i in range(len(this_color)):
            try:
                eye = this_color.loc[i]["Eye"].strip()[0]
                wavelength = int(this_color.loc[i]["Wavelength"])
                filt = WAVELENGTH_TO_FILTER[instrument][eye][wavelength]
                data[f"{filt}"].loc[color] = this_color.loc[i][
                    f"{color} Mean Value"
                ]
                data[f"{filt}_ERR"].loc[color] = this_color.loc[i][
                    f"{color} Standard Deviation"
                ]
                if feature is not None:
                    data["FEATURE"].loc[color] = feature
            except AttributeError:  # empty entries
                try:
                    if (
                        this_color.iloc[i]["Wavelength"].strip() == "Notes"
                    ) and (feature is None):
                        data["FEATURE"].loc[color] = this_color.iloc[i][
                            f"{color} Mean Value"
                        ]
                    elif this_color.iloc[i]["Wavelength"].strip() == "Float":
                        data["FLOAT"].loc[color] = this_color.iloc[i][
                            f"{color} Mean Value"
                        ]
                    elif (
                        this_color.iloc[i]["Wavelength"].strip() == "Formation"
                    ):
                        data["FORMATION"].loc[color] = this_color.iloc[i][
                            f"{color} Mean Value"
                        ]
                    elif this_color.iloc[i]["Wavelength"].strip() == "Member":
                        data["MEMBER"].loc[color] = this_color.iloc[i][
                            f"{color} Mean Value"
                        ]
                # TODO: what was this for?
                except:
                    pass

    # Set values for constant columns

    data["INSTRUMENT"] = obsparams["INSTRUMENT"]
    data["SOL"] = obsparams["SOL"]
    data["SEQ_ID"] = obsparams["SEQ_ID"]
    # Clean up for consistency
    data["FORMATION"].replace("Murray ", "Murray", inplace=True)
    data["FORMATION"].replace("Kiimberly", "Kimberly", inplace=True)
    data["MEMBER"].replace(
        "Knockfarril Hill ", "Knockfarril Hill", inplace=True
    )
    # Clean up some entries for human-readability
    data["FORMATION"].replace(np.nan, "-", inplace=True)
    data["MEMBER"].replace(np.nan, "-", inplace=True)
    data["FLOAT"].replace(np.nan, "N", inplace=True)
    data["FLOAT"].replace("x", "Y", inplace=True)
    data["FLOAT"].replace("float", "N", inplace=True)
    data["FEATURE"].replace(np.nan, "-", inplace=True)

    data.reset_index(inplace=True)
    data = data[columns]

    fn = fn.replace("-BEFORE", "")  # for test files
    outfile = fn.replace(".csv", "-marslab.csv")
    if write:
        data.to_csv(outfile, index=False)
    return data


def roi_color_ix_to_color_name(color_ix: int, instrument: str = "MCAM") -> str:
    """
    convert MERSpect ROI color index to color name
    """
    # this works because colors are listed in the dictionaries in
    # marslab.compat.mertools in the order
    # they are listed in the MERSpect IDL code. No explicit mapping
    # between MERSpect ROI color indices and actual colors exists.
    if instrument in ["MCAM", "PCAM"]:
        return list(MERSPECT_MSL_COLOR_MAPPINGS.keys())[color_ix]
    elif instrument == "ZCAM":
        # The unit offset to the color indices is because "0" is reserved
        # in MERspect as something like "blank canvas." The "erase"
        # capability has a different color associated with it.
        return list(MERSPECT_M20_COLOR_MAPPINGS.keys())[color_ix - 1]
    else:
        raise ValueError(
            "I don't have ROI color information about "
            + instrument
            + ". I know about PCAM, MCAM, and ZCAM."
        )


def add_merspect_colors_to_edgemaps(edgemap_dict):
    for roi_color_eye, edge in edgemap_dict.items():
        color = roi_color_eye.replace(" right", "").replace(" left", "")
        edgemap_dict[roi_color_eye]["color"] = MERSPECT_COLOR_MAPPINGS[color]
    return edgemap_dict
