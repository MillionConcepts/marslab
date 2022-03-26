# Functions for parsing the metadata out of Mars 2020 observational data filenames

# This code is derived exclusively from information in
# "D-99960 M2020 Camera EDR/RDR Data Products SIS Version 2.0 (Draft)"
# (https://pds-imaging.jpl.nasa.gov/reviews/mars2020/mars2020_mission/document_camera/Mars2020_Camera_SIS.pdf)

import string
import itertools

import pandas as pd


def suite(inst):
    return


def instrument(fn, ptype="IMAGE"):
    # Valid values of ptype are ["IMAGE","MOSAIC1","MOSAIC2","MESH"]
    if ptype == "IMAGE":
        return {
            "FL": "Front Hazcam Left (RCE-A)",
            "FR": "Front Hazcam Right (RCE-A)",
            "FA": "Front Hazcam Anaglyph (RCE-A)",
            "FG": "Front Hazcam Colorglyph (RCE-A)",
            "BL": "Front Hazcam Left (RCE-B)",
            "BR": "Front Hazcam Right (RCE-B)",
            "BA": "Front Hazcam Anaglyph (RCE-B)",
            "BG": "Front Hazcam Colorglyph (RCE-B)",
            "CC": "Cache Cam",
            "EA": "EDL Parachute Uplook Cam A (PUC-A)",
            "EB": "EDL Parachute Uplook Cam B (PUC-B)",
            "EC": "EDL Parachute Uplook Cam C (PUC-C)",
            "ED": "EDL Rover Downlook Cam (RDC)",
            "EL": "EDL Lander Vision System (LVS)",
            "ES": "EDL Descent Stage Downlook Cam (DSD)",
            "EU": "EDL Rover Uplook Cam (RUC)",
            "HN": "Helicopter Navigation Cam",
            "HS": "Helicopter Return To Earth Cam",
            "ZL": "Mastcam-Z Left",
            "ZR": "Mastcam-Z Right",
            "ZA": "Mastcam Anaglyph",
            "ZG": "Mastcam Colorglyph",
            "WS": "MEDA Skycam",
            "NL": "Navcam Left",
            "NR": "Navcam Right",
            "NA": "Navcam Anaglyph",
            "NG": "Navcam Colorglyph",
            "PC": "PIXL Micro Context Cam (MCC)",
            "RL": "Rear Hazcam Left",
            "RR": "Rear Hazcam Right",
            "RA": "Rear Hazcam Anaglyph",
            "RG": "Rear Hazcam Colorglyph",
            "SC": "SHERLOC Context Imager (ACI)",
            "SE": "SHERLOC Engineering (Imagers)",
            "SI": "SHERLOC Imaging (Watson)",
            "SA": "SHERLOC Imaging (Watson) Anaglyph",
            "SG": "SHERLOC Imaging (Watson) Colorglyph",
            "SL": "Watson treated as left-eye",
            "SR": "Watson treated as right-eye",
            "LR": "Supercam RMI",
            "WE": "MEDA Environment",
            "OX": "MOXIE",
            "PE": "PIXL Engineering",
            "PS": "PIXL Spectrometer",
            "LS": "SuperCam Non-Imaging Data",
            "SS": "SHERLOC Spectrometer",
            "XM": "RIMFAX Mobile",
            "XS": "RIMFAX Stationary",
        }[fn[0:2]]
    elif ptype.startswith("MOSAIC"):
        return {
            "N": "Navcam",
            "Z": "Mastcam-Z",
            "L": "Supercam RMI",
            "P": "PIXL Micro Context Cam (MCC)",
            "I": "SHERLOC Imaging (WATSON)",
            "C": "SHERLOC ACI",
            "F": "Front Hazcam (RCE-A)",
            "B": "Front Hazcam (RCE-B)",
            "R": "Rear Hazcam",
            "E": "EDL Camera",
            "V": "Mars Helicopter Navigation Cam",
            "X": "Use when there are more than two instruments",
            "_": "N/A",  # Use when there is only one instrument
        }[fn[0:1] if ptype.endswith("1") else fn[1:2]]
    elif ptype == "MESH":
        return "TBD"
    else:
        raise (f"Unknown filename type: {ptype}")


def color_filter(fn, ptype="IMAGE"):
    # Only “E”, “F”, or “M” or Filter/LED can appear in EDRs.
    if ptype == "IMAGE":
        if "PIXL" in instrument(fn):
            return {
                "R": "LED Red",
                "G": "LED Green",
                "B": "LED Blue",
                "W": "LED Multiple",
                "U": "LED UV",
                "D": "LED SLI-A (Dense)",
                "S": "LED SLI-B (Sparse)",
                "_": "LED Off",
            }[fn[2:3]]
        elif ("Mastcam" in instrument(fn)) and (fn[2:3] in "01234567"):

            return {
                "L0": "L0 (530nm)",
                "L1": "L1 (800nm)",
                "L2": "L2 (754nm)",
                "L3": "L3 (677nm)",
                "L4": "L4 (605nm)",
                "L5": "L5 (528nm)",
                "L6": "L6 (442nm)",
                "L7": "L7 (Solar)",
                "R0": "R0 (530nm)",
                "R1": "R1 (800nm)",
                "R2": "R2 (866nm)",
                "R3": "R3 (910nm)",
                "R4": "R4 (939nm)",
                "R5": "R5 (978nm)",
                "R6": "R6 (1022nm)",
                "R7": "R7 (Solar)",
            }[f"{fn[1:2]}{fn[2:3]}"]
        elif "SHERLOC" in instrument(fn):
            return {
                "0": "cover closed, LEDs off",
                "1": "cover open, LEDs on",
                "2": "cover closed, LEDs off",
                "3": "cover open, LEDs on",
            }[fn[2:3]]
        else:
            return {
                "E": "Raw Bayer pattern",
                "M": "Grayscale",  # (Monochrome/Panchromatic)",
                "A": "Upper Green Bayer",  # cells",
                "D": "Lower Green Bayer",  # cells",
                "O": "Other",
                "_": "N/A",
                "F": "3-Band RGB",
                "R": "Band 1 RGB",
                "G": "Band 2 RGB",
                "B": "Band 3 RGB",
                "T": "3-Band XYZ",
                "X": "Band 1 XYZ",
                "Y": "Band 2 XYZ",
                "Z": "Band 3 XYZ",
                "C": "3-Band xyY",
                "J": "Band 1 xyY",
                "K": "Band 2 xyY",
                "L": "Band 3 xyY",
                "P": "3-Band HSI",
                "H": "Band 1 HSI",
                "S": "Band 2 HSI",
                "I": "Band 3 HSI",
            }[fn[2:3]]
    elif ptype == "MOSAIC":
        return {
            "0": "N/A",
            "_": "N/A",
            "M": "Grayscale",  # (Monochrome/Panchromatic)",
            "U": "UV (PIXL)",
            "F": "3-Band RGB",
            "R": "Band 1 RGB",
            "G": "Band 2 RGB",
            "B": "Band 3 RGB",
            "T": "3-Band XYZ",
            "X": "Band 1 XYZ",
            "Y": "Band 2 XYZ",
            "Z": "Band 3 XYZ",
            "C": "3-Band xyY",
            "J": "Band 1 xyY",
            "K": "Band 2 xyY",
            "L": "Band 3 xyY",
            "P": "3-Band HSI",
            "H": "Band 1 HSI",
            "S": "Band 2 HSI",
            "I": "Band 3 HSI",
        }[fn[3:6]]


def special_flag(fn, ptype="IMAGE"):
    if ptype == "IMAGE":
        return "N/A" if fn[3:4] == "_" else [fn[3:4]]
    elif ptype == "MOSAIC":
        return "Nominal" if fn[6:7] == "_" else fn[6:7]
    else:
        raise (f"Unknown filename type: {ptype}")


def sol(fn, ptype="IMAGE"):
    if ptype == "IMAGE":
        if fn[4:8][0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":  # Flight or ground test
            return f"{fn[4:8]} - TRANSLATION TO BE DONE"
        else:
            return int(fn[4:8])  # Either flight or test Sol number
    elif ptype == "MOSAIC":
        return "Nominal" if fn[6:7] == "_" else fn[6:7]


def venue(fn, ptype="IMAGE"):
    return {
        "_": "Flight",  # (surface or cruise)
        "A": "AVSTB",
        "F": "FSWTB",
        "M": "MSTB",
        "R": "ROASTT",
        "S": "Scarecrow",
        "V": "VSTB",
        "T": "?ORT?",  # Strange option in PDS review data
    }[{"IMAGE": fn[8:9], "MOSAIC": fn[15:16]}[ptype]]


def secondary_timestamp(fn):  # fn[9:19]) # SCLK
    return int(fn[9:19])
    # Note that there is a special calculation to dial this in precisely
    #  for sequences of ZCAM and SHERLOCK images which will
    #  all have the same SCLK as the first in the sequence


def tertiary_timestamp(fn):  # fn[20:23]) # SCLK Milliseconds
    return int(fn[20:23])


def product_type(fn, ptype="IMAGE"):  # fn[23:26])
    val = {"IMAGE": fn[23:26], "MOSAIC": fn[12:15]}[ptype]
    return val


def parse_product_type(fn, ptype="IMAGE"):
    # There is a HUGE lookup table of possible values
    #   which might be implemented here.
    # See: M2020 camera image EDR/RDR product binary formats
    # I have only implemented the ones coded "GREEN" for now
    # (being the standard products to be produced).
    try:
        return {
            "ECM": "Original image product (possibly companded)",
            "EJP": "Original JPEG as recieved from the rover",
            "ECV": "Single frame of a video (possibly companded)",
            "ECZ": "Z-stack frame (possibly companded)",
            "EDM": "Depth map",
            "ECR": "Recovered product (possibly companded)",
            "ERP": "Reference pixel",
            "ERS": "Row summed",
            "ECS": "Column summed",
            "EHG": "Histogram",
            "EAU": "Audio",
            "EDR": "Image (decompanded)",
            "EVD": "Single frame of a video (decompanded)",
            "EZS": "Z-stack frame (decompanded)",
            "ERD": "Recovered product (decompanded)",
            "EBY": "Debayered (demosaicked)",
            "FDR": "Base image from which RDRs are derived",
            "TDR": "Tile data record",
            "FAU": "Audio reconstructed",
            "IOF": "Rad-corrected IOF radiance factor (float)",
            "IOI": "Rad-corrected IOF radiance factor (integer)",
            "RAF": "Rad-corrected (absolute units, float)",
            "RAD": "Rad-corrected (absolute units, 15-bit int static scale)",
            "RAS": "Rad-corrected (absolute units, 12-big int static scale)",
            "RAY": "Rad-corrected (absolute units, 15-bit int dynamic scale)",
            "RDM": "Rad-corrected (absolute units, 15-bit int static scale, masked)",
            "RIE": "Rad corrected for instrument effects only (int DN)",
            "RIF": "Rad-corrected for instrument effects only (float)",
            "RSM": "Rad-corrected (absolute units, 12-big int static scale, masked)",
            "RAG": "Gamma-corrected radiance (no longer calibrated units, byte)",
            "RZF": "Zenith-scaled radiance (float)",
            "RZD": "Zenith-scaled radiance (15-bit int static scale)",
            "RZS": "Zenith-scaled raidance (12-bit int static scale)",
            "RZY": "Zenith-scaled radiance (15-bit int dynamic scale)",
            ### TODO: ALL OF THE "COLOR" PRODUCTS ARE SKIPPED HERE
            "DDD": "Stereo delta disparity (2-band,true disparity offset",
            "DDL": "Stereo delta disparity of lines (single-band)",
            "DDS": "Stereo delta disparity of samples (single-band)",
            "DSE": "Stereo disparity error metric",
            "DSL": "Stereo disparity of lines (single-band)",
            "DSS": "Stereo disparity of sample (single-band)",
            "DSP": "Stereo disparity (final)",
            "DSR": "Stereo disparity (raw)",
            "MDS": "Stereo disparity mask",
            "DFF": "Stereo first-stage disparity (final)",
            "DFL": "Stereo first-stage disparity (line)",
            "DFS": "Stereo first-stage disparity (sample)",
            "XYZ": "XYZ in Site frame",
            "XYM": "XYZ in Site frame (masked)",
            "XYR": "XYZ in Rover Nav frame",
            "ZRM": "XYZ in Rover Nav frame (masked)",
            "XYE": "XYZ error metric",
            "MXY": "XYZ mask",
            "XYF": "XYZ (filled)",
            "XYO": "XYZ w/ overlay",
            "XRZ": "XYZ from disparity raw expressed in Site frame",
            "XRR": "XYZ from disparity raw expressed in Rover Nav frame",
            "XXX": "XYZ X-band",
            "YYY": "XYZ Y-band",
            "ZZZ": "XYZ Z-band",
            "DEM": "Digital elevation model",
            "XOZ": "Orbital XYZ",
            "XXF": "XYZ X-band (filled)",
            "YYF": "XYZ Y-band (filled)",
            "ZZF": "XYZ Z-band (filled)",
            "ZIH": "Instrument Z-value for helicoptor",
            "RNE": "Range error metric",
            "RNF": "Range (filled)",
            "RNG": "Range from camera",
            "RNM": "Range from camera (masked)",
            "RNR": "Range from rover origin",
            "RNO": "Range w/ overlay",
            "UVF": "Surface normal (filled)",
            "UVO": "Surface normal w/ overlay",
            "UVP": "Surface normal projected onto plane",
            "UVS": "Surfaace normal for slope computations",
            "UVT": "Surface normal angle (theta) between normal and plane",
            "UVW": "Surface normal",
            "UUU": "Surface normal U-band",
            "VVV": "Surface normal V-band",
            "WWW": "Surface normal W-band",
            "UUF": "Surface normal U-band filled",
            "VVF": "Surface normal V-band filled",
            "WWF": "Surface normal W-band filled",
            "UIH": "Instrument normal for Helicoptor",
            "TFH": "Tilt for Helicoptor",
            "SEN": "Solar energy",
            "SHD": "Slope heading",
            "SHO": "Slope heading w/ overlay",
            "SLO": "Slope w/ overlay",
            "SLP": "Slope",
            "SMG": "Slope magnitude",
            "SMO": "Slope magnitude w/ overlay",
            "SNO": "Slope northerly tilt w/ overlay",
            "SNT": "Slope northerly tilt",
            "SRD": "Slope radial direction",
            "ARK": "Arm reachability (masked)",
            "ARM": "Arm reachability",
            "ARO": "Arm reachability w/ overlay",
            "MAR": "Arm reachability mask",
            "RUF": "Surface roughness (general, not instrument-specific)",
            "RUS": "Surface roughness (for Drill Stabilizer and Coring)",
            "RUC": "Surface curvature for Drill",
            "RUH": "Surface roughness for Helicoptor",
            "GUN": "Goodness (overall for Natural Surface Tools)",
            "GUA": "Goodness (overall for Abrading Tools)",
            "GUC": "Goodness (overall for Coring Tools)",
            "GRN": "Goodness (for reachability of Natural Surface Tools)",
            "GRA": "Goodness (for reachability of Abrading Tools)",
            "GRC": "Goodness (for reachability of Coring Tools)",
            "GSR": "Goodness (combined for surface roughness)",
            "GUH": "Goodness (Helicoptor)",
            "SLH": "Slope goodness for helicoptor",
            "MSK": "Image mask",
            "TEN": "Terrain class confidence",
            "TER": "Terrain class",
            "IEF": "Incidence, emission, and phase angles (filled)",
            "IEP": "Incidence, emission, and phase angles",
            "IDM": "Index depth map",
            "IDX": "Mosaic input index",
            "ICM": "Mosaic input coregistration map",
        }[product_type(fn, ptype=ptype)]
    except:
        return product_type(fn, ptype=ptype)


def geometry(fn):  # fn[26:27])
    return {
        "_": "N",  # "Non-linearized (raw geometry)",
        "L": "Y",  # "Product has been linearized with nominal stereo partner",
        "A": "Y",  # "Product has been linearized with an actual stereo partner",
    }[fn[26:27]]


def thumbnail(fn):  # fn[27:28])
    return {
        "T": "Y",  # "Product is a thumbnail",
        "N": "N",  # "Nominal Product is a non-thumbnail (full-frame, sub-frame, downsample)",
    }[fn[27:28]]


def site_counter():
    # generates a lookup table per p185 of the SIS
    # 0 - 32767 inclusively, so total 32768
    # last valid is 7DV
    digits, ascii_uppercase = string.digits, string.ascii_uppercase
    collection = (
        itertools.product(digits, repeat=3),
        itertools.product(ascii_uppercase, digits, digits),
        itertools.product(ascii_uppercase, ascii_uppercase, digits),
        itertools.product(ascii_uppercase, repeat=3),
        itertools.product(digits[0:7], ascii_uppercase, ascii_uppercase),
    )
    table = {
        k: v
        for v, k in enumerate(
            map("".join, itertools.islice(itertools.chain(*collection), 32768))
        )
    }
    return table


# greedily evaluate at import time for efficiency
SITE_LOOKUP_TABLE = site_counter()


def site(fn, ptype="IMAGE"):  # fn[28:31])
    val = {"IMAGE": fn[28:31], "MOSAIC": fn[16:19]}[ptype]
    if val == "___":
        return "OUT OF RANGE"
    return SITE_LOOKUP_TABLE[val]


def drive_counter():
    # generates a lookup table per p185-186 of the SIS,
    # 0-65535 inclusively, so total 65536
    # last valid is LJ35
    digits, ascii_uppercase = string.digits, string.ascii_uppercase
    collection = (
        itertools.product(digits, repeat=4),
        itertools.product(ascii_uppercase, digits, digits, digits),
        itertools.product(
            ascii_uppercase[0:12], ascii_uppercase, digits, digits
        ),
    )
    table = {
        k: v
        for v, k in enumerate(
            map("".join, itertools.islice(itertools.chain(*collection), 65536))
        )
    }
    return table


# greedily evaluate at import time for efficiency
DRIVE_LOOKUP_TABLE = drive_counter()


def drive(fn, ptype="IMAGE"):  # fn[31:35])
    val = {"IMAGE": fn[31:35], "MOSAIC": fn[19:23]}[ptype]
    if val == "____":
        return "OUT OF RANGE"
    return DRIVE_LOOKUP_TABLE[val]


def sequence(fn):  # fn[35:44])
    return fn[35:44]


def cam_specific(fn):  # fn[44:48])
    return fn[44:48]  # To do complex reconstruction


def downsample(fn):  # fn[48:49])
    n = int(2 ** int(fn[48:49]))
    return f"{n}x{n}"


def compression(fn):  # fn[49:51])
    if fn[49:51].startswith("I"):
        if int(fn[49:51][1]) < 9:
            return f"{int(fn[49:51][1])} bpp ICER (lossy)"
        else:
            return f">8 bpp ICER (lossy)"
    elif fn[49:51].startswith("L"):
        return {
            "LI": "Lossless (ICER)",
            "LL": "Lossless (LOCO)",
            "LM": "Lossless (Malin)",
            "LU": "Uncompressed",
        }[fn[49:51]]


def producer(fn, ptype="IMAGE"):  # fn[51:52])
    val = {"IMAGE": fn[51:52], "MOSAIC": fn[37:38]}[ptype]
    if val == "J":
        return "JPL"
    elif val == "A":
        return "ASU"
    elif val == "P":
        inst = instrument(fn, ptype="MOSAIC1" if "MOSAIC" in ptype else ptype)
        if ("Hazcam" in inst) or ("Navcam" in inst):  # ECAM
            return "JPL"
        elif "Mastcam" in inst:
            return "ASU (Tempe, AZ)"
        elif inst == "Supercam RMI":
            return "IRAP (France)"
        elif "MCC" in inst:
            return "JPL"
        elif "SHERLOC" in inst:
            return "JPL"
        elif inst == "MEDA Skycam":
            return "Ministry of Education and Science (Spain)"
        elif "EDL" in inst:
            return "JPL"
        elif "Helicopter" in inst:
            return "JPL"
        else:
            return val
    elif val == "_":
        return "undefined/other"
    else:
        return "Instrument Co-I"


def version_counter():
    # generates a lookup table per p189 of the SIS
    # version counter indexing starts at 1
    # first valid 01, last valid ZZ
    digits, ascii_uppercase = string.digits, string.ascii_uppercase
    collection = (
        itertools.islice(
            itertools.product(digits, digits), 1, None
        ),  # 00 is not valid
        itertools.product(ascii_uppercase, digits + ascii_uppercase),
    )
    table = {
        k: v
        for v, k in enumerate(
            map("".join, itertools.chain(*collection)), start=1
        )
    }
    return table


def version(fn, ptype="IMAGE"):  # fn[52:54])
    val = {"IMAGE": fn[52:54], "MOSAIC": fn[38:40]}[ptype]
    if val == "__":
        return "OUT OF RANGE"
    return version_counter()[val]


def ext(fn, ptype="IMAGE"):  # fn[54:58])
    try:
        return {
            "VIC": "VICAR",
            "IMG": "VIC w/ ODL label",
            "TIF": "TIFF",
            "JPG": "JPEG",
            "PNG": "PNG",
            "TXT": "ASCII",
            "iv": "Inventor-format",
            "ht": "VICAR (Height-map)",
            "rgb": "SGI RGB (Skin)",
            "obj": "Wavefront OBJ (Mesh)",
            "mtl": "OBJ (Material)",
            "png": "Browse image -or- Mesh",  # ???
            "mlp": "MeshLab project",
            "xml": "PDS label in xml format",
            "xmlf": "filter file w/ rover mask polygons in xml format",
            "WAV": "WAV-format audio",
            "flac": "Free Lossless Audio",
            # "MXML" : "masked XYZ (MXY) xml", # Deprecated?
            "png.xml": "Browse image xml format",  # ????
        }[
            {
                "IMAGE": fn.split(".", 1)[-1],  # fn[55:],
                "MOSAIC:": fn.split(".", 1)[-1],  # fn[42:]
            }[ptype]
        ]
    except KeyError:
        return f"Unknown: {fn.split('.',1)[-1]}"


def get_ptype(fn):  # get the product type
    try:
        return {54: "IMAGE", 40: "MOSAIC"}[
            len(fn.split(".")[0])
        ]  # cut off the extension
    except:
        return "UNKNOWN"


def parse(fn):
    ptype = get_ptype(fn)
    if not ptype in ["IMAGE", "MOSAIC", "MESH"]:
        print(f"Unknown filename type.")
        return
    elif ptype == "MESH":
        print("MESH filenames are not yet supported.")
        return
    # All the ugly conditionals are just to order the information in the
    # most useful way for display.
    fntable = {}
    fntable["FILENAME"] = fn
    if ptype == "IMAGE":
        fntable["INSTRUMENT"] = instrument(fn, ptype=ptype)
    elif ptype == "MOSAIC":
        fntable["INSTRUMENT1"] = instrument(fn, ptype=ptype + "1")
        fntable["INSTRUMENT2"] = instrument(fn, ptype=ptype + "2")
    fntable["FILTER"] = color_filter(fn, ptype=ptype)
    fntable["SOL"] = sol(fn, ptype=ptype)
    fntable["PRODUCT_TYPE"] = product_type(fn, ptype=ptype)
    if ptype == "IMAGE":
        fntable["SCLK"] = (
            secondary_timestamp(fn) + tertiary_timestamp(fn) / 1000
        )
        fntable["SEQUENCE"] = sequence(fn)
        fntable["LINEARIZED"] = geometry(fn)
        fntable["THUMBNAIL"] = thumbnail(fn)
        # fntable["CAM_SPECIFIC"] = cam_specific(fn)  ###
        fntable["DOWNSAMPLED"] = downsample(fn)
        fntable["COMPRESSION"] = compression(fn)
    # fntable["SPECIAL"] = special_flag(fn,ptype=ptype)
    fntable["VENUE"] = venue(fn, ptype=ptype)
    fntable["SITE"] = site(fn, ptype=ptype)
    fntable["DRIVE"] = drive(fn, ptype=ptype)
    fntable["PRODUCER"] = producer(fn, ptype=ptype)
    fntable["VERSION"] = version(fn, ptype=ptype)
    fntable["FILETYPE"] = ext(fn, ptype=ptype)
    fntable["PRODUCT_DESCRIPTION"] = parse_product_type(fn, ptype=ptype)

    return pd.Series(fntable)


class Parse:
    def __call__(self, fn):
        # TODO: A decision tree between missions. Right now, everything is M2020.
        return parse(fn)
