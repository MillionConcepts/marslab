"""
This module contains utilities for composing lightweight image processing
pipelines intended principally for generating "quicklook"-type browse images
from multispectral data.

Its centerpiece is the class `Look`. `Look` objects iteratively apply a
sequence of functions to one or more input arrays. This module also implements
a simple DSL intended to allow users to modify the quicklook behavior of an
application by editing external configuration files.
"""
from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import signature
from pathlib import Path
from typing import (
    Any,
    Collection,
    Hashable,
    Literal,
    Optional,
    NotRequired,
    TYPE_CHECKING,
    TypedDict,
    Union,
)
import warnings

from dustgoggles.composition import Composition
from dustgoggles.func import argstop, triggerize

import marslab.spectops
from marslab.imgops.imgutils import crop_all, map_filter

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    import numpy as np
    import pandas as pd
    from PIL.Image import Image

STEP_NAMES = (
    "crop",
    "prefilter",
    "mask",
    "look",
    "limiter",
    "postfilter",
    "plotter",
    "bang",
)
"""
Valid steps of instructions in the DSL. Look treats each of these differently
when constructing itself with Look.compile_from_instruction(). Note that the
order of these names is syntactic; do not rearrange the elements of STEP_NAMES.
"""
# noinspection PyTypeHints
LookName = Literal[
    Literal[marslab.spectops.SPECTOP_NAMES],
    Literal[
        "enhanced color", "true color", "dcs", "composite", "nested_composite"
    ],
]
"""
Recognized names of operations for the 'look' step. (If none are appropriate, 
a DSL instruction can instead explicitly specify a Python function.)
"""


class MaskFill(TypedDict):
    """
    Format of the optional 'colorfill' value of a sub-instruction in a 'mask'
    step.
    """

    # what color should we fill the masked regions with?
    # expressed either as a single 0-255 integer (for a grayscale fill))
    # or a tuple of 3 0-255 integers defining the fill's color as RGB.
    color: Union[int, tuple[int, int, int]]
    # alpha / opacity for the color fill layer, expressed as a number between
    # 0 and 1. 0 specifies a fully transparent fill (which is pointless!);
    # 1 specifies a fully opaque fill.
    mask_alpha: float


class LayerDef(TypedDict):
    """
    Format of an element of the 'layers' list that is an optional insert to
    a 'plotter' step. Only relevant if the plotter function itself understands
    it.
    """

    layer_ix: int
    image: "np.ndarray"


# NOTE: This is written in a weird way because 'pass' is a reserved keyword.
MaskInstruction = TypedDict(
    "MaskInstruction",
    {
        # the primary masking function
        "function": Callable,
        "params": NotRequired[Mapping[str, Any]],
        # specifies a color fill for the mask in downstream
        "colorfill": NotRequired[MaskFill],
        # send the mask as an insert to the plotter step?
        "send": NotRequired[bool],
        # pass the mask on to the next function?
        "pass": NotRequired[bool],
    },
)
"""
'mask' steps defined in the DSL cause the generated Look to construct one
or more masks in sequence. This is the format of an sub-instruction in
a 'mask' step.
"""


class StepDef(TypedDict):
    """Format of most steps in a Look DSL instruction."""

    # explicitly-given Python function
    function: Callable
    # kwargs to pass to `function``
    params: NotRequired[Mapping[str, Any]]


class PrefilterDef(StepDef):
    """Format of prefilter step."""

    # if True, assumes the first argument is a tuple of arrays, and applies the
    # prefilter function separately to each.
    map: NotRequired[bool]


class BangDef(TypedDict):
    function: Callable[[], Any]
    params: NotRequired[Mapping[str, Any]]


class LookInstruction(TypedDict):
    """
    Look DSL instructions are Python Mappings that follow this structure.
    """

    # defines a crop on the input image in pixels: (left, right, bottom, top)
    crop: NotRequired[tuple[int, int, int, int]]
    # a first-pass preprocessing step
    prefilter: NotRequired[PrefilterDef]
    # an additional preprocessing step
    limiter: NotRequired[StepDef]
    # the primary look operation
    look: Union[LookName, Callable]
    # kwargs for the primary look operation
    params: NotRequired[Mapping[str, Any]]
    # if this Look is passed to BandSet.make_look_set(), this specifies which
    # of the BandSet's bands it should send to the Look. Does nothing in other
    # cases.
    bands: NotRequired[Sequence[Hashable]]
    # TODO: it's a silly hack to have a required 'instructions' key;
    #  'mask' should just be a sequence of MaskInstructions.
    mask: NotRequired[
        Mapping[Literal["instructions"], Sequence[MaskInstruction]]
    ]
    limiter: NotRequired[StepDef]
    plotter: NotRequired[StepDef]
    # optional niladic function called on successful execution of the rest
    # of the Look. can be used for logging or whatever.
    bang: NotRequired[BangDef]
    # optional identifier for look
    name: NotRequired[str]


# TODO, maybe: we need complicated Protocol stuff to precisely type-hint this
def look_to_function(look: str) -> Callable:
    """
    Implements a mapping between plain-language names of core look operations
    and Python functions.
    """
    from marslab.imgops import render

    if look in marslab.spectops.SPECTOP_NAMES:
        return partial(
            render.spectop_look, spectop=getattr(marslab.spectops, look)
        )
    elif look in ("enhanced color", "true color", "composite"):
        return render.render_rgb_composite
    elif look == "dcs":
        return render.decorrelation_stretch
    elif look == "nested_composite":
        return render.render_nested_rgb_composite
    raise ValueError("unknown look operation " + look)


def interpret_look_step(instruction: LookInstruction):
    """
    Heart of look pipeline: the look function (spectrum op, dcs, etc.), by
    name or as an explicitly-defined Python function. This step is _mandatory_.
    """
    try:
        step = instruction["look"]
    except KeyError:
        raise ValueError("The instruction must include at least a look type.")
    if isinstance(step, str):
        step = look_to_function(step)
    elif callable(step) is False:
        raise TypeError(
            "An instruction's 'look' must be a Python function or a string."
        )
    return step, instruction.get("params", {})


def interpret_crop_step(
    bounds: Optional[tuple[int, int, int, int]]
) -> tuple[Callable, dict]:
    """
    Interpret the crop step of a LookInstruction.
    note at present a cropper is always included whether you ask for it or not,
    but by default it acts as the identity function.
    """
    return crop_all, {"bounds": bounds}


def interpret_prefilter_step(inst: PrefilterDef) -> tuple[Callable, dict]:
    """Interpret the prefilter step of a LookInstruction."""
    step = inst["function"]
    # by default, wrap the prefilter function so that it works as a variadic
    # function, applying itself to each input channel individually; some
    # notional prefilters may of course not want this!
    if inst.get("map") is not False:
        step = map_filter(step)
    # noinspection PyTypeChecker
    return step, inst.get("params", {})


def interpret_mask_step(chunk):
    """Interpret a 'mask' step from a Look instruction."""
    from marslab.imgops.masking import extract_masks

    return extract_masks, chunk


def interpret_instruction_step(
    instruction: LookInstruction, step_name: Literal[STEP_NAMES]
) -> tuple[Optional[Callable], Mapping]:
    """
    unpack an individual element of a LookInstruction into steps, inserts,
    and sends for a Composition.
    """
    # check central mandatory element first
    if step_name == "look":
        return interpret_look_step(instruction)
    # bail out if non-mandatory element is not present
    if step_name not in instruction.keys():
        return None, {}
    if step_name not in STEP_NAMES:
        warnings.warn(
            f"{step_name} is not a known Look step; this may cause unintended "
            f"behavior."
        )
    # noinspection PyTypedDict
    chunk = instruction[step_name]
    # other special cases
    if step_name == "crop":
        return interpret_crop_step(chunk)
    if step_name == "prefilter":
        return interpret_prefilter_step(chunk)
    if step_name == "mask":
        return interpret_mask_step(chunk)
    if step_name == "bang":
        return triggerize(chunk["function"]), chunk.get("params", {}).copy()
    # If no special case is specified, prep for default Composition behavior.
    # specifying a function is mandatory.
    # specifying bound parameters is not.
    return chunk["function"], chunk.get("params", {}).copy()


class Look(Composition, ABC):
    def __init__(
        self,
        *args,
        metadata: "pd.DataFrame" = None,
        bands: tuple[str] = None,
        special_constants: Collection[Any] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        self.bands = bands
        self.special_constants = special_constants
        if (self.metadata is not None) or (self.special_constants is not None):
            self.populate_kwargs()

    def _add_wavelengths(self, wavelengths: Sequence[float]) -> bool:
        """
        Attempt to add an insert to the look step giving it wavelengths. If -
        the look step isn't a spectop, don't actually do this.
        Returns True if the look step is a spectop, and False if not.
        """
        look = self.steps["look"]
        if "__name__" in dir(look):
            name = look.__name__
        else:
            name = look.func.__name__
        if name != "spectop_look":
            return False
        self.add_insert("look", "wavelengths", wavelengths)
        return True

    def _get_plotter_layers(self) -> list[LayerDef]:
        """
        If it exists, return the list of layers prepped as an insert to
        the plotter step. If not, construct an empty one and return it.
        """
        try:
            return self.inserts["plotter"]["layers"]
        except KeyError:
            self.add_insert("plotter", "layers", [])
            return self._get_plotter_layers()

    def add_underlay(self, underlay: "np.ndarray", layer_ix: int = -1):
        """
        Add an underlay to the plotter step. Will only do
        anything if the plotter step understands/accepts layers.
        """
        if "crop" in self.steps:
            underlay = self.steps["crop"](
                underlay, **self.inserts.get("crop", {})
            )
        self._get_plotter_layers().append(
            {"layer_ix": layer_ix, "image": underlay}
        )

    @classmethod
    def compile_from_instruction(
        cls,
        instruction: LookInstruction,
        metadata: "pd.DataFrame" = None,
        # TODO: is this cruft?
        special_constants: Collection[Any] = None,
    ):
        """
        Compile an instruction written in the Look DSL into a rendering
        pipeline. See the docstring of `LookInstruction` for syntax.

        `metadata` is an optional DataFrame formatted like the `metadata`
        property of a `Bandset`. Some `Look` functionality will not work
        without metadata.
        """
        steps = {}
        inserts = {}
        for step_name in STEP_NAMES:
            step, kwargs = interpret_instruction_step(instruction, step_name)
            if step is not None:
                steps[step_name] = step
            if kwargs != {}:
                inserts[step_name] = kwargs
        look = cls(
            steps,
            inserts=inserts,
            metadata=metadata,
            special_constants=special_constants,
            bands=instruction.get("bands"),
        )
        if "mask" in steps.keys():
            # add 'splitter'
            # TODO, maybe: implement this at Composition level
            look.steps["look"] = argstop(look.steps["look"], 2)
            look.add_send("mask", lambda s: s[2], look._get_plotter_layers())
        return look

    # TODO: is this excessively baroque; would an internal dispatch be better?
    def populate_kwargs(self):
        """
        Helper function for Look construction. Populates `Composition`
        structure implied by `Look.steps` and `Look.metadata`.
        """
        for step in self.steps:
            params = signature(self.steps[step]).parameters.values()
            param_names = [param.name for param in params]
            for thing in ("special_constants", "metadata"):
                if (hasattr(self, thing)) and (thing in param_names):
                    self.add_insert(step, thing, getattr(self, thing))
        if self.bands is not None:
            if "WAVELENGTH" in self.metadata.columns:
                wavelengths = []
                for band in self.bands:
                    try:
                        wavelengths.append(
                            self.metadata.loc[
                                self.metadata["BAND"] == band, "WAVELENGTH"
                            ].iloc[0]
                        )
                    except (KeyError, IndexError):
                        continue
                self._add_wavelengths(wavelengths)


# TODO: doesn't go in this module.
def save_plainly(
    look: Union["Figure", "Image"],
    filename: Union[str, Path],
    outpath: Union[str, Path],
    dpi: int = 275,
):
    """
    Save a PIL Image or matplotlib Figure as simply as possible to
    `filename`/`outpath` with specified `dpi`. dpi is ignored if `look` is
    an `Image`.
    """
    from matplotlib.figure import Figure

    if isinstance(look, Figure):
        for ix, axis in enumerate(look.axes):
            if ix > 0:
                axis.remove()
            else:
                axis.axis("off")
        look.savefig(
            Path(outpath, filename), dpi=dpi, bbox_inches="tight", pad_inches=0
        )
    else:
        look.save(Path(outpath, filename))
