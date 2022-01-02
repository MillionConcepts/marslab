"""
utilities for composing lightweight imaging pipelines
"""
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.figure
from dustgoggles.composition import Composition

import marslab.spectops
from marslab.imgops.imgutils import map_filter, crop_all
from marslab.imgops.render import (
    render_overlay,
    spectop_look,
    render_rgb_composite,
    decorrelation_stretch,
    render_nested_rgb_composite,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def look_to_function(look: str) -> Callable:
    if look in marslab.spectops.SPECTOP_NAMES:
        return partial(spectop_look, spectop=getattr(marslab.spectops, look))
    elif look in ("enhanced color", "true color", "composite"):
        return render_rgb_composite
    elif look == "dcs":
        return decorrelation_stretch
    elif look == "nested_composite":
        return render_nested_rgb_composite
    else:
        raise ValueError("unknown look operation " + look)


def interpret_look_step(instruction):
    """
    heart of look pipeline: the look function (spectrum op, dcs, etc.)
    note this step is _mandatory_.
    """
    try:
        step = instruction["look"]
        if isinstance(step, str):
            step = look_to_function(step)
    except KeyError:
        raise ValueError("The instruction must include at least a look type.")
    return step, instruction.get("params", {})


def interpret_crop_step(bounds):
    """
    note at present a cropper is always included whether you ask for it or not,
    but by default it acts as the identity function.
    """
    return crop_all, {"bounds": bounds}


def interpret_prefilter_step(chunk):
    step = chunk["function"]
    # by default, wrap the prefilter function so that it works as a variadic
    # function, applying itself to each input channel individually; some
    # notional prefilters may of course not want this!
    if chunk.get("map") is not False:
        step = map_filter(step)
    return step, chunk.get("params", {})


def interpret_overlay_step(chunk):
    return render_overlay, chunk.get("params", {})


def interpret_instruction_step(
    instruction: Mapping, step_name: str
) -> tuple[Optional[Callable], Mapping]:
    """
    unpack individual elements of the look instruction markup
    syntax into steps and kwargs for a Composition.
    """
    # check central mandatory element first
    if step_name == "look":
        return interpret_look_step(instruction)
    # bail out if non-mandatory element is not present
    if step_name not in instruction.keys():
        return None, {}
    chunk = instruction[step_name]
    # other special cases
    if step_name == "crop":
        return interpret_crop_step(chunk)
    if step_name == "prefilter":
        return interpret_prefilter_step(chunk)
    if step_name == "overlay":
        return interpret_overlay_step(chunk)
    # specifying a function is mandatory,
    # specifying bound parameters is not
    return chunk["function"], chunk.get("params", {})


class Look(Composition, ABC):
    def __init__(
        self,
        *args,
        metadata: "pd.DataFrame" = None,
        bands: tuple[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        self.bands = bands
        if self.metadata is not None:
            self.populate_kwargs_from_metadata()

    def _add_wavelengths(self, wavelengths: Sequence[float]):
        look = self.steps["look"]
        if "__name__" in dir(look):
            name = look.__name__
        else:
            name = look.func.__name__
        if name != "spectop_look":
            return False
        self.add_kwargs("look", wavelengths=wavelengths)
        return True

    def _add_underlay(self, underlay: "np.ndarray"):
        if "overlay" not in self.index:
            return False
        if "crop" in self.steps:
            underlay = self.steps["crop"](underlay)
        self.add_kwargs("overlay", base_image=underlay)

    def _bind_special_runtime_kwargs(self, special_kwargs: Mapping):
        if special_kwargs.get("base_image") is not None:
            self._add_underlay(special_kwargs["base_image"])
        if special_kwargs.get("wavelengths") is not None:
            self._add_wavelengths(special_kwargs["wavelengths"])

    @classmethod
    def compile_from_instruction(
        cls, instruction: Mapping, metadata: "pd.DataFrame" = None
    ):
        """
        compile a look instruction into a rendering pipeline
        """
        # all of cropper, pre, post, overlay, plotter can potentially be
        # absent --
        # these are _possible_ steps in the pipeline.
        step_names = (
            "crop",
            "prefilter",
            "look",
            "limiter",
            "postfilter",
            "overlay",
            "plotter",
            "bang",
        )
        steps = {}
        parameters = {}
        for step_name in step_names:
            step, kwargs = interpret_instruction_step(instruction, step_name)
            if step is not None:
                steps[step_name] = step
            if kwargs != {}:
                parameters[step_name] = kwargs
        return cls(
            steps,
            parameters=parameters,
            metadata=metadata,
            bands=instruction.get("bands"),
        )

    # TODO: is this excessively baroque; would an internal dispatch be better?
    def populate_kwargs_from_metadata(self):
        if "metadata" in [
            param.name
            for param in signature(self.steps['look']).parameters.values()
        ]:
            self.add_kwargs("look", metadata=self.metadata)
        if self.bands is not None:
            if "WAVELENGTH" in self.metadata.columns:
                wavelengths = []
                for band in self.bands:
                    wavelengths.append(
                        self.metadata.loc[
                            self.metadata["BAND"] == band, "WAVELENGTH"
                        ].iloc[0]
                    )
                self._add_wavelengths(wavelengths)


def save_plainly(look, filename, outpath, dpi=275):
    if isinstance(look, matplotlib.figure.Figure):
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
