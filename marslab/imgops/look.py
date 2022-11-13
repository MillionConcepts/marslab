"""
utilities for composing lightweight imaging pipelines
"""
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Collection, Any

from dustgoggles.composition import Composition
from dustgoggles.func import argstop

import marslab.spectops
from marslab.imgops.imgutils import map_filter, crop_all

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def look_to_function(look: str) -> Callable:
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


def interpret_mask_step(chunk):
    from marslab.imgops.masking import extract_masks
    return extract_masks, chunk


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
    if step_name == "mask":
        return interpret_mask_step(chunk)
    # specifying a function is mandatory,
    # specifying bound parameters is not
    return chunk["function"], chunk.get("params", {}).copy()


class Look(Composition, ABC):
    def __init__(
        self,
        *args,
        metadata: "pd.DataFrame" = None,
        bands: tuple[str] = None,
        special_constants: Collection[Any] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        self.bands = bands
        self.special_constants = special_constants
        if (self.metadata is not None) or (self.special_constants is not None):
            self.populate_kwargs()

    def _add_wavelengths(self, wavelengths: Sequence[float]):
        look = self.steps["look"]
        if "__name__" in dir(look):
            name = look.__name__
        else:
            name = look.func.__name__
        if name != "spectop_look":
            return False
        self.add_insert("look", "wavelengths", wavelengths)
        return True

    def _get_plotter_layers(self):
        try:
            return self.inserts['plotter']['layers']
        except KeyError:
            self.add_insert('plotter', 'layers', [])
            return self._get_plotter_layers()

    def add_underlay(self, underlay: "np.ndarray", layer_ix: int = -1):
        if "crop" in self.steps:
            underlay = self.steps["crop"](
                underlay, **self.inserts.get('crop', {})
            )
        self._get_plotter_layers().append(
            {'layer_ix': layer_ix, "image": underlay}
        )

    @classmethod
    def compile_from_instruction(
        cls,
        instruction: Mapping,
        metadata: "pd.DataFrame" = None,
        special_constants: Collection[Any] = None
    ):
        """compile a look instruction into a rendering pipeline"""
        # only look is required. these are _possible_ steps in the pipeline.
        step_names = (
            "crop",
            "prefilter",
            "mask",
            "look",
            "limiter",
            "postfilter",
            "plotter",
            "bang",
        )
        steps = {}
        inserts = {}
        for step_name in step_names:
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
            # TODO -- maybe implement this at Composition level
            look.steps['look'] = argstop(look.steps['look'], 2)
            look.add_send('mask', lambda s: s[2], look._get_plotter_layers())
        return look

    # TODO: is this excessively baroque; would an internal dispatch be better?
    def populate_kwargs(self):
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


def save_plainly(look, filename, outpath, dpi=275):
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
