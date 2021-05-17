"""
utilities for composing lightweight imaging pipelines
"""
from abc import ABC
from functools import partial
from collections.abc import Callable, Mapping
from typing import Optional

from cytoolz.functoolz import curry, compose_left, identity

from marslab.pipeline import Pipeline
import marslab.spectops
from marslab.imgops.imgutils import map_filter, crop_all
from marslab.imgops.render import (
    render_overlay,
    spectop_look,
    render_rgb_composite,
    decorrelation_stretch,
)


def unpack_overlay(specification: Optional[Mapping]):
    if specification is None:
        return None
    options = specification.get("options", {})
    return curry(render_overlay)(**options)


def compose_if(*callables, callfilter=None):
    return compose_left(*list(filter(callfilter, callables)))


def assemble_overlay_pipeline(main_pipeline, overlay_component, cropper=None):
    if cropper is None:
        cropper = identity

    def overlay_pipeline(channels, *, base_image, **kwargs):
        return overlay_component(
            main_pipeline(channels, **kwargs), cropper(base_image)
        )

    return overlay_pipeline


def check_wavelengths(looker: Callable, **kwargs):
    """gets wavelengths from arguments for spectop looks; ignores otherwise"""
    if looker.__name__ == "spectop_look":
        return looker(wavelengths=kwargs.get("wavelengths"))
    return looker


def assemble_look_pipeline(
    looker: Callable,
    cropper: Callable = None,
    prefilter: Callable = None,
    postfilter: Callable = None,
    overlay: Callable = None,
    plotter: Callable = None,
    broadcast_prefilter: bool = True,
):
    # by default, wrap the prefilter function so that it works on each channel;
    # some notional prefilters may of course not want this!
    if (broadcast_prefilter is True) and (prefilter is not None):
        prefilter = map_filter(prefilter)
    # no overlay? great, bands just go straight down the pipe, no secondary
    #  entry point. check for wavelengths.
    if overlay is None:

        def execute_pipeline(images, **kwargs):
            boundlooker = check_wavelengths(looker, **kwargs)
            return compose_if(
                cropper, prefilter, boundlooker, postfilter, plotter
            )(images)

        return execute_pipeline

    # otherwise, make pipeline with secondary entry point for underlay image
    def execute_overlay_pipeline(images, base_image, **kwargs):
        boundlooker = check_wavelengths(looker, **kwargs)
        pipeline = compose_if(cropper, prefilter, boundlooker, postfilter)
        return overlay(
            pipeline(images, **kwargs), compose_if(cropper)(base_image)
        )

    return execute_overlay_pipeline


def look_to_function(look):
    if look in marslab.spectops.SPECTOP_NAMES:
        return partial(spectop_look, spectop=getattr(marslab.spectops, look))
    elif look in ("enhanced color", "true color", "composite"):
        return render_rgb_composite
    elif look == "dcs":
        return decorrelation_stretch
    else:
        raise ValueError("unknown look operation " + look)


def interpret_look_step(instruction, step_name, broadcast_prefilter=True):
    """
    unpack individual elements of the look instruction markup
    syntax into steps and kwargs for a Pipeline.
    """
    step = None
    kwargs = {}
    # heart of look pipeline: the look function (spectrum op, dcs, etc.)
    # TODO: this step is mandatory but this can be made a little cleaner
    if step_name == "look":
        step = look_to_function(instruction["look"])
        kwargs = instruction.get("params", {})
    # bail out if non-mandatory element is not present
    elif step_name not in instruction.keys():
        return step, kwargs
    # slightly different syntax for this one too
    elif step_name == "crop":
        step = crop_all
        kwargs = {"bounds": instruction.get("crop")}
    # by default, wrap the prefilter function so that it works on each channel;
    # some notional prefilters may of course not want this!
    elif step_name == "prefilter":
        step = instruction["prefilter"]["function"]
        if broadcast_prefilter is True:
            step = map_filter(step)
        kwargs = instruction["prefilter"].get("params", {})
    elif step_name == "overlay":
        step = render_overlay
        kwargs = instruction["overlay"].get("params", {})
    else:
        # specifying a function is mandatory
        step = instruction.get(step_name)["function"]
        # specifying bound parameters is not
        kwargs = instruction.get(step_name).get("params", {})
    return step, kwargs


class Look(Pipeline, ABC):
    def __init__(self, *args, metadata=None, bands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata
        self.bands = bands
        if (self.metadata is not None) and (self.bands is not None):
            self.get_kwargs_from_metadata()

    def _add_wavelengths(self, wavelengths):
        look = self.steps["look"]
        if "__name__" in dir(look):
            name = look.__name__
        else:
            name = look.func.__name__
        if name != "spectop_look":
            return False
        self.add_kwargs("look", wavelengths=wavelengths)
        return True

    def _add_underlay(self, underlay):
        if "overlay" not in self.index:
            return False
        if "crop" in self.steps:
            underlay = self.steps["crop"](underlay)
        self.add_kwargs("overlay", base_image=underlay)

    def _bind_special_runtime_kwargs(self, special_kwargs):
        if special_kwargs.get("base_image") is not None:
            self._add_underlay(special_kwargs["base_image"])
        if special_kwargs.get("wavelengths") is not None:
            self._add_wavelengths(special_kwargs["wavelengths"])

    @classmethod
    def compile_from_instruction(cls, instruction, metadata=None):
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
        )
        steps = {}
        parameters = {}
        for step_name in step_names:
            step, kwargs = interpret_look_step(instruction, step_name)
            if step is not None:
                steps[step_name] = step
            if kwargs is not None:
                parameters[step_name] = kwargs
        return cls(
            steps,
            parameters=parameters,
            metadata=metadata,
            bands=instruction.get("bands"),
        )

    def get_kwargs_from_metadata(self):
        assert (self.bands is not None) and (self.metadata is not None)
        if "WAVELENGTH" in self.metadata.columns:
            wavelengths = []
            for band in self.bands:
                wavelengths.append(
                    self.metadata.loc[
                        self.metadata["BAND"] == band, "WAVELENGTH"
                    ].iloc[0]
                )
            self._add_wavelengths(wavelengths)
