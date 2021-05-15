"""
utilities for composing lightweight imaging pipelines
"""

from collections.abc import Callable, Mapping
from typing import Optional

from cytoolz.functoolz import curry, compose_left, identity

import marslab.spectops
from marslab.imgops.imgutils import broadcast_filter, crop_all
from marslab.imgops.render import (
    render_overlay,
    spectop_look,
    render_rgb_composite,
    decorrelation_stretch,
)


def unpack_pipe_component(specification: Optional[Mapping]):
    if specification is None:
        return None
    func = curry(specification.get("function"))
    params = specification.get("params", {})
    return func(**params)


def unpack_overlay(specification: Optional[Mapping]):
    if specification is None:
        return None
    options = specification.get("options", {})
    return curry(render_overlay)(**options)


def compose_if(*callables, callfilter=None):
    return compose_left(*list(filter(callfilter, callables)))


# this contains the kernel of a comprehensive pipeline definition algebra,
# which I unfortunately do not have room for in this margin. Adding hooks for
# specific arguments, locations in a composition where new arguments can be
# inserted, etc, in the meantime I will compensate by making hacky adjustments
# to rendering functions.


def assemble_overlay_pipeline(main_pipeline, overlay_component, cropper=None):
    if cropper is None:
        cropper = identity

    def overlay_pipeline(channels, *, base_image, **kwargs):
        return overlay_component(
            main_pipeline(channels, **kwargs), cropper(base_image)
        )

    return overlay_pipeline


#
#
# def look_pipeline(
#
# )


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
        prefilter = curry(broadcast_filter)(prefilter)
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
        return curry(spectop_look)(spectop=getattr(marslab.spectops, look))
    elif look in ("enhanced color", "true color", "composite"):
        return curry(render_rgb_composite)
    elif look == "dcs":
        return curry(decorrelation_stretch)
    else:
        raise ValueError("unknown look operation " + look)


def compile_look_instruction(instruction):
    """
    compile a full look instruction into a rendering pipeline
    """
    # get heart of pipeline: the look function (spectrum op, dcs, etc.)
    looker = look_to_function(instruction["operation"])
    # this is a curried function. bind options and auxiliary info:
    looker = looker(**instruction.get("look_params", {}))
    # add wavelength values to spectops -- others don't care as of now
    # all of cropper, pre, post, overlay, plotter can potentially be absent --
    # these are _possible_ steps in the pipeline.
    cropper = curry(crop_all)(bounds=instruction.get("crop"))
    pre = unpack_pipe_component(instruction.get("prefilter"))
    post = unpack_pipe_component(instruction.get("postfilter"))
    overlay = unpack_overlay(instruction.get("overlay"))
    # finally, are we rendering a matplotlib image, and if so, how?
    # TODO: maybe this should be merged in some way with overlay,
    #  which is in and of itself a fancy matplotlib trick --
    #  or the overlay should be performed differently?
    plotter = unpack_pipe_component(instruction.get("mpl_settings"))
    return assemble_look_pipeline(looker, cropper, pre, post, overlay, plotter)
