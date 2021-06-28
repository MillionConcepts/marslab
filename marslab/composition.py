from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import wraps, partial
from itertools import repeat, chain
from operator import attrgetter
from typing import Any, Optional

from cytoolz import identity, compose


def zero(*_, **__):
    """take anything, return nothing"""
    return


def triggerize(niladic_function: Callable) -> Callable:
    """
    implicitly turns a function into a trigger step for
    a pipeline
    """

    @wraps(niladic_function)
    def trigger(deferred_state=None):
        niladic_function()
        return deferred_state

    return trigger


def reindex_mapping(mapping: Mapping) -> Mapping[int, Any]:
    assert isinstance(mapping, Mapping), "reindex_mapping only takes Mappings"
    return {
        ix: value for ix, value in zip(range(len(mapping)), mapping.values())
    }


def enumerate_as_mapping(sequence):
    if sequence is None:
        return {}
    if isinstance(sequence, Mapping):
        return sequence
    return {ix: element for ix, element in enumerate(sequence)}


def insert_after(
    new_key: Any, new_value: Any, prior_key: Any, mapping: Mapping
) -> Mapping:
    new_dict = {}
    for key, value in mapping.items():
        new_dict[key] = value
        if key == prior_key:
            new_dict[new_key] = new_value
    return new_dict


class Composition:
    """
    class defining a composition of steps, optionally with
    additional input, output, and i/o points ("inserts," "sends," "captures,"
    and "loops") -- conceptually like a signal processing chain but _not_
    designed for real-time signal processing

    inserts: dict of index: args/kwargs to be added to function at that step
    in pipeline, or just sequence of None / args/kwargs; can also give
    index:None to defer argument pass to runtime; currently only
    one insert per step is supported
    sends: dict of index: function to call
    with the state at that step in pipeline, or just sequence of None /
    function currently only one send per step is supported
    loops: **not yet supported** dict of (index, index): function to
        call with the state at that step in pipeline, step in pipeline to
        return evaluation result
    parameters: dict of (index or function name, args/kwargs to
    bind to pipeline
    """

    def __init__(
        self,
        steps: Optional[Sequence[Callable]] = None,
        parameters: Optional[Mapping] = None,
        sends: Optional[Mapping[Any, Callable]] = None,
        inserts: Optional[Mapping] = None,
        captures: Optional[Mapping] = None,
    ):
        if steps is None:
            steps = [identity]
        self.steps = enumerate_as_mapping(steps)
        self.parameters = enumerate_as_mapping(parameters)
        self.sends = enumerate_as_mapping(sends)
        self.inserts = enumerate_as_mapping(inserts)
        self.captures = enumerate_as_mapping(captures)
        self.bind_all()

    def check_types(self):
        # todo: probably useless in practice
        raise NotImplementedError

    def arity(self):
        # todo: hard?
        raise NotImplementedError

    def _check_for_step(self, step_name: Hashable) -> Any:
        if step_name not in self.steps.keys():
            raise KeyError(
                str(step_name) + " is not an element of the pipeline."
            )
        return self.steps[step_name]

    def bind_kwargs(self, step_name, rebind=False, **kwargs):
        if kwargs == {}:
            return
        step = self._check_for_step(step_name)
        if (rebind is True) and ("func" in dir(step)):
            step = step.func
        if (rebind is True) or (
            not isinstance(self.parameters.get("step_name"), Mapping)
        ):
            self.parameters[step_name] = kwargs
        else:
            self.parameters[step_name] |= kwargs
        self.steps[step_name] = partial(step, **kwargs)

    def swap_kwargs(self, step_name, **kwargs):
        self.bind_kwargs(step_name, True, **kwargs)

    def add_kwargs(self, step_name, **kwargs):
        self.bind_kwargs(step_name, False, **kwargs)

    def bind_all(self, kwarg_mapping=None, rebind=False):
        """
        attach parameter dictionary to pipeline steps.
        this does not presently support positional-only arg bindings;
        use inserts for those.

        warning: rebind=True rebinds _all_ arguments of matching
        partially-evaluated pipeline steps!
        """
        if kwarg_mapping is None:
            kwarg_mapping = self.parameters
        else:
            kwarg_mapping = enumerate_as_mapping(kwarg_mapping)
        for step_name, kwargs in kwarg_mapping.items():
            if kwargs is None:
                continue
            self.bind_kwargs(step_name, rebind, **kwargs)

    def add_step(self, step, name=None, replace=False):
        """
        add step with name "name" to end of pipeline.
        """
        assert replace in (False, True, "compose")

        if name is None:
            name = len(self.steps) + 1

        if self.steps == {}:
            self.steps[name] = step
            return

        if (self.steps.get("name") is not None) and (replace is False):
            raise KeyError(
                "there's already a step named "
                + str(name)
                + ". Pass 'replace=True' if you want to reuse this name."
            )
        self.steps[name] = step

    def add_aux(self, aux, aux_type, step_name=None, replace=False):
        assert aux_type in ("send", "insert", "capture"), (
            "I don't know " + aux_type + " as a type of auxiliary block."
        )
        if self.steps == {}:
            raise KeyError(
                "At least one step must be defined to place a(n) "
                + aux_type
                + " ."
            )
        if step_name is None:
            step_name = next(reversed(self.steps))
        aux_dict = getattr(self, aux_type + "s")
        if (aux_dict.get(step_name) is None) or (replace is True):
            aux_dict[step_name] = aux
        elif replace == "compose":
            if aux_type == "insert":
                raise ValueError("Can't compose inserts.")
            aux_dict[step_name] = compose(aux, aux_dict[step_name])
        elif replace is False:
            raise KeyError(
                "there's already a(n) "
                + aux_type
                + " attached to "
                + str(step_name)
                + ". Pass replace=True or replace='compose' if you want to "
                + "add more things here."
                + aux_type
                + "."
            )
        else:
            raise ValueError("I don't recognize that replacement type.")

    def add_send(self, send, step_name=None, replace=False):
        """adds a send to the pipeline after step_name"""
        self.add_aux(send, "send", step_name, replace)

    def add_capture(self, step_name, replace=False):
        """
        adds a capture point to the pipeline after step_name; if a send is
        present, sends the output of the send there
        """
        self.add_aux(None, "capture", step_name, replace)

    def add_trigger(self, trigger, step_name=None, replace=False):
        """convenience wrapper for add_send"""
        trigger = triggerize(trigger)
        self.add_send(trigger, step_name, replace)

    def add_insert(self, insert, step_name=None, replace=False):
        """adds an insert to the pipeline before step_name"""
        self.add_aux(insert, "insert", step_name, replace)

    def add_loop(self, loop, step_after, step_before, replace=False):
        raise NotImplementedError

    def reindex(self):
        self.steps = reindex_mapping(self.steps)

    def flatten_sends(self):
        raise NotImplementedError
        # while self.steps.get(name) is not None:
        #     if str(name)[-1].isdigit():
        #         name = name[:-1] + str(int(name[-1]) + 1)
        #     else:
        #         name = str(name) + "_1"

    @property
    def index(self):
        return tuple(self.steps.keys())

    @property
    def insert_index(self):
        return tuple(self.inserts.keys())

    @property
    def send_index(self):
        return tuple(self.sends.keys())

    @property
    def function_names(self):
        return list(map(attrgetter("__name__"), self.steps.values()))

    def _process_insert_parameters(
        self, step_name, rt_insert_chain, rt_insert_kwargs
    ):
        insert_args = ()
        insert_kwargs = {}
        if step_name in self.inserts.keys():
            if self.inserts.get(step_name) is None:
                insert_args = [next(rt_insert_chain)]
            else:
                insert_kwargs = self.inserts[step_name]
            if step_name in rt_insert_kwargs.keys():
                insert_kwargs |= rt_insert_kwargs[step_name]
        return insert_args, insert_kwargs

    def _perform_send(self, step_name, state):
        """
        send pipeline state to send function, if one matching this step name
        exists. unlike inserts, there is currently no provision for sends with
        value None; there probably should not be.

        """
        if self.sends.get(step_name) is None:
            return state
        return self.sends.get(step_name)(state)

    def _do_step(self, state, step_name, rt_insert_chain, rt_insert_kwargs):
        """
        perform an individual step of the pipeline
        """
        insert_args, insert_kwargs = self._process_insert_parameters(
            step_name, rt_insert_chain, rt_insert_kwargs
        )
        step = self.steps[step_name]
        state = step(state, *insert_args, **insert_kwargs)
        if step_name in self.captures.keys():
            self.captures[step_name] = self._perform_send(step_name, state)
        else:
            self._perform_send(step_name, state)
        return state

    def _get_ready(self, rt_insert_args, rt_insert_kwargs, special_kwargs):
        self._bind_special_runtime_kwargs(special_kwargs)
        if rt_insert_kwargs is None:
            rt_insert_kwargs = {}
        runtime_insert_chain = chain(rt_insert_args, repeat(None))
        return runtime_insert_chain, rt_insert_kwargs

    # todo: modify this to produce a generator that can access internal state.
    #  this is basically a way to create returns.
    #  or actually: it's possibly better not even as a generator?
    def itercall(
        self,
        signal: Any = None,
        *rt_insert_args,
        rt_insert_kwargs: Mapping[Any] = None,
        **special_kwargs
    ):
        rt_insert_chain, rt_insert_kwargs = self._get_ready(
            rt_insert_args, rt_insert_kwargs, special_kwargs
        )
        state = signal
        for step_name, step in self.steps.items():
            state = self._do_step(
                state, step_name, rt_insert_chain, rt_insert_kwargs
            )
            yield state

    def execute(
        self,
        signal: Any = None,
        *rt_insert_args,
        rt_insert_kwargs: Mapping[Any] = None,
        **special_kwargs
    ):
        """
        execute the pipeline, initializing it with signal.
        """
        iterpipe = self.itercall(
            signal, rt_insert_args, rt_insert_kwargs, **special_kwargs
        )
        state = None
        for state in iterpipe:
            pass
        return state

    def iter(self):
        raise NotImplementedError

    def _bind_special_runtime_kwargs(self, runtime_insert_kwargs):
        pass

    def __str__(self):
        me_string = ""
        for attribute in (
            "steps",
            "parameters",
            "sends",
            "inserts",
            "captures",
        ):
            if not getattr(self, attribute):
                continue
            me_string += (
                attribute + ":\n" + getattr(self, attribute).__repr__() + "\n"
            )
        if not me_string:
            return "empty Composition"
        return me_string

    def __repr__(self):
        return self.__str__()


# class IterPipeline:
#     def __init__(
#         self,
#         pipeline: Composition,
#         signal: Any = None,
#         *rt_insert_args,
#         rt_insert_kwargs: Mapping[Any],
#         **_supplementary_kwargs
#     ):
#         self.pipeline = pipeline
#         self.state = signal
#         self.runtime_insert_chain = chain(rt_insert_args, repeat(None))
#         self.rt_insert_kwargs = rt_insert_kwargs
#       ...
