"""
Utilities for watching worker pools. Can be used for progress updates, logging,
etc.
"""
import os
from collections.abc import Callable, Mapping
import logging
from multiprocessing import active_children, Pool
import time
from types import MappingProxyType
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pathos.multiprocessing import ProcessPool


class ChangeReporter:
    """
    Simple class for monitoring changes in a dict or other mapping.
    """

    def __init__(self, mapping: Mapping):
        self.state = MappingProxyType(mapping)
        self.reference = mapping

    def check(self, new_state: Mapping):
        return {
            key: value
            for key, value in new_state.items()
            if self.state.get(key) != new_state.get(key)
        }

    def update(self, new_state: Mapping):
        changes = self.check(new_state)
        self.state = MappingProxyType(new_state)
        return changes

    def query(self):
        return self.update(self.reference)


def watch_pool(
    result_map: Mapping,
    interval: float = 0.1,
    callback: Optional[Callable] = None,
    timeout: Optional[float] = None
) -> Mapping:
    start = time.time()
    in_readiness = {
        task: result.ready() for task, result in result_map.items()
    }
    task_report = ChangeReporter(in_readiness)
    while not all(in_readiness.values()):
        if timeout is not None:
            if time.time() - start > timeout:
                raise TimeoutError
        in_readiness = {
            task_ix: result.ready() for task_ix, result in result_map.items()
        }
        report = task_report.update(in_readiness)
        if callback is not None:
            callback(report)
        time.sleep(interval)
    return result_map


def simple_log_callback(
    logger: Optional[logging.Logger] = None, prefix: str = ""
):
    if logger is None:
        logger = logging.getLogger(__name__)

    def log_changed_keys(report):
        for key in report.keys():
            logger.info(prefix + key)

    return log_changed_keys


def wait_for_it(
    pool: Union["ProcessPool", "Pool"],
    results: Mapping,
    log: logging.Logger = None,
    message: str = None,
    callback: Optional[Callable] = None,
    interval: float = 0.1,
    as_dict: bool = False,
    timeout: Optional[float] = None
) -> Union[dict, list]:
    if (callback is None) and (log is not None):
        callback = simple_log_callback(log, message)
    pool.close()
    if results is not None:
        try:
            watch_pool(results, interval, callback, timeout)
        except TimeoutError:
            pool.terminate()
            for c in active_children():
                os.kill(c.pid, 30)
            raise TimeoutError("Pool timed out")
    pool.join()
    if as_dict:
        return {key: result.get() for key, result in results.items()}
    return [result.get() for result in results.values()]
