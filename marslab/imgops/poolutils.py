"""utilities for watching worker pools"""
import logging
import time
from types import MappingProxyType


class ChangeReporter:
    def __init__(self, mapping):
        self.state = MappingProxyType(mapping)
        self.reference = mapping

    def check(self, new_state):
        return {
            key: value
            for key, value in new_state.items()
            if self.state.get(key) != new_state.get(key)
        }

    def update(self, new_state):
        changes = self.check(new_state)
        self.state = MappingProxyType(new_state)
        return changes

    def query(self):
        return self.update(self.reference)


def watch_pool(result_map, interval=1, callback=None):
    in_readiness = {
        task: result.ready() for task, result in result_map.items()
    }
    task_report = ChangeReporter(in_readiness)
    while not all(in_readiness.values()):
        in_readiness = {
            task_ix: result.ready() for task_ix, result in result_map.items()
        }
        report = task_report.update(in_readiness)
        if callback is not None:
            callback(report)
        time.sleep(interval)
    return result_map


def simple_log_callback(logger=None, prefix=""):
    if logger is None:
        logger = logging.getLogger(__name__)

    def log_changed_keys(report):
        for key in report.keys():
            logger.info(prefix + key)

    return log_changed_keys


def wait_for_it(pool, results=None, log=None, message=None, callback=None,
                interval=0.1, as_dict=False):
    if (callback is None) and (log is not None):
        callback = simple_log_callback(log, message)
    pool.close()
    if results is not None:
        watch_pool(results, interval, callback)
    pool.join()
    if as_dict:
        return {key: result.get() for key, result in results.items()}
    return [result.get() for result in results.values()]
