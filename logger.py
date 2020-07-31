from typing import Callable, List, Union, Dict
import numpy as np


class Log:
    def __init__(self, compute_value: Callable = None):
        self.value = None
        self.batch_index = -1
        if compute_value is not None:
            self.compute_value = compute_value

    def compute_and_update(self, logger, batch_index):
        self.batch_index = batch_index
        self.value = self.compute_value(logger)
        return self.value

    def set_value(self, v, b):
        self.value = v
        self.batch_index = b

class Logger:
    def __init__(self, logs = None):
        self.batch_index = 0
        self.logs = {k: Log() for k in logs.keys()}
        [self.logs[k].set_value(v, self.batch_index) for k, v in logs.items()]

    def update_logs(self, batch_index, logs):
        self.batch_index = batch_index
        [self.logs[k].set_value(v,  self.batch_index) for k, v in logs.items()]

    def add_log(self, name, log: Log):
        self.logs[name] = log

    def get(self, log: str, this_batch=True):
        assert log in self.logs, f'Log value {log} not found'
        if self.logs[log].batch_index is not None:
            if this_batch:
                if self.logs[log].batch_index == self.batch_index:
                    return self.logs[log].value
                else:
                    if self.logs[log].compute_value is None:
                        return self.logs[log].value
                    else:
                        return self.logs[log].compute_and_update(self, self.batch_index)
            else:
                return self.logs[log].value


if __name__ == '__main__':
    logs = {'labels': np.array([0, 1, 1]), 'predicted': np.array([1, 1, 0])}

    my_logger = Logger(logs)

    print(my_logger.get('labels'))
    print(my_logger.get('predicted'))

    acc_log = Log(lambda logger: np.sum(logger.get('labels') == logger.get('predicted'))/np.sum(len(logger.get('labels'))))
    my_logger.add_log('acc_log', acc_log)
    print(my_logger.get('acc_log'))
    print(my_logger.get('acc_log'))

    logs = {'labels': np.array([0, 1, 1]), 'predicted': np.array([1, 1, 1])}
    my_logger.update_logs(1, logs)
    my_logger.get('acc_log')

    logs = {'labels': np.array([1, 1, 1]), 'predicted': np.array([1, 1, 1])}
    my_logger.update_logs(2, logs)
    my_logger.get('acc_log', this_batch=False)

    my_logger.get('acc_log', this_batch=True)
    complex_log = Log(lambda logger: logger.get('acc_log') + 1)
    my_logger.add_log('complex_log', complex_log)
    a=my_logger.get('complex_log')

