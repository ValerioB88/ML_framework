from typing import Callable, List, Union, Dict
import numpy as np
import warnings

class Log:
    def __init__(self, name, init_value=None, compute_value: Callable = None, index=0, auto_update=False):
        self.value = init_value
        self.index = index
        self.name = name
        self.auto_update = auto_update
        self.compute_value = None
        if compute_value is not None:
            self.compute_value = compute_value

    def compute_and_update(self, logger, index):
        self.value = self.compute_value(logger)
        self.index = index
        return self.value

    def set_value(self, v, b):
        self.value = v
        self.index = b

    def set_index(self, i):
        self.index = i
# ToDo: this should be a singleton
# ToDo: also consider a Messenger/Data Transfer Object: https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Messenger.html
class Logger:
    def __init__(self, dict_logs=None):
        if dict_logs is None:
            self.logs = {}
        self.index = -1
        self.logs: Dict[str, Log] = {}
        if dict_logs != {}:
            self.next_index(dict_logs)
        # [self.logs[k].set_value(v, self.batch_index) for k, v in logs.items()]

    def next_index(self, dict_logs: Dict = None):
        """
        Update the logs that are passed to this function (that is, the dict logs, not the computed ones)
        You cannot pass computed logs here, as they need to be computed, not passed.
        """
        self.index += 1
        self._update_logs(dict_logs)

    def _update_logs(self, dict_logs):
        if dict_logs is not None:
            self.logs.update({k: Log(name=k, init_value=v, index=self.index) for k, v in dict_logs.items()})

        if dict_logs is None:  # I am not sure this is a good idea.
            [v.set_index(self.index) for k, v in self.logs.items() if v.compute_value is None]

        # Some logs needs to be updated everytime that the index is updated
        [self.get(k) for k, v in self.logs.items() if v.auto_update]

    def add_log(self, log: Log):
        self.logs[log.name] = log
        self.logs[log.name].index = self.index

    def get(self, log: str, same_index=True):
        assert log in self.logs, f'Log value {log} not found'
        if self.logs[log].index is not None:
            if same_index:
                if self.logs[log].index == self.index and self.logs[log].value is not None:
                    return self.logs[log].value
                else:
                    if self.logs[log].compute_value is None:
                        warnings.warn('Stale version of log [{}] used'.format(self.logs[log].name))
                        return self.logs[log].value
                    else:
                        return self.logs[log].compute_and_update(self, self.index)
            else:
                return self.logs[log].value


LOGGER = Logger()


if __name__ == '__main__':
    logs = {'labels': np.array([0, 1, 1]), 'predicted': np.array([1, 1, 0])}
    my_logger = Logger(logs)

    assert np.all(my_logger.get('labels') == np.array([0, 1, 1]))
    assert np.all(my_logger.get('predicted') == np.array([1, 1, 0]))

    acc_log = Log(name='acc_log', compute_value=lambda logger: np.sum(logger.get('labels') == logger.get('predicted')) / np.sum(len(logger.get('labels'))))
    my_logger.add_log(acc_log)
    assert my_logger.get('acc_log') == 1/3

    logs = {'labels': np.array([0, 1, 1]), 'predicted': np.array([1, 1, 1])}
    my_logger.next_index(logs)
    assert my_logger.get('acc_log') == 2/3

    logs = {'labels': np.array([1, 1, 1]), 'predicted': np.array([1, 1, 1]), 'new_value': 100}
    my_logger.next_index(logs)
    assert my_logger.get('new_value') == 100
    assert my_logger.get('acc_log', same_index=False) == 2/3  # value is not updated as we specifically set same_index=False
    assert my_logger.get('acc_log') == 1

    complex_log = Log(name='complex_log', compute_value=lambda logger: logger.get('acc_log') + 1)
    my_logger.add_log(complex_log)
    assert my_logger.get('complex_log') == 2.0

####################################
    # update logs with new logs
    my_logger = Logger()
    # If a value is initialized to 0, that's going to be it's value at the first index.
    logs = {'labels': np.array([1, 1, 1]), 'predicted': np.array([1, 1, 1]), 'new_value': 100}
    my_logger.next_index(logs)
    auto_log = Log(name='auto_log', init_value=0, compute_value=lambda logger: logger.get('new_value') + 1)
    my_logger.add_log(auto_log)
    assert my_logger.get('auto_log') == 0
    my_logger.next_index(logs)
    assert my_logger.get('auto_log') == 101

    # If it has no init value, the compute_value will be called as soon as the value is asked
    auto_log = Log(name='auto_log', init_value=None, compute_value=lambda logger: logger.get('new_value') + 1)
    my_logger.add_log(auto_log)
    assert my_logger.get('auto_log') == 101

    # update log on same index, but with only few of the available logs
    logs = {'labels': np.array([1, 1, 1]), 'predicted': np.array([1, 1, 1]), 'new_value': 100}
    my_logger = Logger(logs)
    acc_log = Log(name='acc_log', compute_value=lambda logger: np.sum(logger.get('labels') == logger.get('predicted')) / np.sum(len(logger.get('labels'))))
    my_logger.add_log(acc_log)

    logs = {'b': 110}
    my_logger.next_index(logs)
    assert np.all(my_logger.get('labels') == np.array([1, 1, 1]))
    assert my_logger.get('b') == 110
    assert my_logger.logs['labels'].index == 0
    assert my_logger.logs['b'].index == 1

    # update only the index of the passed logs
    my_logger.next_index(logs)
    assert np.all(my_logger.get('labels') == np.array([1, 1, 1]))  # stale version of labels used (but still ok)
    assert my_logger.get('b') == 110
    assert my_logger.logs['labels'].index == 0
    assert my_logger.logs['b'].index == 2
    assert my_logger.get('acc_log') == 1  # acc_log uses a stale version of labels but still works

    # update all indexes that do not require any computation
    # auto_log = Log(name='auto_log', init_value=0, compute_value=lambda logger: logger.get('auto_log') + 1)
    # my_logger.add_log(auto_log)
    #
    # my_logger.next_index()
    # assert my_logger.logs['b'].index == 3
    # assert my_logger.logs['auto_log'].index == 0  # notice how auto_log index, which requires computation, is not updated unless it's called
    # assert my_logger.get('auto_log') == 1
    # assert my_logger.logs['auto_log'].index == 5
    # my_logger.next_index()
    # assert my_logger.get('auto_log') == 2

#####################################################
    ## Add value to itself, remember that it only compute it for a new index
    # If a value is initialized to 0, that's going to be it's value at the first index.
    logs = {'labels': np.array([1, 1, 1]), 'predicted': np.array([1, 1, 1]), 'new_value': 100}
    my_logger = Logger(logs)
    auto_log = Log(name='auto_log', init_value=0, compute_value=lambda logger: logger.logs['auto_log'].value + 1)
    my_logger.add_log(auto_log)
    assert my_logger.get('auto_log') == 0
    assert my_logger.get('auto_log') == 0
    my_logger.next_index()
    assert my_logger.get('auto_log') == 1

    my_logger.next_index(logs)  # here we update the index of only the element in logs, but auto_log is computed anyway
    assert my_logger.get('auto_log') == 2
    assert my_logger.get('auto_log') == 2

    def comp(logger):
        logger.get('mat')[0, 1] += 100
        return logger.get('mat')
    my_logger.add_log(Log(name='mat', init_value=np.zeros((3, 3)), compute_value=comp))
    my_logger.next_index(logs)
    assert np.all(my_logger.get('mat') == np.array([[0, 100, 0], [0, 0, 0], [0, 0, 0]]))

    ## In some cases we have auto updated logs
    update_log = Log(name='update_log', init_value=0, compute_value=lambda logger: logger.get('update_log') + 1, auto_update=True)
    no_update_log = Log(name='no_update_log', init_value=0, compute_value=lambda logger: logger.get('no_update_log') + 1, auto_update=False)
    use_update = Log(name='use_update', init_value=0, compute_value=lambda logger: logger.get('update_log') + 1)
    no_use_update = Log(name='no_use_update', init_value=0, compute_value=lambda logger: logger.get('no_update_log') + 1)

    my_logger = Logger()

    my_logger.add_log(update_log)
    my_logger.add_log(no_update_log)
    my_logger.add_log(use_update)
    my_logger.add_log(no_use_update)

    my_logger.get('update_log')
    my_logger.next_index()
    my_logger.get('use_update')
    my_logger.get('no_use_update')
