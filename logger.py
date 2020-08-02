from typing import Callable, List, Union, Dict
import numpy as np
import warnings

class Log:
    def __init__(self, name, init_value=None, compute_value: Callable = None, index=0, auto_update=False, tmp_after_effect=None):
        self.value = init_value
        self.index = index
        self.name = name
        self.auto_update = auto_update
        self.compute_value = None
        if compute_value is not None:
            self.compute_value = compute_value
        self.init_value = init_value

    def compute_and_update(self, logger, index):
        self.index = index
        self.value = self.compute_value(logger)
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
        self.end_fun_list = []  # a list of functions to be computed once at the end of the index (at the beginning of the next index)
        self.logs: Dict[str, Log] = {}
        if dict_logs != {}:
            self.next_index(dict_logs)
        # [self.logs[k].set_value(v, self.batch_index) for k, v in logs.items()]

    def add_end_fun(self, effect: Callable):
        self.end_fun_list.append(effect)

    def next_index(self, dict_logs: Dict = None):
        """
        Update the logs that are passed to this function (that is, the dict logs, not the computed ones)
        You cannot pass computed logs here, as they need to be computed, not passed.
        """
        for i in self.end_fun_list:
            i(self)
        self.end_fun_list = []
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
        self.logs[log.name].index = self.index - 1
        self.logs[log.name].value = log.init_value if log.init_value is not None else log.value

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

    def set_value(self, name, value):
        self.logs[name].value = value
        self.logs[name].index = self.index
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
    # Initializing a value only matters if the computation refer to itself.
    # The computation is gonna be performed even when the value is initialized
    logs = {'labels': np.array([1, 1, 1]), 'predicted': np.array([1, 1, 1]), 'new_value': 100}
    my_logger.next_index(logs)
    auto_log = Log(name='auto_log', init_value=0, compute_value=lambda logger: logger.get('new_value') + 1)
    my_logger.add_log(auto_log)
    assert my_logger.get('auto_log') == 101

    # Having the init_value is meaningless if the value does not refer to itself
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
    assert my_logger.get('acc_log') == 1  # acc_log uses a stale version of labels but still works (gives a warning)



#####################################################
    ## Add value to itself, remember that it only compute it for a new index
    # If a log refers to itself, then it must be initialized
    logs = {'labels': np.array([1, 1, 1]), 'predicted': np.array([1, 1, 1]), 'new_value': 100}
    my_logger = Logger(logs)
    auto_log = Log(name='auto_log', init_value=0, compute_value=lambda logger: logger.logs['auto_log'].value + 1)
    my_logger.add_log(auto_log)
    assert my_logger.get('auto_log') == 1
    assert my_logger.get('auto_log') == 1
    my_logger.next_index()
    assert my_logger.get('auto_log') == 2

    my_logger.next_index(logs)  # here we update the index of only the element in logs, but auto_log is computed anyway
    assert my_logger.get('auto_log') == 3
    assert my_logger.get('auto_log') == 3

    def comp(logger):
        logger.get('mat')[0, 1] += 100
        return logger.get('mat')
    my_logger.add_log(Log(name='mat', init_value=np.zeros((3, 3)), compute_value=comp))
    my_logger.next_index(logs)
    assert np.all(my_logger.get('mat') == np.array([[0, 100, 0], [0, 0, 0], [0, 0, 0]]))

    ## In some cases we have auto updated logs
    update_log = Log(name='update_log', init_value=0, compute_value=lambda logger: logger.get('update_log') + 1, auto_update=True)
    no_update_log = Log(name='no_update_log', init_value=0, compute_value=lambda logger: logger.get('no_update_log') + 1, auto_update=False)
    use_update = Log(name='use_update', init_value=0, compute_value=lambda logger: logger.get('update_log'))
    no_use_update = Log(name='no_use_update', init_value=0, compute_value=lambda logger: logger.get('no_update_log'))

    my_logger = Logger()

    my_logger.add_log(update_log)
    my_logger.add_log(no_update_log)
    my_logger.add_log(use_update)
    my_logger.add_log(no_use_update)

    assert my_logger.get('update_log') == 1
    my_logger.next_index()
    my_logger.next_index()
    assert my_logger.get('use_update') == 3
    assert my_logger.get('no_use_update') == 1

    # If we add_log, the value will be reset to its init_value if present
    update_log = Log(name='update_log', init_value=0, compute_value=lambda logger: logger.get('update_log') + 1, auto_update=True)
    my_logger = Logger()
    my_logger.add_log(update_log)
    assert my_logger.get('update_log') == 1
    my_logger.next_index()
    assert my_logger.get('update_log') == 2
    my_logger.add_log(update_log)
    assert my_logger.get('update_log') == 1

    # When we get a log, we may want to reset/change the value to some other logs. We can do that:
    my_logger = Logger()
    logs = {'labels': np.array([0, 1, 1]), 'predicted': np.array([1, 1, 0])}
    my_logger = Logger(logs)

    acc_log = Log(name='auto_log', init_value=0, compute_value=lambda logger: logger.get('auto_log') + 1, auto_update=True)
    def return_and_reset(logger):
        x = logger.get('auto_log')
        logger.set_value('auto_log', -10)
        return x
    return_log = Log(name='return_log', init_value=0, compute_value=return_and_reset)
    another_return_log = Log(name='another_return_log', init_value=0, compute_value=lambda logger: logger.get('auto_log'))
    my_logger.add_log(return_log)
    my_logger.add_log(acc_log)
    my_logger.add_log(another_return_log)
    my_logger.next_index()
    my_logger.next_index()
    my_logger.next_index()
    assert my_logger.get('auto_log') == 3
    assert my_logger.get('return_log') == 3
    assert my_logger.get('auto_log') == -10
    assert my_logger.get('another_return_log') == -10  #  we may want return_log to not use the resetted value!
    my_logger.next_index()
    assert my_logger.get('auto_log') == -9
    my_logger.get('return_log')
    # however, this may affect the behaviour for other logs.


    # What we really want is a way to say that at the end of the index, we should do something:

    my_logger = Logger()
    logs = {'labels': np.array([0, 1, 1]), 'predicted': np.array([1, 1, 0])}
    my_logger = Logger(logs)

    def return_and_reset(logger):
        x = logger.get('auto_log')
        logger.add_end_fun(lambda logger: logger.set_value('auto_log', 0))
        return x
    acc_log = Log(name='auto_log', init_value=0, compute_value=lambda logger: logger.get('auto_log') + 1, auto_update=True)
    another_return_log = Log(name='another_return_log', init_value=0, compute_value=lambda logger: logger.get('auto_log'))
    return_log = Log(name='return_log', init_value=0, compute_value=return_and_reset)

    my_logger.add_log(return_log)
    my_logger.add_log(acc_log)
    my_logger.add_log(another_return_log)
    my_logger.next_index()
    my_logger.next_index()
    my_logger.next_index()
    assert my_logger.get('auto_log') == 3
    assert my_logger.get('return_log') == 3
    assert my_logger.get('auto_log') == 3  # we will reset this at the end of the index!
    assert my_logger.get('another_return_log') == 3  # we may want return_log to not use the resetted value!
    my_logger.next_index()
    assert my_logger.get('auto_log') == 1
    assert my_logger.get('return_log') == 1
    assert my_logger.get('another_return_log') == 1

    # however, this may affect the behaviour for other logs.

