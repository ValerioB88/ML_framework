"""
Ports of Callback classes from the Keras library.
"""
import seaborn as sn
from sty import fg, rs, ef
from abc import ABC
import numpy as np
import neptune.new as neptune
import framework_utils
import pathlib
import matplotlib.pyplot as plt
import torch
from time import time
import framework_utils as utils
import pandas as pd
import signal, os
# from neptunecontrib.api import log_chart
import time
import math


class CallbackList(object):
    """Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
    """

    def __init__(self, callbacks):
        self.callbacks = [c for c in callbacks]

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def set_optimizer(self, model):
        for callback in self.callbacks:
            callback.set_optimizer(model)

    def set_loss_fn(self, model):
        for callback in self.callbacks:
            callback.set_loss_fn(model)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_training_step_end(self, batch, logs=None):
        """Called after training is finished, but before the batch is ended.
                # Arguments
                    batch: integer, index of batch within the current epoch.
                    logs: dictionary of logs.
                """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_training_step_end(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_training_step_end(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class StopFromUserInput(Callback):
    stop_next_iter = False

    def __init__(self):
        super().__init__()
        signal.signal(signal.SIGINT, self.handler)  # CTRL + C

    def handler(self, signum, frame):
        self.stop_next_iter = True

    def on_batch_end(self, batch, logs=None):
        if self.stop_next_iter:
            logs['stop'] = True
            print('Stopping from user input')

class TriggerActionWhenReachingValue(Callback):
    def __init__(self, value_to_reach, metric_name, mode='max', patience=1, check_after_batch=True, action=None, action_name='', check_every=1):
        self.patience = patience
        self.check_every = check_every
        self.action = action
        self.action_name = action_name
        self.count_patience = 0
        self.mode = mode
        self.value_to_reach = value_to_reach
        self.metric_name = metric_name
        self.check_after_batch = check_after_batch
        self.check_idx = 0
        print(fg.green + f"Action [{self.action_name}] when [{self.metric_name}] has reached value {'higher' if self.mode == 'max' else 'lower'} than [{self.value_to_reach}] for {self.patience} checks (checked every {self.check_every} {'batches' if self.check_after_batch else 'epoches'})" + rs.fg)
        super().__init__()

    def compare(self, metric, value):
        if self.mode == 'max':
            return metric >= value
        if self.mode == 'min':
            return metric <= value

    def check_and_stop(self, logs=None):
        self.check_idx += 1
        if self.check_idx >= self.check_every:
            self.check_idx = 0
            if self.compare(logs[self.metric_name], self.value_to_reach):
                self.count_patience += 1
                # print(f'PATIENCE +1 : {self.count_patience}/{self.patience}')
                if self.count_patience >= self.patience:
                    logs['stop'] = True
                    print(fg.green + f"\nMetric [{self.metric_name}] has reached value {'higher' if self.mode == 'max' else 'lower'} than [{self.value_to_reach}]. Action [{self.action_name}] triggered" + rs.fg)
            else:
                self.count_patience = 0

    def on_batch_end(self, batch, logs=None):
        if self.check_after_batch:
            self.check_and_stop(logs)

    def on_epoch_end(self, epoch, logs=None):
        if not self.check_after_batch:
            self.check_and_stop(logs)

class TriggerActionWithPatience(Callback):
    def __init__(self, mode='min', min_delta=0, patience=10, min_delta_is_percentage=False, reaching_goal=None, metric_name='nept/mean_acc', check_every=100, triggered_action=None, action_name='', alpha=1, weblogger=False, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.triggered_action = triggered_action
        self.mode = mode  # mode refers to what are you trying to reach.
        self.check_every = check_every
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_iters = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, min_delta_is_percentage)
        self.reaching_goal = reaching_goal
        self.metric_name = metric_name
        self.action_name = action_name
        self.first_iter = True
        self.alpha = alpha
        self.weblogger = weblogger
        self.exp_metric = None
        # reaching goal: if the metric stays higher/lower (max/min) than the goal for patience steps
        if reaching_goal is not None:
            self.best = reaching_goal
            # self.mode = 'min'
        self.patience = self.patience // self.check_every
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False
        self.string = f'Action {self.action_name} for metric [{self.metric_name}] <> {self.reaching_goal if self.reaching_goal is not None else self.mode}, checking every [{self.check_every} batch iters], patience: {self.patience} [corresponding to [{patience}] batch iters]]' + ((f' with ExpMovAvg alpha: {self.alpha}') if self.alpha != 1 else '')
        print(f'Set up action: {self.string}')

    def on_batch_end(self, batch, logs=None):
        if self.metric_name not in logs:
            return True

        if self.first_iter:
            self.exp_fun = framework_utils.ExpMovingAverage(logs[self.metric_name], alpha=self.alpha)
            self.first_iter = False
            return

        if logs['tot_iter'] % self.check_every == 0:
            metrics = self.exp_fun(logs[self.metric_name]).value
            print(f"Iter: {logs['tot_iter']}, Metric: {logs[self.metric_name]}, Exp Metric: {metrics}") if self.verbose else None

            if isinstance(self.weblogger, neptune.run.Run):
                self.weblogger[f'{self.metric_name} - a: {self.alpha} - action: {self.action_name}'].log(metrics)
            if self.best is None:
                self.best = metrics
                return

            if np.isnan(metrics):
                return
            if self.reaching_goal is not None:
                if self.is_better(metrics, self.best):
                    self.num_bad_iters += 1
                else:
                    self.num_bad_iters = 0

            else:
                if self.is_better(metrics, self.best):
                    self.num_bad_iters = 0  # bad epochs: does not 'improve'
                    self.best = metrics
                else:
                    self.num_bad_iters += 1
            print(f"Num Bad Iter: {self.num_bad_iters}") if self.verbose else None
            print(f"Patience: {self.num_bad_iters}/{self.patience}") if (self.verbose or self.patience - self.num_bad_iters < 20) else None

            if self.num_bad_iters >= self.patience:
                print(f"Action triggered: {self.string}")
                self.triggered_action(logs, self)
                # needs to reset itself
                self.num_bad_iters = 0
        else:
            print(f"Not updating now {self.check_every - (logs['tot_iter'] % self.check_every)}") if self.verbose else None

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)


class ClipGradNorm(Callback):
    def __init__(self, net, max_norm):
        self.net = net
        self.max_norm = max_norm

    def on_training_step_end(self, batch, logs=None):
        torch.nn.utils.clip_grad_norm(self.net.parameters(), self.max_norm)


from torch.optim.lr_scheduler import ReduceLROnPlateau
class PlateauLossLrScheduler(Callback):
    def __init__(self, optimizer, check_batch=False, patience=2, loss_metric='loss'):
        self.loss_metric = loss_metric
        self.scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        self.last_lr = [i['lr'] for i in self.scheduler.optimizer.param_groups]
        self.check_batch = check_batch

    def on_epoch_end(self, epoch, logs=None):
        if not self.check_batch:
            self.check_and_update(logs)


    def check_and_update(self, logs):
        self.scheduler.step(logs[self.loss_metric])
        if self.last_lr != [i['lr'] for i in self.scheduler.optimizer.param_groups]:
            print((fg.blue + "learning rate: {} => {}" + rs.fg).format(self.last_lr, [i['lr'] for i in self.scheduler.optimizer.param_groups]))
            self.last_lr = [i['lr'] for i in self.scheduler.optimizer.param_groups]

    def on_batch_end(self, batch, logs):
        if self.check_batch:
            self.check_and_update(logs)


class CallLrScheduler(Callback):
    def __init__(self, scheduler, step_epoch=True, step_batch=False):
        self.scheduler = scheduler
        self.step_epoch = step_epoch
        self.step_batch = step_batch
        self.last_lr = [i['lr'] for i in self.scheduler.optimizer.param_groups]

    def step(self):
        lr = self.scheduler.get_last_lr()
        self.scheduler.step()
        if self.last_lr != [i['lr'] for i in self.scheduler.optimizer.param_groups]:
            print((fg.blue + "learning rate: {} => {}" + rs.fg).format(self.last_lr, [i['lr'] for i in self.scheduler.optimizer.param_groups]))
            self.last_lr = [i['lr'] for i in self.scheduler.optimizer.param_groups]

    def on_batch_end(self, batch, logs=None):
        if self.step_batch:
            self.step()

    def on_epoch_end(self, epoch, logs=None):
        if self.step_epoch:
            self.step()

class StopWhenMetricIs(Callback):
    def __init__(self, value_to_reach, metric_name, check_after_batch=True):
        self.value_to_reach = value_to_reach
        self.metric_name = metric_name
        self.check_after_batch = check_after_batch
        print(fg.cyan + f"This session will stop when metric [{self.metric_name}] has reached the value  [{self.value_to_reach}]" + rs.fg)
        super().__init__()

    def check_and_stop(self, logs=None):
        if logs[self.metric_name] >= self.value_to_reach:
            logs['stop'] = True
            print(f'Metric [{self.metric_name}] has reached the value [{self.value_to_reach}]. Stopping')

    def on_batch_end(self, batch, logs=None):
        if self.check_after_batch:
            self.check_and_stop(logs)

    def on_epoch_end(self, epoch, logs=None):
        if not self.check_after_batch:
            self.check_and_stop(logs)

class SaveModel(Callback):
    def __init__(self, net, output_path, loss_metric_name='loss', log_in_weblogger=False, epsilon_loss=0.1, min_iter=100):
        self.output_path = output_path
        self.net = net
        self.log_in_weblogger = log_in_weblogger
        self.last_loss = np.inf
        self.last_iter = 0
        self.min_iter = min_iter
        self.epsilone_loss = epsilon_loss
        self.loss_metric_name = loss_metric_name
        super().__init__()

    def save_model(self, path):
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        print(fg.yellow + ef.inverse + 'Saving model in {}'.format(path) + rs.fg + rs.inverse)
        torch.save(self.net.state_dict(), path)

    def on_batch_end(self, batch, logs=None):
        if self.output_path is not None:
            if ((self.last_loss - logs[self.loss_metric_name]) > self.epsilone_loss) and ((logs['tot_iter'] - self.last_iter) > self.min_iter):
                self.last_iter = logs['tot_iter']
                self.last_loss = logs[self.loss_metric_name].value  ## ouch! You cannot reimpliement assignment operator!
                self.save_model(os.path.splitext(self.output_path)[0] + f'_checkpoint' + os.path.splitext(self.output_path)[1])

    def on_train_end(self, logs=None):
        if self.output_path is not None:
            self.save_model(self.output_path)


#
# class AccuracyMetrics(Callback):
#     def __init__(self, use_cuda, log_every, log_text='', pred_str='y_pred', true_str='y_true'):
#         self.use_cuda = use_cuda
#         self.log_every = log_every
#         self.pred_str = pred_str
#         self.true_str = true_str
#         self.log_text = log_text
#         self.correct_train, self.total_samples = 0, 0
#         super().__init__()
#
#     def update_classic_logs(self, batch_logs):
#
#         self.correct_train += ((batch_logs[self.pred_str].cuda() if self.use_cuda else batch_logs[self.pred_str]) ==
#                                (batch_logs[self.true_str].cuda() if self.use_cuda else batch_logs[self.true_str])).sum().item()
#         self.total_samples += batch_logs[self.true_str].size(0)

class PrintLogs(Callback, ABC):
    def __init__(self, id, plot_every=100, plot_at_end=True):
        self.id = id
        # self.running_logs = {}
        self.last_iter = 0
        self.plot_every = plot_every
        self.plot_at_end = plot_at_end

    def on_training_step_end(self, batch, logs=None):
        # if self.id in logs:
        #     if self.id not in self.running_logs:
        #         self.running_logs[self.id] = []
        #     self.running_logs[self.id].append(logs[self.id])

            if logs['tot_iter'] - self.last_iter > self.plot_every:
                # self.print_logs(self.get_value(self.running_logs[self.id]), logs)
                self.print_logs(logs[self.id], logs)
                # self.running_logs[self.id] = []
                self.last_iter = logs['tot_iter']

    def on_train_end(self, logs=None):
        if self.plot_at_end:
            self.print_logs(logs[self.id], logs)



class PrintConsole(PrintLogs):
    def __init__(self, endln="\n", **kwargs):
        self.endln = endln
        super().__init__(**kwargs)

    def print_logs(self, values, logs):
        if values is str:
            value_format = values
        else:
            value_format = f'{values:.3}'
        print(fg.cyan + f'{self.id}: {value_format}' + rs.fg, end=self.endln)




class PrintNeptune(PrintLogs):
    def __init__(self,  weblogger, convert_str=False, **kwargs):
        self.convert_str = convert_str
        self.weblogger = weblogger
        super().__init__(**kwargs)

    def print_logs(self, values, logs):
        if isinstance(self.weblogger,  neptune.run.Run):
            if self.convert_str:
                self.weblogger[self.id].log(str(values))
            else:
                self.weblogger[self.id].log(values)


# class RunningAccuracyMetrics(AccuracyMetrics):
#     def __init__(self, use_cuda, log_every, log_text='', **kwargs):
#         super().__init__(use_cuda, log_every, log_text, **kwargs)
#         self.running_loss = 0
#         self.init_classic_logs()
#
#     def update_classic_logs(self, batch_logs):
#         super().update_classic_logs(batch_logs)
#         self.running_loss += batch_logs['loss']
#
#     def compute_mean_loss_acc(self, num_iter):
#         mean_loss = self.running_loss / num_iter
#         mean_acc = 100 * self.correct_train / self.total_samples
#         return mean_loss, mean_acc
#
#     def init_classic_logs(self):
#         self.correct_train, self.total_samples, self.running_loss = 0, 0, 0

class ProgressBar(Callback):
    def __init__(self, l):
        self.length_bar = l

    def on_training_step_end(self, batch_index, batch_logs=None):
        framework_utils.progress_bar(batch_index, self.length_bar)

#
# class StandardMetrics(RunningAccuracyMetrics):
#     num_iter = 0
#     time = None
#
#     def __init__(self, print_it=True, metrics_prefix='', weblogger=False, size_dataset=None, **kwargs):
#         self.weblogger = weblogger
#         self.print_it = print_it
#         self.metrics_prefix = metrics_prefix
#         self.size_dataset = size_dataset
#         super().__init__(**kwargs)
#
#     def on_train_begin(self, logs=None):
#         self.time = time.time()
#
#     def on_training_step_end(self, batch_index, batch_logs=None):
#         self.num_iter += 1
#         self.update_classic_logs(batch_logs)
#         if batch_logs['tot_iter'] % self.log_every == 0:
#             batch_logs[f'{self.metrics_prefix}/mean_loss'], batch_logs[f'{self.metrics_prefix}/mean_acc'] = self.compute_mean_loss_acc(num_iter=self.num_iter)
#             self.num_iter = 0
#             if self.print_it:
#                 framework_utils.progress_bar(batch_index, self.size_dataset, msg=('loss: {:.4f}, acc: ' + bg.white + fg.black+ '{:.4f}%' + rs.fg+rs.bg).format(batch_logs[f'{self.metrics_prefix}/mean_loss'], batch_logs[f'{self.metrics_prefix}/mean_acc']))
#                 # print(('[iter{}, {:.2f}sec] loss: {:.4f}, train_acc: ' + bg.white + fg.black+ '{:.4f}%' + rs.fg+rs.bg).format(batch_logs['tot_iter'], time.time()- self.time, batch_logs[f'{self.metrics_prefix}/mean_loss'], batch_logs[f'{self.metrics_prefix}/mean_acc']))
#
#                 self.time = time.time()
#             metric2 = f'accuracy {self.log_text}'
#             metric1 = 'loss'
#             if isinstance(self.weblogger, neptune.run.Run):
#                 self.weblogger[metric1].log(batch_logs[f'{self.metrics_prefix}/mean_loss'])
#                 self.weblogger[metric2].log(batch_logs[f'{self.metrics_prefix}/mean_acc'])
#             self.init_classic_logs()


class DuringTrainingTest(Callback):
    test_time = 0
    num_tests = 0

    def __init__(self, testing_loaders, every_x_epochs=None, every_x_iter=None, every_x_sec=None, weblogger=0, multiple_sec_of_test_time=None, auto_increase=False, log_text='', use_cuda=None, call_run=None):
        self.testing_loaders = testing_loaders
        self.use_cuda = use_cuda
        self.every_x_epochs = every_x_epochs
        self.auto_increase = auto_increase
        self.every_x_iter = every_x_iter
        self.every_x_sec = every_x_sec
        if self.auto_increase:
            self.every_x_sec = 20
        self.weblogger = weblogger
        self.log_text = log_text
        self.call_run = call_run
        self.time_from_last_test = None
        self.multiple_sec_of_test_time = multiple_sec_of_test_time

    def on_train_begin(self, logs=None):
        self.time_from_last_test = time.time()

    def get_callbacks(self, log=''):
        return [
            StopWhenMetricIs(value_to_reach=100, metric_name='tot_iter', check_after_batch=True),
            StopWhenMetricIs(value_to_reach=0, metric_name='epoch', check_after_batch=False),
            TotalAccuracyMetric(use_cuda=self.use_cuda,
                                weblogger=self.weblogger, log_text=self.log_text + log)]

    def run_tests(self, logs, last_test=False):
        start_test_time = time.time()
        print(fg.green, end="")
        print(f"################ TEST DURING TRAIN - NUM {self.num_tests} ################")
        print(rs.fg, end="")

        def test(testing_loader, log='', last_test=False):
            print(f"Testing " + fg.green + f"[{testing_loader.dataset.name_generator}]" + rs.fg)
            mid_test_cb = self.get_callbacks(log)
            if last_test:
                mid_test_cb += [ComputeConfMatrix(num_classes=len(testing_loader.dataset.classes),
                                                  weblogger=self.weblogger,
                                                  weblog_text='ConfMatrix',
                                                  class_names=testing_loader.dataset.classes)]

            with torch.no_grad():
                _, logs = self.call_run(testing_loader,
                                        train=False,
                                        callbacks=mid_test_cb,
                                        )

        print("TEST IN EVAL MODE")
        self.model.eval()
        for testing_loader in self.testing_loaders:
            test(testing_loader, log=f' EVALmode [{testing_loader.dataset.name_generator}]', last_test=last_test)

        self.model.train()
        print("TEST IN TRAIN MODE")
        for testing_loader in self.testing_loaders:
            test(testing_loader, log=f' TRAINmode [{testing_loader.dataset.name_generator}]', last_test=last_test)

        self.num_tests += 1

        self.time_from_last_test = time.time()
        self.test_time = time.time() - start_test_time
        if self.auto_increase and 'tot_iter' in logs:
            self.every_x_sec = self.test_time + 0.5 * self.test_time * math.log(logs['tot_iter']+1, 1.2)
            print("Test time is {:.4f}s, next test is gonna happen in {:.4f}s".format(self.test_time, self.every_x_sec))

        if self.multiple_sec_of_test_time:
            print("Test time is {:.4f}s, next test is gonna happen in {:.4f}s".format(self.test_time, self.test_time*self.multiple_sec_of_test_time))
        print(fg.green, end="")
        print("#############################################")
        print(rs.fg, end="")

    def on_epoch_begin(self, epoch, logs=None):
        if (self.every_x_epochs is not None and epoch % self.every_x_epochs == 0) or epoch==0:
            print(f"\nTest every {self.every_x_epochs} epochs")
            self.run_tests(logs)

    def on_batch_end(self, batch, logs=None):
        if (self.every_x_iter is not None and logs['tot_iter'] % self.every_x_iter) or \
                (self.every_x_sec is not None and self.every_x_sec < time.time() - self.time_from_last_test) or \
                (self.multiple_sec_of_test_time is not None and time.time() - self.time_from_last_test > self.multiple_sec_of_test_time * self.test_time):
            if (self.every_x_iter is not None and logs['tot_iter'] % self.every_x_iter):
                print(f"\nTest every {self.every_x_iter} iterations")
            if (self.every_x_sec is not None and self.every_x_sec < time.time() - self.time_from_last_test):
                print(f"\nTest every {self.every_x_sec} seconds ({time.time() -self.time_from_last_test} secs passed from last test)")
            if (self.multiple_sec_of_test_time is not None and time.time() - self.time_from_last_test > self.multiple_sec_of_test_time * self.test_time):
                print(f"\nTest every {self.multiple_sec_of_test_time * self.test_time} seconds ({time.time() - self.time_from_last_test} secs passed from last test)")

            self.run_tests(logs)

    def on_train_end(self, logs=None):
        print("End training")
        self.run_tests(logs, last_test=True)


class EndEpochStats(Callback):
    timer_train = None
    def on_train_begin(self, logs=None):
        self.timer_train = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        print(fg.red + '\nEpoch: {}'.format(epoch) + rs.fg)
        self.timer_epoch = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(fg.red, end="")
        print("End epoch. Tot iter: [{}] - in [{:.4f}] seconds - tot [{:.4f}] seconds\n".format(logs['tot_iter'], time.time() - self.timer_epoch, time.time() - self.timer_train))
        print(rs.fg, end="")

#
# class TotalAccuracyMetric(AccuracyMetrics):
#     def __init__(self, use_cuda, weblogger=True, log_text='', **kwargs):
#         super().__init__(use_cuda, log_every=None, log_text=log_text, **kwargs)
#         self.weblogger = weblogger
#         self.start = time.time()
#
#     def on_training_step_end(self, batch_index, batch_logs=None):
#         super().on_training_step_end(batch_index, batch_logs)
#         self.update_classic_logs(batch_logs)
#
#     def on_train_end(self, logs=None):
#         logs['total_accuracy'] = 100.0 * self.correct_train / self.total_samples
#         print((fg.cyan + 'Total Accuracy for [{}] samples, [{}] iter, ' + ef.inverse + ef.bold + '[{}]' + rs.inverse + ': ' + ef.inverse  + '{:.4f}%'  + rs.inverse + rs.bold_dim + fg.cyan + ' - in {:.4f} seconds\n' + rs.fg).format(self.total_samples, logs["tot_iter"], self.log_text, logs["total_accuracy"], time.time() - self.start))
#
#         metric_str = 'Metric/{} Acc'.format(self.log_text)
#
#         if isinstance(self.weblogger, neptune.run.Run):
#             self.weblogger[metric_str].log(logs['total_accuracy'])

class ComputeConfMatrix(Callback):
    def __init__(self, num_classes, reset_every=None, weblogger=0, weblog_text='', class_names=None):

        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.log_text_plot = weblog_text
        self.reset_every = reset_every
        self.class_names = class_names
        self.num_iter = 0
        self.weblogger = weblogger
        super().__init__()

    def on_training_step_end(self, batch, logs=None):
        if self.reset_every is not None and logs['tot_iter'] % self.reset_every == 0:
            self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
            self.num_iter = 0
        try:
            for t, p in zip(logs['y_true'].view(-1), logs['y_pred'].view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1
        except IndexError:
            print(fg.red + "Index error during confusion matrix calculation. This shouldn't happen (unless you are on the local machine)!" + rs.fg)
        self.num_iter += 1

    def on_train_end(self, logs=None):
        conf_mat_acc = (self.confusion_matrix / self.confusion_matrix.sum(1)[:, None]).numpy()

        if self.weblogger:
            figure = plt.figure(figsize=(20, 15))
            sn.heatmap(conf_mat_acc, annot=True, fmt=".2f", xticklabels=self.class_names, yticklabels=self.class_names, annot_kws={"size": 15}, vmin=0, vmax=1)  # font size
            plt.yticks(np.arange(len(self.class_names)) + 0.5, self.class_names, rotation=0, fontsize="10", va="center")
            plt.ylabel('truth')
            plt.xlabel('predicted')
            plt.title(self.log_text_plot + ' last {} iters'.format(self.num_iter), size=22)
            metric_str = 'Confusion Matrix'  # {}'.format(self.log_text_plot)
            if self.weblogger == 1:
                wandb.log({metric_str: wandb.Image(plt)})
            if isinstance(self.weblogger, neptune.run.Run):
                self.weblogger[metric_str].log(figure)
                # log_chart(name=metric_str, chart=figure)

            plt.close()


class RollingAccEachClassWeblog(Callback):
    def __init__(self, log_every, num_classes, weblog_text='', weblogger=1):
        self.log_every = log_every
        self.confusion_matrix = torch.zeros(num_classes, num_classes)
        self.neptune_text = weblog_text
        self.num_classes = num_classes
        self.weblogger = weblogger
        super().__init__()

    def on_training_step_end(self, batch, logs=None):
        for t, p in zip(logs['y_true'].view(-1), logs['y_pred'].view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1
        if logs['tot_iter'] % self.log_every == 0:
            correct_class = self.confusion_matrix.diag() / self.confusion_matrix.sum(1)
            self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
            if correct_class is not None:
                for idx, cc in enumerate(correct_class.numpy()):
                    metric_str = f'Metric/Class Acc Training {idx} - [{self.neptune_text}]'
                    if self.weblogger == 1:
                        wandb.log({metric_str: cc * 100 if not np.isnan(cc) else -1})  # step=logs['tot_iter'])
                    if isinstance(self.weblogger, neptune.run.Run):
                        self.weblogger[metric_str].log(cc * 100 if not np.isnan(cc) else -1)


class PlotIterationsInfo(Callback):
    def __init__(self, time_every=100, endl=""):
        super().__init__()
        self.time_every = time_every
        self.start_time = 0
        self.endl = endl

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        if logs['tot_iter'] % self.time_every == self.time_every - 1:
            print('{} - {:.4f}s'.format(logs["tot_iter"], time.time() - self.start_time), end=self.endl)
            self.start_time = time.time()




class GenericDataFrameSaver(Callback):
    @staticmethod
    def build_columns(cat_to_save):
        array = [np.concatenate(np.array([np.array(['softmax', 'softmax']),
                                          np.repeat(['softmax'], len(cat_to_save)),
                                          np.array(['logits', 'logits']),
                                          np.repeat(['logits'], len(cat_to_save))])),
                 np.concatenate(np.array([np.array(['main_softm', 'main_softm']),
                                          np.repeat(['useful_softm'], len(cat_to_save)),
                                          np.array(['main_logits', 'main_logits']),
                                          np.repeat(['useful_logits'], len(cat_to_save))])),
                 np.concatenate(np.array([np.array(['max_softm', 'softm_class']),
                                          cat_to_save,
                                          np.array(['max_logits', 'logits_class']),
                                          cat_to_save]))] ## this currently doesn't work!!
        return array

    def __init__(self, num_classes, use_cuda, network_name, size_canvas, weblogger=0, log_text_plot='', output_and_softmax=False):
        super().__init__()
        self.output_and_softmax = output_and_softmax
        self.size_canvas = size_canvas
        self.num_classes = num_classes
        self.network_name = network_name
        self.additional_logs_names = []

        self.index_dataframe = ['net', 'class_name', 'class_output']
        self.column_names = ['is_correct']
        if output_and_softmax:  # ToDo: output_and_softmax True doesn't work currently
            self.column_names.extend(self.build_columns(['class {}'.format(i) for i in range(self.num_classes)]))
        self.rows_frames = []
        self.use_cuda = use_cuda
        self.weblogger = weblogger
        self.log_text_plot = log_text_plot

    def _get_additional_logs(self, logs, c):
        return []

    def _compute_and_log_metrics(self, data_frame):
        return data_frame

    def on_training_step_end(self, batch, logs=None):
        output_batch_t = logs['output']
        labels_batch_t = logs['y_true']
        predicted_batch_t = logs['y_pred']

        correct_batch_t = (utils.make_cuda(predicted_batch_t, self.use_cuda) == utils.make_cuda(labels_batch_t, self.use_cuda))
        labels = labels_batch_t.tolist()
        predicted_batch = predicted_batch_t.tolist()
        correct_batch = correct_batch_t.tolist()

        if self.output_and_softmax:
            softmax_batch_t = torch.softmax(utils.make_cuda(output_batch_t, self.use_cuda), 1)
            softmax_batch = np.array(softmax_batch_t.tolist())
            output_batch = np.array(output_batch_t.tolist())

        for c, _ in enumerate(correct_batch_t):
            correct = correct_batch[c]
            label = labels[c]
            predicted = predicted_batch[c]

            add_logs = self._get_additional_logs(logs, c)
            assert len(add_logs) == len(self.additional_logs_names), "The additiona log names and additional logs metric length do not match"

            if self.output_and_softmax:
                softmax_all_cat = softmax_batch[c]
                output = output_batch[c]
                softmax = softmax_batch[c]
                softmax_correct_category = softmax[labels[c]]
                output_correct_category = output[labels[c]]
                max_softmax = np.max(softmax)
                max_output = np.max(output)

                assert softmax_correct_category == max_softmax if correct else True, 'softmax values: {}, is correct? {}'.format(softmax, correct)
                assert softmax_correct_category != max_softmax if not correct else True, 'softmax values: {}, is correct? {}'.format(softmax, correct)
                assert predicted == label if correct else predicted != label, 'softmax values: {}, is correct? {}'.format(softmax, correct)

                self.rows_frames.append([self.network_name, int(label), int(predicted), *add_logs, max_softmax, softmax_correct_category, *softmax, max_output, output_correct_category, *output, correct])
            else:
                self.rows_frames.append([self.network_name, int(label), int(predicted), *add_logs, correct])

    def on_train_end(self, logs=None):
        data_frame = pd.DataFrame(self.rows_frames)
        data_frame = data_frame.set_index([i for i in range(len(self.index_dataframe))])
        data_frame.index.names = self.index_dataframe
        data_frame.columns = self.column_names

        data_frame = self._compute_and_log_metrics(data_frame)
        logs['dataframe'] = data_frame



# for now this is only supported with the UnityImageSampelrGenerator


class PlotGradientWeblog(Callback):
    grad = []
    layers = []

    def __init__(self, net, log_every=50, plot_every=500, log_txt='', weblogger=1):
        self.log_every = log_every
        self.plot_every = plot_every
        self.log_txt = log_txt
        self.net = net
        self.weblogger = weblogger

        super().__init__()
        for n, p in self.net.named_parameters():
            if p.requires_grad and ("bias" not in n):
                self.layers.append(n)
        self.layers = [i.split('.weight')[0] for i in self.layers]

    def on_training_step_end(self, batch, logs=None):
        if logs['tot_iter'] % self.log_every == 0:
            ave_grads = []
            for n, p in self.net.named_parameters():
                if p.requires_grad and ("bias" not in n):
                    if p.grad is None:
                        print(f"Gradient for layer {n} is None. Skipping")
                    else:
                        ave_grads.append(p.grad.abs().mean())
            self.grad.append(ave_grads)

        if logs['tot_iter'] % self.plot_every == 0:
            lengrad = len(self.grad[0])
            figure = plt.figure(figsize=(10, 7))
            plt.plot(np.array(self.grad).T, alpha=0.3, color="b")
            plt.hlines(0, 0, lengrad + 1, linewidth=1, color="k")
            plt.xticks(range(0, lengrad, 1), self.layers, rotation=35)
            plt.xlim(xmin=0, xmax=lengrad)
            plt.xlabel("Layers")
            plt.ylabel("average gradient")
            plt.title("Gradient flow iter{}".format(logs['tot_iter']))
            plt.grid(True)
            metric_str = 'Hidden Panels/Gradient Plot [{}]'.format(self.log_txt)
            if self.weblogger == 1:
                wandb.log({metric_str: wandb.Image(plt)})
            if isinstance(self.weblogger, neptune.run.Run):
                self.weblogger[metric_str].log(figure)

            self.grad = []
            plt.close()
##

