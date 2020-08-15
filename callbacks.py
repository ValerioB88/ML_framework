"""
Ports of Callback classes from the Keras library.
"""
from tqdm import tqdm
import numpy as np
import torch
from collections import OrderedDict, Iterable
import warnings
import os
import csv
import io
import neptune

import framework_utils
import pathlib
import seaborn as sn
import matplotlib.pyplot as plt
import torch
from time import time
import framework_utils as utils
import pandas as pd
import signal, os

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

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

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

class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.optimiser = self.params['optimiser']

    def on_epoch_begin(self, epoch, logs=None):
        lrs = [self.schedule(epoch, param_group['lr']) for param_group in self.optimiser.param_groups]

        if not all(isinstance(lr, (float, np.float32, np.float64)) for lr in lrs):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        self.set_lr(epoch, lrs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if len(self.optimiser.param_groups) == 1:
            logs['lr'] = self.optimiser.param_groups[0]['lr']
        else:
            for i, param_group in enumerate(self.optimiser.param_groups):
                logs['lr_{}'.format(i)] = param_group['lr']

    def set_lr(self, epoch, lrs):
        for i, param_group in enumerate(self.optimiser.param_groups):
            new_lr = lrs[i]
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: setting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

class DefaultCallback(Callback):
    """Records metrics over epochs by averaging over each batch.

    NB The metrics are calculated with a moving model
    """
    def on_epoch_begin(self, batch, logs=None):
        self.seen = 0
        self.totals = {}
        self.metrics = ['loss'] + self.params['metrics']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 1) or 1
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.metrics:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.seen


class ProgressBarLogger(Callback):
    """TQDM progress bar that displays the running average of loss and other metrics."""
    def __init__(self):
        super(ProgressBarLogger, self).__init__()

    def on_train_begin(self, logs=None):
        self.num_batches = self.params['num_batches']
        self.verbose = self.params['verbose']
        self.metrics = ['loss'] + self.params['metrics']

    def on_epoch_begin(self, epoch, logs=None):
        self.target = self.num_batches
        self.pbar = tqdm(total=self.target, desc='Epoch {}'.format(epoch))
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        self.log_values = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.seen += 1

        for k in self.metrics:
            if k in logs:
                self.log_values[k] = logs[k]

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.pbar.update(1)
            self.pbar.set_postfix(self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        # Update log values
        self.log_values = {}
        for k in self.metrics:
            if k in logs:
                self.log_values[k] = logs[k]

        if self.verbose:
            self.pbar.update(1)
            self.pbar.set_postfix(self.log_values)

        self.pbar.close()


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
                 **kwargs):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        if mode not in ['auto', 'min', 'max']:
            raise ValueError('Mode must be one of (auto, min, max).')
        self.mode = mode
        self.monitor_op = None

        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self.optimiser = self.params['optimiser']
        self.min_lrs = [self.min_lr] * len(self.optimiser.param_groups)
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if len(self.optimiser.param_groups) == 1:
            logs['lr'] = self.optimiser.param_groups[0]['lr']
        else:
            for i, param_group in enumerate(self.optimiser.param_groups):
                logs['lr_{}'.format(i)] = param_group['lr']

        current = logs.get(self.monitor)

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.wait = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimiser.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.min_delta:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def in_cooldown(self):
        return self.cooldown_counter > 0


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options, which will be filled the value of `epoch` and keys in `logs`
    (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved
    with the epoch number and the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            raise ValueError('Mode must be one of (auto, min, max).')

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less

        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        torch.save(self.model.state_dict(), filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                torch.save(self.model.state_dict(), filepath)


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


class EarlyStopping(Callback):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False, reaching_goal=None, metric_name='nept/mean_acc', check_every=100):
        super().__init__()
        self.mode = mode  # mode refers to what are you trying to reach.
        self.check_every = check_every
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.reaching_goal = reaching_goal
        self.metric_name = metric_name
        # reaching goal: if the metric stays higher/lower (max/min) than the goal for patience steps
        if reaching_goal is not None:
            self.best = reaching_goal
            # self.mode = 'min'

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False
        print(f'Set up early stopping with metric [{self.metric_name}] < {self.reaching_goal}, checking every [{self.check_every}], patience [{patience}]')

    def on_batch_end(self, batch, logs=None):
        if batch % self.check_every == 0:
            metrics = logs[self.metric_name]
            if self.best is None:
                self.best = metrics
                return False

            if np.isnan(metrics):
                return True
            if self.reaching_goal is not None:
                if self.is_better(metrics, self.best):
                    self.num_bad_epochs += 1
                else:
                    self.num_bad_epochs = 0
            else:
                if self.is_better(metrics, self.best):
                    self.num_bad_epochs = 0  # bad epochs: does not 'improve'
                    self.best = metrics
                else:
                    self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                logs['stop'] = True
                print('Early Stopping')


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


class StopWhenMetricIs(Callback):
    def __init__(self, value_to_reach, metric_name):
        self.value_to_reach = value_to_reach
        self.metric_name = metric_name
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        if logs[self.metric_name] >= self.value_to_reach:
            logs['stop'] = True
            print(f'Metric [{self.metric_name}] has reached the value [{self.value_to_reach}]. Stopping')


class SaveModel(Callback):
    def __init__(self, net, output_path, log_in_neptune=False):
        self.output_path = output_path
        self.net = net
        self.log_in_neptune =log_in_neptune
        super().__init__()

    def on_train_end(self, logs=None):
        if self.output_path is not None:
            pathlib.Path(os.path.dirname(self.output_path)).mkdir(parents=True, exist_ok=True)
            print('Saving model in {}'.format(self.output_path))
            torch.save(self.net.state_dict(), self.output_path)
            # neptune.log_artifact(self.output_path, self.output_path) if self.log_in_neptune else None


class Metrics(Callback):
    def __init__(self, use_cuda, log_every, to_neptune=True, log_text=''):
        self.use_cuda = use_cuda
        self.log_every = log_every
        self.to_neptune = to_neptune
        self.log_text = log_text
        self.correct_train, self.total_samples = 0, 0
        super().__init__()

    def update_classic_logs(self, batch_logs):
        self.correct_train += ((batch_logs['y_pred'].cuda() if self.use_cuda else batch_logs['y_pred']) ==
                               (batch_logs['y_true'].cuda() if self.use_cuda else batch_logs['y_true'])).sum().item()
        self.total_samples += batch_logs['y_true'].size(0)


class RunningMetrics(Metrics):
    def __init__(self, use_cuda, log_every, to_neptune=True, log_text=''):
        super().__init__(use_cuda, log_every, to_neptune, log_text)
        self.running_loss = 0
        self.init_classic_logs()

    def update_classic_logs(self, batch_logs):
        super().update_classic_logs(batch_logs)
        self.running_loss += batch_logs['loss']

    def compute_mean_loss_acc(self):
        mean_loss = self.running_loss / self.log_every
        mean_acc = 100 * self.correct_train / self.total_samples
        return mean_loss, mean_acc

    def init_classic_logs(self):
        self.correct_train, self.total_samples, self.running_loss = 0, 0, 0


class StandardMetrics(RunningMetrics):
    def __init__(self, log_every=5, print_it=True, use_cuda=True, to_neptune=True, log_text='', metrics_prefix=''):
        super().__init__(use_cuda, log_every, to_neptune, log_text)
        self.print_it = print_it
        self.metrics_prefix = metrics_prefix

    def on_training_step_end(self, batch_index, batch_logs=None):
        super().on_batch_end(batch_index, batch_logs)
        self.update_classic_logs(batch_logs)
        if batch_index % self.log_every == 0:
            batch_logs[f'{self.metrics_prefix}/mean_loss'], batch_logs[f'{self.metrics_prefix}/mean_acc'] = self.compute_mean_loss_acc()
            if self.print_it:
                print('[iter{}] loss: {}, train_acc: {}'.format(batch_logs['tot_iter'], batch_logs[f'{self.metrics_prefix}/mean_loss'], batch_logs[f'{self.metrics_prefix}/mean_acc']))
            if self.to_neptune:
                neptune.send_metric('{} / Mean Running Loss '.format(self.log_text), batch_logs[f'{self.metrics_prefix}/mean_loss'])
                neptune.send_metric('{} / Mean Train Accuracy train'.format(self.log_text), batch_logs[f'{self.metrics_prefix}/mean_acc'])
            self.init_classic_logs()


class TotalAccuracyMetric(Metrics):
    def __init__(self,  use_cuda, to_neptune=True, log_text=''):
        super().__init__(use_cuda, log_every=None, to_neptune=to_neptune, log_text=log_text)

    def on_training_step_end(self, batch_index, batch_logs=None):
        super().on_training_step_end(batch_index, batch_logs)
        self.update_classic_logs(batch_logs)

    def on_train_end(self, logs=None):
        logs['total_accuracy'] = 100.0 * self.correct_train / self.total_samples
        print('Total Accuracy for [{}] samples, [{}]: {}%'.format(self.total_samples, self.log_text, logs['total_accuracy']))
        if self.to_neptune:
            neptune.log_metric('{} Acc'.format(self.log_text), logs['total_accuracy'])


class ComputeConfMatrix(Callback):
    def __init__(self, num_classes, reset_every=None, send_to_neptune=True, neptune_text=''):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        self.send_to_neptune = send_to_neptune
        self.neptune_text = neptune_text
        self.reset_every = reset_every
        self.num_iter = 0
        super().__init__()


    def on_training_step_end(self, batch, logs=None):
        if self.reset_every is not None and batch % self.reset_every == 0:
            self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
            self.num_iter = 0
        for t, p in zip(logs['y_true'].view(-1), logs['y_pred'].view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1
        self.num_iter += 1

    def on_train_end(self, logs=None):
        logs['conf_mat_acc'] = (self.confusion_matrix / self.confusion_matrix.sum(1)[:, None]).numpy()

        if self.send_to_neptune:
            figure = plt.figure(figsize=(10, 7))
            sn.heatmap(logs['conf_mat_acc'], annot=True, annot_kws={"size": 16})  # font size
            plt.ylabel('truth')
            plt.xlabel('predicted')
            plt.title(self.neptune_text + ' last {} iters'.format(self.num_iter))
            neptune.log_image('Confusion Matrix [{}]'.format(self.neptune_text), figure)
            plt.close()


class RollingAccEachClassNeptune(Callback):
    def __init__(self, log_every, num_classes, neptune_text=''):
        self.log_every = log_every
        self.confusion_matrix = torch.zeros(num_classes, num_classes)
        self.neptune_text = neptune_text
        self.num_classes = num_classes
        super().__init__()

    def on_training_step_end(self, batch, logs=None):
        for t, p in zip(logs['y_true'].view(-1), logs['y_pred'].view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1
        if batch % self.log_every:
            correct_class = self.confusion_matrix.diag() / self.confusion_matrix.sum(1)
            self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
            if correct_class is not None:
                for idx, cc in enumerate(correct_class.numpy()):
                    neptune.send_metric(f'Training {idx} Class Acc - [{self.neptune_text}]', cc * 100 if cc is not None else -1)


class PlotTimeElapsed(Callback):
    def __init__(self, time_every=100):
        super().__init__()
        self.time_every = time_every
        self.start_time = 0

    def on_train_begin(self, logs=None):
        self.start_time = time()

    def on_batch_end(self, batch, logs=None):
        if batch % self.time_every == 0:
            print('Time Elapsed {} iter: {}'.format(self.time_every, time() - self.start_time))
            self.start_time = time()


class ComputeDataframe(Callback):
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
                                          cat_to_save]))]
        return array

    def __init__(self, num_classes, use_cuda, translation_type_str, network_name, size_canvas, log_density_neptune=True, log_text_plot='', output_and_softmax=True):
        super().__init__()
        self.output_and_softmax = output_and_softmax
        self.size_canvas = size_canvas
        self.num_classes = num_classes
        self.translation_type_str = translation_type_str
        self.network_name = network_name

        self.index_dataframe = ['net', 'class_name', 'transl_X', 'transl_Y', 'tested_area', 'is_correct', 'class_output']
        self.column_names = self.build_columns(['class {}'.format(i) for i in range(self.num_classes)])
        self.rows_frames = []
        self.use_cuda = use_cuda
        self.plot_density_on_neptune = log_density_neptune
        self.log_text_plot = log_text_plot

    def on_training_step_end(self, batch, logs=None):
        output_batch_t = logs['output']
        labels_batch_t = logs['y_true']
        predicted_batch_t = logs['y_pred']
        face_center_batch_t = logs['more']['center']

        correct_batch_t = (utils.make_cuda(predicted_batch_t, self.use_cuda) == utils.make_cuda(labels_batch_t, self.use_cuda))
        labels = labels_batch_t.tolist()
        predicted_batch = predicted_batch_t.tolist()
        correct_batch = correct_batch_t.tolist()
        face_center_batch = np.array([np.array(i) for i in face_center_batch_t]).transpose()

        if self.output_and_softmax:
            softmax_batch_t = torch.softmax(utils.make_cuda(output_batch_t, self.use_cuda), 1)
            softmax_batch = np.array(softmax_batch_t.tolist())
            output_batch = np.array(output_batch_t.tolist())

        for c, _ in enumerate(correct_batch_t):
            correct = correct_batch[c]
            label = labels[c]
            predicted = predicted_batch[c]
            face_center = face_center_batch[c]

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

                self.rows_frames.append([self.network_name, label, face_center[0], face_center[1], self.translation_type_str, correct, predicted, max_softmax, softmax_correct_category, *softmax, max_output, output_correct_category, *output])
            else:
                self.rows_frames.append([self.network_name, label, face_center[0], face_center[1], self.translation_type_str, correct, predicted])

    def on_train_end(self, logs=None):
        data_frame = pd.DataFrame(self.rows_frames)
        data_frame = data_frame.set_index([i for i in range(len(self.index_dataframe))])
        data_frame.index.names = self.index_dataframe

        if self.output_and_softmax:
            data_frame.columns = self.column_names
        data_frame.reset_index(level='is_correct', inplace=True)

        if self.plot_density_on_neptune:
            mean_accuracy = data_frame.groupby(['transl_X', 'transl_Y']).mean()['is_correct']
            ax, fig, im = framework_utils.imshow_density(mean_accuracy, lim=[1 / self.num_classes - 1 / self.num_classes * 0.2, 1], plot_args={'interpolate': True, 'size_canvas': self.size_canvas})
            plt.title(self.log_text_plot)
            cbar = fig.colorbar(im)
            cbar.set_label('Mean Accuracy (%)', rotation=270, labelpad=25)
            neptune.log_image('{} Density Plot Accuracy'.format(self.log_text_plot), fig)
            plt.close()

        logs['dataframe'] = data_frame

class CompConfMatrixFewShot(ComputeConfMatrix):
    def on_training_step_end(self, batch, logs=None):
        if self.reset_every is not None and batch % self.reset_every == 0:
            self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
            self.num_iter = 0
        for t, p in zip(logs['y_true_real_lab'].view(-1), logs['y_pred_real_lab'].view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1
        self.num_iter += 1


class PlotGradientNeptune(Callback):
    grad = []
    layers = []

    def __init__(self, net, log_every=50, plot_every=500, log_txt=''):
        self.log_every = log_every
        self.plot_every = plot_every
        self.log_txt = log_txt
        self.net = net
        super().__init__()
        for n, p in self.net.named_parameters():
            if p.requires_grad and ("bias" not in n):
                self.layers.append(n)
        self.layers = [i.split('.weight')[0] for i in self.layers]

    def on_training_step_end(self, batch, logs=None):
        if batch % self.log_every == 0:
            ave_grads = []
            for n, p in self.net.named_parameters():
                if p.requires_grad and ("bias" not in n):
                    ave_grads.append(p.grad.abs().mean())
            self.grad.append(ave_grads)

        if batch % self.plot_every == 0:
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
            neptune.log_image('Gradient Plot [{}]'.format(self.log_txt), figure)
            self.grad = []
            plt.close()



