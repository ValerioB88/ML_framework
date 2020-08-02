"""
The `fit` function in this file implements a slightly modified version
of the Keras `model.fit()` API.
"""
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union
from time import time
from callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback
# from metrics import NAMED_METRICS
# from logger import LOGGER, Log
import neptune

def gradient_step(model: Module, optimiser: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    """Takes a single gradient step.

    # Arguments
        model: Model to be fitted
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


# def batch_metrics(model: Module, y_pred: torch.Tensor, y: torch.Tensor, metrics: List[Union[str, Callable]],
#                   batch_logs: dict):
#     """Calculates metrics for the current training batch
#
#     # Arguments
#         model: Model being fit
#         y_pred: predictions for a particular batch
#         y: labels for a particular batch
#         batch_logs: Dictionary of logs for the current batch
#     """
#     model.eval()
#     for m in metrics:
#         if isinstance(m, str):
#             batch_logs[m] = NAMED_METRICS[m](y, y_pred)
#         else:
#             # Assume metric is a callable function
#             batch_logs = m(y, y_pred)
#
#     return batch_logs


def fit(model: Module, optimiser: Optimizer, loss_fn: Callable, epochs: int, dataloader: DataLoader,
        prepare_batch: Callable, metrics: List[Union[str, Callable]] = None, callbacks: List[Callback] = None,
        verbose: bool =True, fit_function: Callable = gradient_step, fit_function_kwargs: dict = {}):
    """Function to abstract away training loop.

    The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
    common training functionality provided they are written as a subclass of voicemap.Callback (following the
    Keras API).

    # Arguments
        model: Model to be fitted.
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        epochs: Number of epochs of fitting to be performed
        dataloader: `torch.DataLoader` instance to fit the model to
        prepare_batch: Callable to perform any desired preprocessing
        metrics: Optional list of metrics to evaluate the model with
        callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
            checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
        verbose: All print output is muted if this argument is `False`
        fit_function: Function for calculating gradients. Leave as default for simple supervised training on labelled
            batches. For more complex training procedures (meta-learning etc...) you will need to write your own
            fit_function
        fit_function_kwargs: Keyword arguments to pass to `fit_function`
    """
    # Determine number of samples:
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimiser
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index, batch_logs)

            x, y = prepare_batch(batch)

            loss, y_pred = fit_function(model, optimiser, loss_fn, x, y, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            # Loops through all metrics
            batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()



def standard_net_step(images, labels, model, loss_fn, optimizer, use_cuda, train):
    logs = {}
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    optimizer.zero_grad()
    output_batch = model(images.cuda() if use_cuda else images)
    loss = loss_fn(output_batch.cuda() if use_cuda else output_batch,
                   labels.cuda() if use_cuda else labels)
    max_output, predicted = torch.max(output_batch, 1)
    if train:
        loss.backward()
        optimizer.step()
    return loss, labels, predicted, logs

from external.few_shot.few_shot.matching import matching_net_predictions, pairwise_distances
def matching_net_step(x, y, model, loss_fn, optimizer, use_cuda, train, n_shot, k_way, q_queries, distance='l2', fce=False):
    logs = {}
    EPSILON = 1e-8
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    embeddings = model.encoder(x.cuda() if use_cuda else x)
    y = torch.arange(0, k_way, 1 / q_queries).long()

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]

    if fce:
        support, _, _ = model.g(support.unsqueeze(1))
        support = support.squeeze(1)

        queries = model.f(support, queries)

    distances = pairwise_distances(queries, support, distance)

    attention = (-distances).softmax(dim=1)

    y_pred = matching_net_predictions(attention, n_shot, k_way, q_queries, use_cuda)
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    _, predicted = torch.max(y_pred, 1)

    # CrossEntropy is log softmax + NLLLoss. Here we do them separately.
    loss = loss_fn(clipped_y_pred.log().cuda() if use_cuda else clipped_y_pred.log(), y.cuda() if use_cuda else y)

    if train:
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    # vis.imshow_batch(x)

    return loss, y, predicted, logs


def train_net(train_loader, use_cuda, net, params_to_update, max_iterations, callbacks: List[Callback] = None, verbose=True, optimizer=None, loss_fn=None, training_step=None, training_step_kwargs={}):
    torch.cuda.empty_cache()

    if params_to_update is None:
        params_to_update = net.parameters()

    if training_step is None:
        training_step = standard_net_step

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss().cuda() if use_cuda else torch.nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(params_to_update, lr=0.0001)

    if use_cuda:
        net.cuda()

    callbacks = CallbackList(callbacks)

    # callbacks = CallbackList([DefaultCallback(), ] + (callbacks or []) + [ProgressBarLogger(), ])
    # callbacks.set_model(net)
    callbacks.set_params({
        'num_batches': max_iterations,
        # 'batch_size': batch_size,
        'verbose': verbose,
        # 'metrics': (metrics or []),
        # 'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimiser': optimizer
    })

    callbacks.on_train_begin()

    start = time()
    # confusion_matrix = torch.zeros(num_classes, num_classes)

    tot_iter = 0
    for epoch in range(20):
        epoch_logs = {}
        callbacks.on_epoch_begin(epoch)
        print('epoch: {}'.format(epoch))
        for batch_index, data in enumerate(train_loader, 0):
            tot_iter += 1
            callbacks.on_batch_begin(batch_index)

            if batch_index % 100 == 100 - 1:
                print('Time Elapsed 100 iter: {}'.format(time() - start))
                start = time()

            # get the inputs; data is a list of [inputs, labels]
            x, y, more = data
            loss, y_true, y_pred, logs = training_step(x, y, net, loss_fn, optimizer, use_cuda, **training_step_kwargs)

            # batch_logs = batch_metrics(net, predicted, labels, metrics, batch_logs)
            batch_logs = {'y_pred': y_pred, 'loss': loss, 'y_true': y_true, 'tot_iter': tot_iter, 'stop': False,
                          **logs}
            # LOGGER.next_index(batch_logs)
            callbacks.on_training_step_end(batch_index, batch_logs)
            callbacks.on_batch_end(batch_index, batch_logs)

            if batch_logs['stop']:
                break

            # callbacks.check_stop(batch_index, batch_logs)

            # correct_train += ((predicted.cuda() if use_cuda else predicted) ==
            #                   (labels.cuda() if use_cuda else labels)).sum().item()
            # total_samples += labels.size(0)
            #
            # for t, p in zip(labels.view(-1), predicted.view(-1)):
            #     confusion_matrix[t.long(), p.long()] += 1
            # print statistics and compute validation
            # running_loss += loss.item()

            # if use_cuda and batch_index % compute_mean_acc_every == 0:
            #     mean_loss = running_loss / compute_mean_acc_every
            #     mean_acc_train = 100 * correct_train / total_samples
            #
            #     if verbose and batch_index % 5 == 0:
            #         print('[iter{}] loss: {}, train_acc: {}'.format(tot_num_iterations, mean_loss, mean_acc_train))
            #
            #     metric = mean_acc_train
            # if use_early_stopping and stop_when_train_acc_is < 100:
            #     if es.step(metric):
            #         stop_running = True
                # correct_train = 0
                # total_samples = 0
                # running_loss = 0
            # if use_cuda and (batch_index % send_metric_to_neptune_every) == send_metric_to_neptune_every - 1:
            #     if mean_loss is not None:
            #         neptune.send_metric('{} / Mean Running Loss '.format(log_text), mean_loss)
            #         neptune.send_metric('{} / Mean Train Accuracy train'.format(log_text), mean_acc_train)
            #     if correct_class is not None:
            #         for idx, cc in enumerate(correct_class.numpy()):
            #             neptune.send_metric('Train {} Class Acc'.format(idx), cc * 100)

            # if use_cuda and (batch_index % (compute_mean_acc_every * num_classes * 5) == 0 or stop_running):
            #     correct_class = confusion_matrix.diag() / confusion_matrix.sum(1)
            #     confusion_matrix = torch.zeros(num_classes, num_classes)
            #
            # if batch_index % call_callback_every == call_callback_every - 1 and callback is not None:
            #     args = {'net': net,
            #             'mean_acc_train': mean_acc_train,
            #             'iteration': batch_index}
            #     callback(args)

            # if tot_num_iterations >= max_iterations - 1:
            #     if verbose:
            #         print('Reached max iterations of {}'.format(max_iterations))
            #     stop_running = True
            #
            # tot_num_iterations += 1
            #
            # if stop_running:
            #     if verbose:
            #         print('Reached early stopping')
            #     break
        callbacks.on_epoch_end(epoch, epoch_logs)
        # if stop_running:
        #     if verbose:
        #         print('Reached early stopping')
        #     break
        if batch_logs['stop']:
            break
    callbacks.on_train_end(batch_logs)
    print('Finished Training')

    return net