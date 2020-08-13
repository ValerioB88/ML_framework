"""
The `fit` function in this file implements a slightly modified version
of the Keras `model.fit()` API.
"""
import torch
from typing import Callable, List, Union
from callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback
from framework_utils import make_cuda
from torch._six import inf
EPSILON = 1e-8


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


def standard_net_step(data, model, loss_fn, optimizer, use_cuda, train):
    logs = {}
    images, labels, more = data
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    optimizer.zero_grad()
    output_batch = model(make_cuda(images, use_cuda))
    loss = loss_fn(make_cuda(output_batch, use_cuda),
                   make_cuda(labels, use_cuda))
    logs['output'] = output_batch
    logs['more'] = more

    max_output, predicted = torch.max(output_batch, 1)
    if train:
        loss.backward()
        optimizer.step()
    return loss, labels, predicted, logs


def create_nshot_task_label(k: int, q: int) -> torch.Tensor:
    y = torch.arange(0, k, 1 / q).long()
    return y


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def matching_net_predictions(attention: torch.Tensor, n: int, k: int, q: int, use_cuda: bool) -> torch.Tensor:
    if attention.shape != (q * k, k * n):
        raise(ValueError(f'Expecting attention Tensor to have shape (q * k, k * n) = ({q * k, k * n})'))

    # Create one hot label vector for the support set
    y_onehot = torch.zeros(k * n, k)

    # Unsqueeze to force y to be of shape (K*n, 1) as this
    # is needed for .scatter()
    y = create_nshot_task_label(k, n).unsqueeze(-1)
    y_onehot = y_onehot.scatter(1, y, 1)

    y_pred = torch.mm(attention, y_onehot.cuda() if use_cuda else y_onehot)

    return y_pred

def matching_net_step_plus(data, model, loss_fn, optimizer, use_cuda, train, n_shot, k_way, q_queries):
    logs = {}
    x, y_real_labels, more = data
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    embeddings = model.encoder(make_cuda(x, use_cuda))
    y = torch.arange(0, k_way, 1 / q_queries).long()

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]
    support_and_test = make_cuda(torch.tensor([]), use_cuda)
    for q in queries:
        support_and_test = torch.cat((support_and_test, torch.cat((support, make_cuda(torch.unsqueeze(q, 0), use_cuda))).flatten(start_dim=0, end_dim=1).unsqueeze(0)), dim=0)

    # support_and_test = support_and_test.permute(dims=[1, 0, 2, 3])
    output = model.evaluator(support_and_test)

    _, predicted = torch.max(output, 1)

    loss = loss_fn(make_cuda(output, use_cuda),
                   make_cuda(y, use_cuda))

    selected_classes = y_real_labels[n_shot * k_way::n_shot]
    prediction_real_labels = selected_classes[predicted]
    queries_real_labels = y_real_labels[n_shot * k_way:]
    logs['output'] = output
    logs['more'] = more
    logs['more']['center'] = [i[n_shot * k_way:] for i in logs['more']['center']]
    logs['y_pred_real_lab'] = prediction_real_labels
    logs['y_true_real_lab'] = queries_real_labels

    if train:
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return loss, y, predicted, logs

def matching_net_step(data, model, loss_fn, optimizer, use_cuda, train, n_shot, k_way, q_queries, distance='l2', fce=False):
    logs = {}
    x, y_real_labels, more = data
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    embeddings = model.encoder(make_cuda(x, use_cuda))
    y = torch.arange(0, k_way, 1 / q_queries).long()

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]
    selected_classes = y_real_labels[n_shot * k_way::n_shot]
    queries_real_labels = y_real_labels[n_shot * k_way:]

    if fce:
        support, _, _ = model.g(support.unsqueeze(1))
        support = support.squeeze(1)

        queries = model.f(support, queries)

    distances = pairwise_distances(queries, support, distance)
    # we generally only care about the test output/more
    logs['output'] = -distances
    logs['more'] = more
    logs['more']['center'] = [i[n_shot * k_way:] for i in logs['more']['center']]

    attention = (-distances).softmax(dim=1)

    y_pred = matching_net_predictions(attention, n_shot, k_way, q_queries, use_cuda)
    clipped_y_pred = y_pred.clamp(EPSILON, 1 - EPSILON)
    _, predicted = torch.max(y_pred, 1)
    prediction_real_labels = selected_classes[predicted]
    # CrossEntropy is log softmax + NLLLoss. Here we do them separately.
    loss = loss_fn(make_cuda(clipped_y_pred.log(), use_cuda), make_cuda(y, use_cuda))
    logs['y_pred_real_lab'] = prediction_real_labels
    logs['y_true_real_lab'] = queries_real_labels

    if train:
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return loss, y, predicted, logs


def run(train_loader, use_cuda, net, callbacks: List[Callback] = None, optimizer=None, loss_fn=None, iteration_step=standard_net_step, iteration_step_kwargs=None, epochs=20):
    torch.cuda.empty_cache()

    if iteration_step_kwargs is None:
        iteration_step_kwargs = {}

    make_cuda(net, use_cuda)

    callbacks = CallbackList(callbacks)
    batch_logs = {}

    callbacks.on_train_begin()

    tot_iter = 0
    for epoch in range(epochs):
        epoch_logs = {}
        callbacks.on_epoch_begin(epoch)
        print('epoch: {}'.format(epoch))
        for batch_index, data in enumerate(train_loader, 0):
            tot_iter += 1
            callbacks.on_batch_begin(batch_index)

            loss, y_true, y_pred, logs = iteration_step(data, net, loss_fn, optimizer, use_cuda, **iteration_step_kwargs)

            batch_logs.update({'y_pred': y_pred, 'loss': loss, 'y_true': y_true, 'tot_iter': tot_iter, 'stop': False, **logs})

            callbacks.on_training_step_end(batch_index, batch_logs)
            callbacks.on_batch_end(batch_index, batch_logs)

            if batch_logs['stop']:
                break

        callbacks.on_epoch_end(epoch, epoch_logs)
        if batch_logs['stop']:
            break

    callbacks.on_train_end(batch_logs)
    return net, batch_logs
