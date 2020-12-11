"""
The `fit` function in this file implements a slightly modified version
of the Keras `model.fit()` API.
"""
import torch
from typing import Callable, List, Union
from callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback
from framework_utils import make_cuda
import framework_utils
from torch._six import inf
from models.sequence_learner import SequenceMatchingNetSimple
import framework_utils
from torchviz import make_dot
import matplotlib.pyplot as plt
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


def matching_net_step(data, model, loss_fn, optimizer, use_cuda, train, n_shot, k_way, q_queries, distance='l2', fce=False):
    logs = {}
    x, y_real_labels, more = data
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    embeddings = model.encoder_fr2seq(make_cuda(x, use_cuda))
    y = torch.arange(0, k_way, 1 / q_queries).long()

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]
    selected_classes = y_real_labels[n_shot * k_way::q_queries]
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
    # logs['y_pred_real_lab'] = prediction_real_labels
    # logs['y_true_real_lab'] = queries_real_labels

    if train:
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return loss, queries_real_labels, prediction_real_labels, logs


def sequence_net_Ntrain_1cand(data, model: SequenceMatchingNetSimple, loss_fn, optimizer, use_cuda, train, dataset, concatenate=False):
    k = dataset.sampler.k
    nSc = dataset.sampler.nSc
    nSt = dataset.sampler.nSt
    nFc = dataset.sampler.nFc
    nFt = dataset.sampler.nFt
    y_matching_labels = dataset.sampler.labels
    camera_positions = dataset.sampler.camera_positions

    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
        k = 1

    logs = {}
    assert nSc == 1 and nFc == 1
    x, _, _ = data

    # framework_utils.imshow_batch(x, stats=dataset.stats, title_lab=y_matching_labels)

    y_matching_correct = torch.tensor([1 if i[0] == i[1] else 0 for i in y_matching_labels], dtype=torch.float)
    #
    # b = torch.randn((30, 3, 32, 32))
    # c = model.image_embedding_candidates(b)
    # c[0]
    #
    # # b = torch.randn((30, 3, 32, 32))
    # c2 = model.image_embedding_candidates(b[:10])
    # c2[0]
    #
    # b10 = b[:10]
    # c10 = model.image_embedding_candidates(b10)
    # c10[0]
    #
    # c10[0] == c[0]  # false

    ## all embeddings are computed together
    # x = make_cuda(x, use_cuda)
    # candidates = x[:k]
    # training = x[k:]
    # emb_candidates = model.image_embedding_candidates(candidates)
    # emb_training = model.image_embedding_trainings(training)

    # print(f"k: {k} \ntraining: {training}\nshapeT: {training.shape} \n shapeX {x.shape} \nshapeC {candidates.shape}")
    ### USE THIS, IT WORKS!!###
    x = make_cuda(x, use_cuda)
    emb_all = model.image_embedding_candidates(x)
    emb_candidates = emb_all[:k]
    emb_trainings = emb_all[k:]
############

    # plt.figure(2)
    # plt.plot(emb_candidates.detach().numpy().T)
    # plt.plot(emb_training.detach().numpy().T)
    # framework_utils.imshow_batch(candidates, stats=dataset.stats)
    # plt.figure(2)
    # framework_utils.imshow_batch(training, stats=dataset.stats)


    # emb_training = model.image_embedding_training(make_cuda(training, use_cuda))

    ## embed nSc sequences of nFc frames
    # index = 0
    # obj_emb_t = make_cuda(torch.zeros(k, model.encoder_fr2seq.hidden_size), use_cuda)
    # obj_emb_t = make_cuda(torch.zeros(k, model.encoder_seq2obj.hidden_size), use_cuda)

    emb_sequence_batch = emb_trainings.reshape(k, nSt * nFt, 64)
    fr_hidden = make_cuda(torch.randn(1, k, model.encoder_fr2seq.hidden_size), use_cuda)

    out, h = model.encoder_fr2seq(emb_sequence_batch, fr_hidden)

    # for kk in range(k):
    #     for f in range(nFt*nSt):
    #         fr_output, fr_hidden = model.encoder_fr2seq(emb_training[f].view(1, 1, -1),  fr_hidden)
    #     obj_emb_t[kk] = fr_hidden

    #
    # for kk in range(k):
    #     for seq in range(nSt):
    #         fr_hidden = make_cuda(torch.randn(1, 1, model.encoder_fr2seq.hidden_size), use_cuda)
    #         curr_seq = emb_training[index: index + nFt]
    #         index += nFt
    #         for f in range(nFt):
    #             fr_output, fr_hidden = model.encoder_fr2seq(curr_seq[f].view(1, 1, -1),  fr_hidden)
    #         seq_emb_t[kk, seq] = fr_hidden

    # for kk in range(k):
    #     seq_hidden = make_cuda(torch.randn(1, 1, model.encoder_fr2seq.hidden_size), use_cuda)
    #     for seq in range(nSt):
    #         seq_output, seq_hidden = model.encoder_seq2obj(seq_emb_t[kk, seq].view(1, 1, -1), seq_hidden)
    #     obj_emb_t[kk] = seq_hidden

    # c_set = emb_candidates.repeat((k, 1))
    # t_set = obj_emb_t.repeat_interleave(k, axis=0)
    # ccat = torch.cat((c_set, q_set), axis=1)
    # relation_scores = model.relation_net(torch.cat((emb_training, emb_candidates), axis=1))

    relation_scores = model.relation_net(torch.abs(emb_candidates-h.squeeze(0)))

    # relation_scores = model.relation_net(torch.abs(emb_candidates-obj_emb_t))

    rs = make_cuda(relation_scores.squeeze(0), use_cuda)
    lb = make_cuda(y_matching_correct, use_cuda)
    loss = loss_fn(rs, lb)
    # assert len(x) == n_shot * k_way + k_way * q_queries
    y_matching_predicted = torch.tensor(rs > 0.5, dtype=torch.int).T

    logs['output'] = relation_scores
    logs['labels'] = y_matching_labels
    logs['camera_positions'] = camera_positions


    # make_dot(loss, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    if train:
        loss.backward()
        # clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    # (y_matching_labels == y_matching_predicted).sum()
    return loss, y_matching_correct, y_matching_predicted, logs


def sequence_net_step(data, model: SequenceMatchingNetSimple, loss_fn, optimizer, use_cuda, train, k, nSt, nSc, nFt, nFc, concatenate=False):
    logs = {}
    x, y_real_labels, more = data
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    # framework_utils.imshow_batch(x)

    ## all embeddings are computed together
    all_emb = model.image_embedding(make_cuda(x, use_cuda))
    ## embed nSc sequences of nFc frames
    index = 0
    seq_emb_c = make_cuda(torch.zeros(k, nSc, model.encoder_fr2seq.hidden_size), use_cuda)
    seq_emb_q = make_cuda(torch.zeros(k, nSt, model.encoder_fr2seq.hidden_size), use_cuda)
    obj_emb_c = make_cuda(torch.zeros(k, model.encoder_seq2obj.hidden_size), use_cuda)
    obj_emb_q = make_cuda(torch.zeros(k, model.encoder_seq2obj.hidden_size), use_cuda)

    #
    tot_seq = 0
    for kk in range(k):
        for seq in range(nSc):
            fr_hidden = make_cuda(torch.randn(1, 1, model.encoder_fr2seq.hidden_size), use_cuda) #make_cuda(model.encoder_fr2seq.init_hidden(), use_cuda)
            # hidden1 = torch.randn(1, 1, model.encoder_fr2seq.hidden_size)
            # hidden2 = torch.randn(1, 1, model.encoder_fr2seq.hidden_size)

            curr_seq = all_emb[index: index + nFc]
            index += nFc
            for f in range(nFc):
                fr_output, fr_hidden = model.encoder_fr2seq(curr_seq[f].view(1, 1, -1), fr_hidden)
            seq_emb_c[kk, tot_seq] = fr_hidden

    #
    for kk in range(k):
        for seq in range(nSt):
            fr_hidden = make_cuda(torch.randn(1, 1, model.encoder_fr2seq.hidden_size), use_cuda)
            # fr_hidden = make_cuda(model.encoder_fr2seq.init_hidden(), use_cuda)
            # hidden1 = torch.randn(1, 1, model.encoder_fr2seq.hidden_size)
            # hidden2 = torch.randn(1, 1, model.encoder_fr2seq.hidden_size)

            curr_seq = all_emb[index: index + nFt]
            index += nFt
            for f in range(nFt):
                fr_output, fr_hidden = model.encoder_fr2seq(curr_seq[f].view(1, 1, -1),  fr_hidden)
            seq_emb_q[kk, seq] = fr_hidden

    #
    for kk in range(k):
        seq_hidden = make_cuda(torch.randn(1, 1, model.encoder_fr2seq.hidden_size), use_cuda)
        for seq in range(nSc):
            seq_output, seq_hidden = model.encoder_seq2obj(seq_emb_c[kk, seq].view(1, 1, -1), seq_hidden)
        obj_emb_c[kk] = seq_hidden

    for kk in range(k):
        seq_hidden = make_cuda(torch.randn(1, 1, model.encoder_fr2seq.hidden_size), use_cuda)
        for seq in range(nSt):
            seq_output, seq_hidden = model.encoder_seq2obj(seq_emb_q[kk, seq].view(1, 1, -1), seq_hidden)
        obj_emb_q[kk] = seq_hidden

    c_set = obj_emb_c
    q_set = obj_emb_q
    # c_set = seq_emb_c.squeeze(axis=1)
    # q_set = seq_emb_q.squeeze(axis=1)
    # c_set = all_emb[:k]
    # q_set = all_emb[k:]


    c_set = c_set.repeat((k, 1))
    q_set = q_set.repeat_interleave(k, axis=0)
    # ccat = torch.cat((c_set, q_set), axis=1)
    relation_scores = model.relation_net(torch.abs(c_set-q_set)).view((k, -1))

    # loss = loss_fn(make_cuda(relation_scores, use_cuda),
    #                make_cuda(torch.eye(k, dtype=torch.long), use_cuda))

    loss = loss_fn(make_cuda(relation_scores, use_cuda),
                   # make_cuda(torch.range(0, k-1, dtype=torch.long), use_cuda))
                   make_cuda(torch.eye(k), use_cuda))
    # assert len(x) == n_shot * k_way + k_way * q_queries

    _, predicted = relation_scores.max(dim=1)
    y_real_labels = y_real_labels[:k]
    prediction_real_labels = y_real_labels[predicted]

    logs['output'] = relation_scores
    logs['more'] = more

    # make_dot(loss, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    if train:
        loss.backward()
        # clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return loss, y_real_labels, prediction_real_labels, logs


def relation_net_step(data, model, loss_fn, optimizer, use_cuda, train, n_shot, k_way, q_queries, concatenate=False):
    logs = {}
    x, y_real_labels, more = data
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    # framework_utils.imshow_batch(x)
    assert len(x) == n_shot * k_way + k_way * q_queries
    y_onehot = torch.zeros(q_queries * k_way, k_way)
    y = torch.arange(0, k_way, 1 / q_queries)
    y_onehot = y_onehot.scatter(1, y.unsqueeze(-1).long(), 1)

    embeddings = model.encoder(make_cuda(x, use_cuda))
    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]

    batch = make_cuda(torch.tensor([]), use_cuda)

    if concatenate:
        # concatenate all features (used in relation_net_cat)
        summed_supp = support.view(k_way, n_shot, *support.size()[-3:]).reshape(k_way, support.shape[1] * n_shot, *support.shape[-2:])
    else:
        # sum across n_shots as in classic relation_net
        summed_supp = support.view(k_way, n_shot, *support.size()[-3:]).sum(dim=1)

    # this could be done without for loop, but for now it seems to be fast enough
    for idx, q in enumerate(queries):
        # K_WAY x 64 * (K_WAY + 1) x 56 x 56
        episode = torch.cat((summed_supp, q.expand(k_way, -1, -1, -1)), dim=1)
        batch = torch.cat((batch, make_cuda(episode, use_cuda)))

    output = model.relation_net(batch).view((k_way, -1)).T

    loss = loss_fn(make_cuda(output, use_cuda),
                   make_cuda(y_onehot, use_cuda))

    _, predicted = output.max(dim=1)

    selected_classes = y_real_labels[n_shot * k_way::q_queries]
    prediction_real_labels = selected_classes[predicted]
    queries_real_labels = y_real_labels[n_shot * k_way:]
    logs['output'] = output
    logs['more'] = more
    logs['more']['center'] = [i[n_shot * k_way:] for i in logs['more']['center']]
    # logs['y_pred_real_lab'] = prediction_real_labels
    # logs['y_true_real_lab'] = queries_real_labels

    if train:
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return loss, queries_real_labels, prediction_real_labels, logs


def run(data_loader, use_cuda, net, callbacks: List[Callback] = None, optimizer=None, loss_fn=None, iteration_step=standard_net_step, iteration_step_kwargs=None, epochs=20):
    torch.cuda.empty_cache()

    if iteration_step_kwargs is None:
        iteration_step_kwargs = {}

    make_cuda(net, use_cuda)

    callbacks = CallbackList(callbacks)
    callbacks.set_model(net)
    batch_logs = {}

    callbacks.on_train_begin()

    tot_iter = 0
    for epoch in range(epochs):
        epoch_logs = {}
        callbacks.on_epoch_begin(epoch)
        print('epoch: {}'.format(epoch))
        for batch_index, data in enumerate(data_loader, 0):

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
