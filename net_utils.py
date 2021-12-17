import torch.backends.cudnn as cudnn

from ML_framework.callbacks import *
import torch
import torchvision
import torch.nn as nn
from sty import fg, ef, rs, bg
import os

class GrabNet():
    @classmethod
    def get_net(cls, network_name, imagenet_pt=False, num_classes=None, **kwargs):
        """
        @num_classes = None indicates that the last layer WILL NOT be changed.
        """
        if imagenet_pt:
            print(fg.red + "Loading ImageNet" + rs.fg)

        nc = 1000 if imagenet_pt else num_classes
        if network_name == 'vgg11':
            net = torchvision.models.vgg11(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg11bn':
            net = torchvision.models.vgg11_bn(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16':
            net = torchvision.models.vgg16(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16bn':
            net = torchvision.models.vgg16_bn(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg19bn':
            net = torchvision.models.vgg19_bn(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'resnet18':
            net = torchvision.models.resnet18(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet50':
            net = torchvision.models.resnet50(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'alexnet':
            net = torchvision.models.alexnet(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'inception_v3':  # nope
            net = torchvision.models.inception_v3(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'densenet121':
            net = torchvision.models.densenet121(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'densenet201':
            net = torchvision.models.densenet201(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'googlenet':
            net = torchvision.models.googlenet(pretrained=imagenet_pt, progress=True, num_classes=nc)
            if num_classes is not None:
                net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = cls.get_other_nets(network_name, num_classes, imagenet_pt, **kwargs)
            assert False if net is False else True, f"Network name {network_name} not recognized"

        return net

    @staticmethod
    def get_other_nets(network_name, num_classes, imagenet_pt, **kwargs):
        pass


def prepare_network(net, config, train=True):
    pretraining_file = 'vanilla' if config.pretraining == 'ImageNet' else config.pretraining
    net = load_pretraining(net, pretraining_file, config.use_cuda)
    net.cuda() if config.use_cuda else None
    cudnn.benchmark = True
    net.train() if train else net.eval()
    framework_utils.print_net_info(net) if config.verbose else None



def load_pretraining(net, pretraining, use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    if pretraining != 'vanilla':
        if os.path.isfile(pretraining):
            print(fg.red + f"Loading.. full model from {pretraining}..." + rs.fg, end="")
            ww = torch.load(pretraining, map_location='cuda' if use_cuda else 'cpu')
            if 'full' in ww:
                ww = ww['full']
            net.load_state_dict(ww)
            print(fg.red + " Done." + rs.fg)
        else:
            assert False, f"Pretraining path not found {pretraining}"

    return net


def get_test_callbacks(config, log_text, testing_loader):
    all_cb = [
        StandardMetrics(log_every=5, print_it=True,
                        use_cuda=config.use_cuda,
                        weblogger=0, log_text=log_text,
                        metrics_prefix='cnsl',
                        size_dataset=len(testing_loader)),
        PlotIterationsInfo(time_every=3000),
        ComputeConfMatrix(num_classes=testing_loader.dataset.num_classes,
                          weblogger=config.weblogger,
                          weblog_text=log_text,
                          class_names=testing_loader.dataset.classes),
        StopFromUserInput(),
        StopWhenMetricIs(value_to_reach=0, metric_name='epoch', check_after_batch=False),  # you could use early stopping for that

        TotalAccuracyMetric(use_cuda=config.use_cuda,
                            weblogger=config.weblogger, log_text=log_text)]

    if config.max_iterations_testing != np.inf:
        all_cb += ([StopWhenMetricIs(value_to_reach=config.max_iterations_testing, metric_name='tot_iter')])
    return all_cb


def get_train_callbacks(net, config, log_text, train_loader, test_loaders=None, call_run=None):
    def stop(logs, cb):
        logs['stop'] = True
        print('Early Stopping')

    all_cb = [
        EndEpochStats(),
        # StandardMetrics(log_every=5, print_it=True,
        #                 use_cuda=config.use_cuda,
        #                 weblogger=config.weblogger, log_text=log_text,
        #                 metrics_prefix='cnsl',
        #                 size_dataset=len(train_loader)),

        StopFromUserInput(),
        PlotIterationsInfo(time_every=100),
        # TotalAccuracyMetric(use_cuda=config.use_cuda,
        #                     weblogger=None, log_text=log_text),
        SaveModel(net=net, output_path=config.model_output_filename, min_iter=500)]

    if config.stop_when_train_acc_is != np.inf:
        all_cb += (
            [TriggerActionWithPatience(mode='max', min_delta=0.01,
                                       patience=800, min_delta_is_percentage=True,
                                       reaching_goal=config.stop_when_train_acc_is,
                                       metric_name='webl/mean_acc' if config.weblogger else 'cnsl/mean_acc',
                                       check_every=config.weblog_check_every if config.weblogger else config.console_check_every,
                                       triggered_action=stop,
                                       action_name='Early Stopping', alpha=0.1,
                                       weblogger=config.weblogger)])  # once reached a certain accuracy
    if config.max_epochs != np.inf:
        all_cb += ([StopWhenMetricIs(value_to_reach=config.max_epochs - 1, metric_name='epoch', check_after_batch=False)])  # you could use early stopping for that

    if config.max_iterations != np.inf:
        all_cb += ([StopWhenMetricIs(value_to_reach=config.max_iterations - 1, metric_name='tot_iter')])  # you could use early stopping for that
    if config.patience_stagnation != np.inf:
        all_cb += ([TriggerActionWithPatience(mode='min',
                                              min_delta=0.01,
                                              patience=config.patience_stagnation,
                                              min_delta_is_percentage=False, reaching_goal=None,
                                              metric_name='webl/mean_loss' if config.weblogger else 'cnsl/mean_loss',
                                              check_every=config.weblog_check_every if config.weblogger else config.console_check_every,
                                              triggered_action=stop, action_name='Early Stopping', alpha=0.05,
                                              weblogger=config.weblogger)])  # for stagnation
    # Extra Stopping Rule. When loss is 0, just stop.
    all_cb += ([TriggerActionWithPatience(mode='min',
                                          min_delta=0,
                                          patience=200,
                                          min_delta_is_percentage=True, reaching_goal=0,
                                          metric_name='webl/mean_loss' if config.weblogger else 'cnsl/mean_loss',
                                          check_every=config.weblog_check_every if config.weblogger else config.console_check_every,
                                          triggered_action=stop, action_name='Stop with Loss=0',
                                          weblogger=config.weblogger)])

    if test_loaders is not None:
        all_cb += ([DuringTrainingTest(testing_loaders=test_loaders, every_x_epochs=None, every_x_iter=None, every_x_sec=None, multiple_sec_of_test_time=None, auto_increase=True, weblogger=config.weblogger, log_text='test during train', use_cuda=config.use_cuda, call_run=call_run)])

    if config.weblogger:
        all_cb += [StandardMetrics(log_every=config.weblog_check_every, print_it=False,
                                   use_cuda=config.use_cuda,
                                   weblogger=config.weblogger, log_text=log_text,
                                   metrics_prefix='webl'),
                   PrintNeptune(config.weblogger, plot_every=config.weblog_check_every)]
    #                PlotGradientWeblog(net=self.net, log_every=50, plot_every=500, log_txt=log_text, weblogger=self.weblogger)]
    return all_cb
