import torch.backends.cudnn as cudnn
import cloudpickle
import argparse
from abc import ABC, abstractmethod

import neptune
import torch.cuda

from callbacks import *
from models.sequence_learner import *
from train_net import *
import random
from torch.optim.lr_scheduler import MultiStepLR

CONFIG = None

class Config(ABC):
    def __init__(self, experiment_class_name='default_name', parser=None, exp_tags=None, **kwargs):
        self.use_cuda = False
        self.loss_fn = None
        self.optimizer = None
        original_kwargs = kwargs.copy()
        if torch.cuda.is_available():
            print('Using cuda - you are probably on the server')
            self.use_cuda = True

        if parser is None:
            parser = argparse.ArgumentParser(allow_abbrev=False)
        parser = self.parse_arguments(parser)
        if len(kwargs) == 0:
            print("Running Experiment in " + ef.inverse + "MAIN" + rs.inverse + " mode. Any keyword params will be ignored")
            PARAMS = vars(parser.parse_known_args()[0])
        else:
            print("Running experiment in " + ef.inverse + "SCRIPT" + rs.inverse + " mode. Command line arguments will have the precedence over keywords arguments.")
            PARAMS = vars(parser.parse_known_args()[0])
            # only update the parser args that are NOT the defaults (so the one actually passed by user
            kwargs.update({k: v for k, v in PARAMS.items() if parser.get_default(k)!=v })
            # Update with default arguments for all the arguments not passed by kwargs
            kwargs.update({k: v for k, v in PARAMS.items() if k not in kwargs})
            PARAMS = kwargs
        # self.net = None
        [self.__setattr__(k, v) for k, v in kwargs.items()]

        self.current_run = -1
        # self.seed = PARAMS['seed']
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # self.num_runs = PARAMS['num_runs']  # this is not used here, and it shouldn't be here, but for now we are creating a weblogger session when an Experiment is created, and we want to save this as a parameter, so we need to do it here.
        inf_if_minus_one = lambda p: np.inf if p == -1 else p

        self.stop_when_train_acc_is = inf_if_minus_one(self.stop_when_train_acc_is)
        self.max_epochs = inf_if_minus_one(self.max_epochs)
        self.max_iterations = inf_if_minus_one(self.max_iterations)
        self.max_iterations_testing = inf_if_minus_one(self.max_iterations_testing)
        self.patience_stagnation = inf_if_minus_one(self.patience_stagnation)

        if self.use_cuda:
            torch.cuda.set_device(self.use_device_num)
            self.device_name = torch.cuda.get_device_name(self.use_device_num)
        else:
            self.device_name = 'cpu'
        print(fg.yellow + f"Device Name: " + ef.inverse + f"[{self.device_name}]" + rs.inverse + ((f' - Selected device num: ' + ef.bold + f'{self.use_device_num}' + rs.bold_dim) if self.use_cuda else '') + rs.fg);

        self.test_results_seed = {}
        self.experiment_loaders = {}  # we separate data from loaders because loaders are pickled objects and may broke when module name changes. If this happens, at least we preserve the data. We generally don't even need the loaders much.

        self.weblog_check_every = 5
        self.console_check_every = 100
        if self.force_cuda:
            self.use_cuda = True
        elif self.force_cuda == 0:
            print("Cuda Forcesully Disabled")
            self.use_cuda = False
        list_tags = []
        list_tags.extend(exp_tags) if exp_tags is not None else None

        if self.additional_tags is not None:
            [list_tags.append(i) for i in self.additional_tags.split('_') if i != 'emptytag' and i != '']
        list_tags.append('ptvanilla') if self.pretraining == 'vanilla' else None
        list_tags.append('ptImageNet') if self.pretraining == 'ImageNet' else None
        # list_tags.append(self.network_name)
        list_tags.append('lr{}'.format("{:2f}".format(self.learning_rate).split(".")[1])) if self.learning_rate is not None else None
        # list_tags.append('gray') if self.grayscale else None
        if self.max_iterations is None:
            self.max_iterations = 5000 if self.use_cuda else 10

        if not self.verbose:
            print(fg.magenta)

            print('***PARAMS***')
            if not self.use_cuda:
                list_tags.append('LOCALTEST')

            for i in sorted(original_kwargs.keys()):
                print(f'\t{i} : ' + ef.inverse + f'{original_kwargs[i]}' + rs.inverse)
            print(rs.fg)
        self.finalize_init(PARAMS, list_tags)
    #
    # def set_loss_fn(self, loss_fn):
    #     self.loss_fn = loss_fn
    #
    # def set_optimizer(self, optimizer):
    #     self.optimizer = optimizer

    def __str__(self):
        strr = ''
        for i in sorted(self.__dict__.keys()):
            strr += f'\t{i} : ' + ef.inverse + f'{self.__dict__[i]}' + rs.inverse + '\n'
        return strr

    def __setattr__(self, *args, **kwargs):
        if hasattr(self, 'weblogger'):
            if isinstance(self.weblogger, neptune.run.Run):
                self.weblogger[f"parameters/{args[0]}"] = str(args[1])
        super().__setattr__(*args, **kwargs)

    def parse_arguments(self, parser):
        parser.add_argument('-seed', "--seed", type=int, default=1)
        parser.add_argument("-r", "--num_runs",
                            help="run experiment n times",
                            type=int,
                            default=1)
        parser.add_argument("-fcuda", "--force_cuda",
                            help="Force to run it with cuda enabled or disabled (1/0). Set None to check if the GPU is available",
                            type=int,
                            default=None)
        parser.add_argument("-verbose", "--verbose",
                            type=lambda x: bool(int(x)),
                            default=True)
        parser.add_argument("-use_device_num", "--use_device_num",
                            help="Use selected device (int)",
                            type=int,
                            default=0)
        parser.add_argument("-weblogger", "--weblogger",
                            help="Log stuff to the weblogger [0=none, 1=wandb [not supported], 2=neptune])",
                            type=int,
                            default=2)
        parser.add_argument("-tags", "--additional_tags",
                            help="Add additional tags. Separate them by underscore. E.g. tag1_tag2",
                            type=str,
                            default=None)
        parser.add_argument("-prjnm", "--project_name",
                            type=str,
                            default='TestProject')
        parser.add_argument("-pat1", "--patience_stagnation",
                            help="Patience for early stopping for stagnation (num iter). Set to -1 to disable.",
                            type=int,
                            default=-1)
        parser.add_argument("-mo", "--model_output_filename",
                            help="file name of the trained model",
                            type=str,
                            default=None)
        parser.add_argument("-o", "--output_filename",
                            help="output file name for the pandas dataframe files",
                            type=str,
                            default=None)
        parser.add_argument("-mt", "--max_iterations_testing",
                            help="num iterations for testing. If -1, only 1 epoch will be computed",
                            default=-1,
                            type=int)
        parser.add_argument("-lr", "--learning_rate",
                            default=None, help='learning rate. If none the standard one will be chosen',
                            type=float)
        parser.add_argument("-sa", "--stop_when_train_acc_is",
                            default=-1,
                            help='Stop when train accuracy is a value for 800 iterations. -1 to disable',
                            type=int)
        parser.add_argument("-pt", "--pretraining",
                            help="use [vanilla], [ImageNet (only for standard exp)] or a path",
                            type=str,
                            default='vanilla')
        parser.add_argument("-mi", "--max_iterations",
                            help="max number of batch iterations",
                            type=int,
                            default=-1)
        parser.add_argument("-n", "--network_name", help="[vgg11] [vgg11_bn] [vgg16] [vgg16_bn]",
                            default=None,
                            type=str)
        parser.add_argument("-epochs", "--max_epochs",
                            help="max epoch, select -1 for infinite epochs",
                            type=int,
                            default=-1)
        return parser

    def new_run(self):
        self.current_run += 1
        self.test_results_seed[self.current_run] = {}
        self.experiment_loaders[self.current_run] = {'training': [],
                                                     'testing': []}
        print('Run Number {}'.format(self.current_run))

    def finalize_init(self, PARAMS, list_tags):
        print(fg.magenta)
        print('**LIST_TAGS**:')
        print(list_tags)
        if self.verbose:
            print('***PARAMS***')
            if not self.use_cuda:
                list_tags.append('LOCALTEST')

            for i in sorted(PARAMS.keys()):
                print(f'\t{i} : ' + ef.inverse + f'{PARAMS[i]}' + rs.inverse)

        if self.weblogger == 2:
            neptune_run = neptune.init(f'valeriobiscione/{self.project_name}')
            neptune_run["sys/tags"].add(list_tags)
            neptune_run["parameters"] = PARAMS
            self.weblogger = neptune_run
        print(rs.fg)
        self.new_run()

    # @abstractmethod
    # def call_run(self, loader, callbacks=None, train=True):
    #     pass
    #
    # @abstractmethod
    # def get_net(self, num_classes):
    #     return NotImplementedError
    #
    # def prepare_train_callbacks(self, log_text, train_loader, test_loaders=None):
    #     def stop(logs, cb):
    #         logs['stop'] = True
    #         print('Early Stopping')
    #
    #
    #     all_cb = [
    #         EndEpochStats(),
    #         StandardMetrics(log_every=1, print_it=True,
    #                         use_cuda=self.use_cuda,
    #                         weblogger=self.weblogger, log_text=log_text,
    #                         metrics_prefix='cnsl',
    #                         size_dataset=len(train_loader)),
    #
    #         StopFromUserInput(),
    #         PlotTimeElapsed(time_every=100),
    #         TotalAccuracyMetric(use_cuda=self.use_cuda,
    #                             weblogger=None, log_text=log_text),
    #         SaveModel(net=self.net, output_path=self.model_output_filename, min_iter=500)]
    #
    #     if self.stop_when_train_acc_is != np.inf:
    #         all_cb += (
    #             [TriggerActionWithPatience(mode='max', min_delta=0.01,
    #                                        patience=800, min_delta_is_percentage=True,
    #                                        reaching_goal=self.stop_when_train_acc_is,
    #                                        metric_name='webl/mean_acc' if self.weblogger else 'cnsl/mean_acc',
    #                                        check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
    #                                        triggered_action=stop,
    #                                        action_name='Early Stopping', alpha=0.1,
    #                                        weblogger=self.weblogger)])  # once reached a certain accuracy
    #     if self.max_epochs != np.inf:
    #         all_cb += ([StopWhenMetricIs(value_to_reach=self.max_epochs - 1, metric_name='epoch', check_after_batch=False)])  # you could use early stopping for that
    #
    #     if self.max_iterations != np.inf:
    #         all_cb += ([StopWhenMetricIs(value_to_reach=self.max_iterations - 1, metric_name='tot_iter')])  # you could use early stopping for that
    #     if self.patience_stagnation != np.inf:
    #         all_cb += ([TriggerActionWithPatience(mode='min',
    #                                               min_delta=0.01,
    #                                               patience=self.patience_stagnation,
    #                                               min_delta_is_percentage=False, reaching_goal=None,
    #                                               metric_name='webl/mean_loss' if self.weblogger else 'cnsl/mean_loss',
    #                                               check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
    #                                               triggered_action=stop, action_name='Early Stopping', alpha=0.05,
    #                                               weblogger=self.weblogger)])  # for stagnation
    #     # Extra Stopping Rule. When loss is 0, just stop.
    #     all_cb += ([TriggerActionWithPatience(mode='min',
    #                                           min_delta=0,
    #                                           patience=200,
    #                                           min_delta_is_percentage=True, reaching_goal=0,
    #                                           metric_name='webl/mean_loss' if self.weblogger else 'cnsl/mean_loss',
    #                                           check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
    #                                           triggered_action=stop, action_name='Stop with Loss=0',
    #                                           weblogger=self.weblogger)])
    #
    #     if test_loaders is not None:
    #         all_cb += ([DuringTrainingTest(testing_loaders=test_loaders, every_x_epochs=None, every_x_iter=None, every_x_sec=None, multiple_sec_of_test_time=None, auto_increase=True, weblogger=self.weblogger, log_text='test during train', use_cuda=self.use_cuda, call_run=self.call_run)])
    #
    #     if self.weblogger:
    #         all_cb += [StandardMetrics(log_every=self.weblog_check_every, print_it=False,
    #                                    use_cuda=self.use_cuda,
    #                                    weblogger=self.weblogger, log_text=log_text,
    #                                    metrics_prefix='webl'),
    #                    PrintLogsNeptune(self.weblogger, plot_every=self.weblog_check_every)]
    #     #                PlotGradientWeblog(net=self.net, log_every=50, plot_every=500, log_txt=log_text, weblogger=self.weblogger)]
    #     return all_cb
    #
    #
    # def _get_num_classes(self, loader):
    #     return len(loader.dataset.classes)
    #
    # def train(self, train_loader, callbacks=None, test_loaders=None, log_text='train'):
    #     print(f"**Training** [{log_text}]")
    #     self.net = self.get_net(num_classes=self._get_num_classes(train_loader))
    #
    #     if not self.optimizer or not self.loss_fn:
    #         assert False, "Optimizer or Loss Function not set"
    #     self.experiment_loaders[self.current_run]['training'].append(train_loader)
    #     all_cb = self.prepare_train_callbacks(log_text, train_loader, test_loaders=test_loaders)
    #
    #     all_cb += (callbacks or [])
    #
    #     if self.use_cuda:
    #         self.net.cuda()
    #         cudnn.benchmark = True
    #     self.net.train()
    #     net, logs = self.call_run(train_loader,
    #                               train=True,
    #                               callbacks=all_cb,
    #                               )
    #     return net
    #
    # def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe):
    #     all_cb = [
    #         StandardMetrics(log_every=3000, print_it=True,
    #                         use_cuda=self.use_cuda,
    #                         weblogger=0, log_text=log_text,
    #                         metrics_prefix='cnsl',
    #                         size_dataset=len(testing_loader)),
    #         PlotTimeElapsed(time_every=3000),
    #
    #         StopFromUserInput(),
    #         StopWhenMetricIs(value_to_reach=0, metric_name='epoch', check_after_batch=False),  # you could use early stopping for that
    #
    #         TotalAccuracyMetric(use_cuda=self.use_cuda,
    #                             weblogger=self.weblogger, log_text=log_text)]
    #
    #     if self.max_iterations_testing != np.inf:
    #         all_cb += ([StopWhenMetricIs(value_to_reach=self.max_iterations_testing, metric_name='tot_iter')])
    #     return all_cb
    #
    # def test(self, net, test_loaders_list, callbacks=None, log_text: List[str] = None):
    #     self.net = net
    #     if self.use_cuda:
    #         self.net.cuda()
    #     if log_text is None:
    #         log_text = [f'{d.dataset.name_generator}' for d in test_loaders_list]
    #     print(fg.yellow + f"\n**Testing Started** [{log_text}]" + rs.fg)
    #
    #     self.experiment_loaders[self.current_run]['testing'].append(test_loaders_list)
    #     save_dataframe = True if self.output_filename is not None else False
    #
    #     conf_mat_acc_all_tests = []
    #     accuracy_all_tests = []
    #     text = []
    #     df_testing = []
    #     results = {}
    #     for idx, testing_loader in enumerate(test_loaders_list):
    #         results[log_text[idx]] = {}
    #         if self.max_iterations_testing == np.inf:
    #             strg = '1 epoch'
    #         else:
    #             strg = f'{np.min((self.max_iterations_testing, len(testing_loader.dataset)))} iterations'
    #         print(fg.green + f'\nTesting {idx + 1}/{len(test_loaders_list)}: ' + ef.inverse + ef.bold + f'[{testing_loader.dataset.name_generator}]' + rs.bold_dim + rs.inverse + f': {strg}' + rs.fg)
    #         all_cb = self.prepare_test_callbacks(log_text[idx] if log_text is not None else '', testing_loader, save_dataframe)
    #         all_cb += (callbacks or [])
    #         with torch.no_grad():
    #             net, logs = self.call_run(testing_loader,
    #                                       train=False,
    #                                       callbacks=all_cb)
    #
    #         results[log_text[idx]]['total_accuracy'] = logs['total_accuracy']
    #         results[log_text[idx]]['conf_mat_acc'] = logs['conf_mat_acc']
    #         if save_dataframe and 'dataframe' in logs:
    #             results[log_text[idx]]['df_testing'] = logs['dataframe']
    #
    #         if 'dataframe' not in logs:
    #             print("No Dataframe Saved (probably no callback for computing the dataframe was specified")
    #
    #     self.finalize_test(results)
    #
    #     return df_testing, conf_mat_acc_all_tests, accuracy_all_tests

    # def finalize_test(self, results):
    #     self.test_results_seed[self.current_run] = results
    #     self.test_results_seed[self.current_run]['seed'] = self.seed
    #
    # def save_all_runs_and_stop(self):
    #     if self.output_filename is not None:
    #         result_path_data = self.output_filename
    #         result_path_loaders = os.path.dirname(result_path_data) + '/loaders_' + os.path.basename(result_path_data)
    #         pathlib.Path(os.path.dirname(result_path_data)).mkdir(parents=True, exist_ok=True)
    #         cloudpickle.dump(self.test_results_seed, open(result_path_data, 'wb'))
    #         print('Saved data in {}, \nSaved loaders in [nope]'.format(result_path_data))  # result_path_loaders))
    #     else:
    #         Warning('Results path is not specified!')
    #     if self.weblogger == 2:
    #         neptune.stop()


# class SupervisedLearningConfig(Config):
#     def __init__(self, **kwargs):
#         # self.size_canvas = None
#         # self.batch_size = None
#         self.step = standard_net_step
#         super().__init__(**kwargs)
#
#     def parse_arguments(self, parser):
#         super().parse_arguments(parser)
#         parser.add_argument("-bsize", "--batch_size",
#                             help="batch_size",
#                             type=int,
#                             default=32)
#         return parser
#
#     def finalize_init(self, PARAMS, list_tags):
#         self.batch_size = PARAMS['batch_size']
#         # list_tags.append(f'bs{self.batch_size}') if self.batch_size != 32 else None
#         super().finalize_init(PARAMS, list_tags)
#
#     def call_run(self, loader, train=True, callbacks=None):
#         return run(loader,
#                    use_cuda=self.use_cuda,
#                    net=self.net,
#                    callbacks=callbacks,
#                    loss_fn=self.loss_fn,  # torch.nn.CrossEntropyLoss(),
#                    optimizer=self.optimizer,  # torch.optim.SGD(params_to_update, lr=0.1, momentum=0.9),
#                    iteration_step=self.step,
#                    train=train,
#                    loader=loader)
#
#     def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe, dataframe_saver=None):
#         num_classes = self._get_num_classes(testing_loader)
#
#         all_cb = super().prepare_test_callbacks(log_text, testing_loader, save_dataframe)
#         all_cb += [ComputeConfMatrix(num_classes=num_classes,
#                                      weblogger=self.weblogger,
#                                      weblog_text=log_text,
#                                      class_names=testing_loader.dataset.classes)]
#         if save_dataframe and dataframe_saver is None:
#             all_cb += ([GenericDataFrameSaver(num_classes,
#                                               self.use_cuda,
#                                               self.network_name, self.size_canvas,
#                                               weblogger=self.weblogger, log_text_plot=log_text)])
#         if save_dataframe and dataframe_saver is not None:
#             all_cb += ([dataframe_saver(num_classes,
#                                         self.use_cuda,
#                                         self.network_name, self.size_canvas,
#                                         weblogger=self.weblogger, log_text_plot=log_text)])
#         return all_cb

    @classmethod
    def get_net_ff(cls, network_name, imagenet_pt=False, num_classes=1000, **kwargs):
        # if pretraining == 'ImageNet':
        #     imagenet_pt = True
        if imagenet_pt:
            print(fg.red + "Loading ImageNet" + rs.fg)

        nc = 1000 if imagenet_pt else num_classes
        if network_name == 'vgg11':
            net = torchvision.models.vgg11(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg11bn':
            net = torchvision.models.vgg11_bn(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16':
            net = torchvision.models.vgg16(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg16bn':
            net =  torchvision.models.vgg16_bn(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'vgg19bn':
            net = torchvision.models.vgg19_bn(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'resnet18':
            net = torchvision.models.resnet18(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'resnet50':
            net = torchvision.models.resnet50(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'alexnet':
            net = torchvision.models.alexnet(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif network_name == 'inception_v3':  # nope
            net = torchvision.models.inception_v3(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif network_name == 'densenet121':
            net = torchvision.models.densenet121(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'densenet201':
            net = torchvision.models.densenet201(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif network_name == 'googlenet':
            net = torchvision.models.googlenet(pretrained=imagenet_pt, progress=True, num_classes=nc)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = cls.get_other_nets(network_name, num_classes, imagenet_pt, **kwargs)
            assert False if net is False else True, f"Network name {network_name} not recognized"

        return net

    @staticmethod
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


    def get_net(self, num_classes=None):
        self.net = self.get_net_ff(self.network_name, True if self.pretraining == 'ImageNet' else False, num_classes)
        pretraining_file = 'vanilla' if self.pretraining == 'ImageNet' else self.pretraining
        self.net = self.load_pretraining(self.net, pretraining_file, self.use_cuda)
        self.step = standard_net_step
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        framework_utils.print_net_info(self.net) if self.verbose else None
        self.net.cuda() if self.use_cuda else None

        return self.net

    @staticmethod
    def get_other_nets(network_name, num_classes, imagenet_pt):
        """
        A function that can be overwritten by subclasses to add more specialized networks
        """
        return False

    def prepare_train_callbacks(self, log_text, train_loader, test_loaders=None):
        num_classes = self._get_num_classes(train_loader)

        all_cb = super().prepare_train_callbacks(log_text, train_loader, test_loaders=test_loaders)
        # if self.weblogger:
        #     all_cb += [ComputeConfMatrix(num_classes=num_classes,
        #                                  weblogger=self.weblogger,
        #                                  weblog_text=log_text,
        #                                  reset_every=200)]
        #     all_cb +=[RollingAccEachClassWeblog(log_every=self.weblog_check_every,
        #                                          num_classes=num_classes,
        #                                          weblog_text=log_text,
        #                                          weblogger=self.weblogger)]
        return all_cb


class SaveModelBackbone(Callback):
    def __init__(self, net, backbone, output_path, log_in_weblogger=False):
        self.output_path = output_path
        self.net = net
        self.backbone = backbone
        super().__init__()

    def on_train_end(self, logs=None):
        if self.output_path is not None:
            pathlib.Path(os.path.dirname(self.output_path)).mkdir(parents=True, exist_ok=True)
            print('Saving model in {}'.format(self.output_path))
            torch.save({'full': self.net.state_dict(),
                        'backbone': self.backbone.state_dict() if self.backbone is not None else None,
                        'relation_module': self.net.relation_module.state_dict() if 'relation_module' in self.net._modules else None,
                        'classifier': self.net.classifier.state_dict() if 'classifier' in self.net._modules else None
                        }, self.output_path)

def create_backbone_exp(obj_class):
    class BackboneExp(obj_class):
        def __init__(self, **kwargs):
            self.backbone_name = None
            self.pretraining_backbone = None
            self.freeze_backbone = None
            super().__init__(**kwargs)

        def parse_arguments(self, parser):
            super().parse_arguments(parser)

            parser.add_argument("-bkbn", "--backbone_name",
                                help="The network structure used as a backbone [conv4-64], [conv5-128], [conv6-128], [conv6-256]",
                                type=str,
                                default=None)

            parser.add_argument("-ptb", "--pretraining_backbone",
                                help="The path for the backbone pretraining",
                                type=str,
                                default='vanilla')

            parser.add_argument("-freeze_bkbn", "--freeze_backbone",
                                help="",
                                type=lambda x: bool(int(x)),
                                default=False)
            return parser

        @abstractmethod
        def backbone(self) -> nn.Module:
            return NotImplementedError

        def finalize_init(self, PARAMS, list_tags):
            self.freeze_backbone = PARAMS['freeze_backbone']
            self.backbone_name = PARAMS['backbone_name']
            self.pretraining_backbone = PARAMS['pretraining_backbone']

            list_tags.append(f'bkbn{self.backbone_name}') if self.backbone_name is not None else None
            list_tags.append('bkbnfrozen') if self.freeze_backbone else None
            list_tags.append(f'bkbnpretrained' if self.pretraining_backbone != 'vanilla' else 'bkbnvanilla')

            super().finalize_init(PARAMS, list_tags)

        def prepare_train_callbacks(self, log_text, train_loader, test_loaders=None):
            all_cb = super().prepare_train_callbacks(log_text, train_loader, test_loaders=test_loaders)
            # This ComputeConfMatrix is used for matching, that's why num_class = 2
            all_cb = [i for i in all_cb if not isinstance(i, SaveModel)]  # eliminate old SaveModel
            all_cb += ([SaveModelBackbone(self.net, self.backbone(), self.model_output_filename, self.weblogger)] if self.model_output_filename is not None else [])
            if self.max_epochs != np.inf:
                all_cb += ([CallLrScheduler(scheduler=MultiStepLR(self.optimizer, [int(self.max_epochs/2), int(self.max_epochs * 0.75)], gamma=0.1), step_epoch=True, step_batch=False)])
            return all_cb

        def finish_get_net(self):
            self.net = self.pretraining_backbone_or_full()

            if self.freeze_backbone == 1:
                print(fg.red + "Freezing Backbone" + rs.fg)
                for param in self.backbone().parameters():
                    param.requires_grad = False
            elif self.freeze_backbone == 2:  # freeze intermediate conv layers
                for param in self.backbone().parameters():
                    param.requires_grad = False
                for param in self.backbone().features[:4].parameters():
                    param.requires_grad = True

            framework_utils.print_net_info(self.net) if self.verbose else None
            self.net.cuda() if self.use_cuda else None
            return self.net, self.net.parameters()

        def pretraining_backbone_or_full(self):
            if self.pretraining_backbone != 'vanilla':
                if os.path.isfile(self.pretraining_backbone):
                    self.backbone().load_state_dict(torch.load(self.pretraining_backbone, map_location='cuda' if self.use_cuda else 'cpu')['backbone'])
                    print(fg.red + f"Loaded backbone model from {self.pretraining_backbone}" + rs.fg)
                else:
                    assert False, f"Pretraining Backbone not found {self.pretraining_backbone}"

            if self.pretraining != 'vanilla':
                if os.path.isfile(self.pretraining):
                    print(fg.red + f"Loading.. full model from {self.pretraining}..." + rs.fg, end="")
                    ww = torch.load(self.pretraining, map_location='cuda' if self.use_cuda else 'cpu')['full']
                    self.net.load_state_dict(ww)
                    print(fg.red + " Done." + rs.fg)
                else:
                    assert False, f"Pretraining path not found {self.pretraining}"

            return self.net

    return BackboneExp


