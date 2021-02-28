from sty import fg, bg, ef, rs
import collections
import torchvision
import cloudpickle
import glob
from enum import Enum
import re
import argparse
from torch.optim import Adam
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Callable, Union
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss, NLLLoss
from generate_datasets.generators.translate_generator import TranslateGenerator
from callbacks import *
import framework_utils as utils
from models.meta_learning_models import MatchingNetwork, RelationNetSung, get_few_shot_encoder, get_few_shot_encoder_basic, get_few_shot_evaluator
from models.sequence_learner import *
from models.FCnets import FC4
from models.smallCNN import smallCNNnp, smallCNNp
from train_net import *
import time
import wandb
import random

# from wandb import magic

class Experiment(ABC):
    def __init__(self, experiment_class_name='default_name', parser=None, additional_tags=None):
        self.use_cuda = False
        self.loss_fn = None
        self.optimizer = None

        if torch.cuda.is_available():
            print('Using cuda - you are probably on the server')
            self.use_cuda = True
        if parser is None:
            parser = argparse.ArgumentParser(allow_abbrev=False)
        parser = self.parse_arguments(parser)

        PARAMS = vars(parser.parse_known_args()[0])
        self.net = None
        self.current_run = -1
        self.seed = PARAMS['seed']
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.num_runs = PARAMS['num_runs']  # this is not used here, and it shouldn't be here, but for now we are creating a weblogger session when an Experiment is created, and we want to save this as a parameter, so we need to do it here.
        inf_if_minus_one = lambda p: np.inf if p == -1 else p

        self.stop_when_train_acc_is = inf_if_minus_one(PARAMS['stop_when_train_acc_is'])
        self.max_epochs = inf_if_minus_one(PARAMS['max_epochs'])
        self.max_iterations = inf_if_minus_one(PARAMS['max_iterations'])
        self.max_iterations_testing = inf_if_minus_one(PARAMS['num_iterations_testing'])
        self.patience_stagnation = inf_if_minus_one(PARAMS['patience_stagnation'])

        self.pretraining = PARAMS['pretraining']
        self.model_output_filename = PARAMS['model_output_filename']
        self.output_filename = PARAMS['output_filename']
        self.network_name = PARAMS['network_name']
        self.learning_rate = PARAMS['learning_rate']
        self.force_cuda = PARAMS['force_cuda']
        self.additional_tags = PARAMS['additional_tags']
        self.weblogger = PARAMS['use_weblog']
        self.group_name = PARAMS['wandb_group_name']
        self.project_name = PARAMS['project_name']
        self.experiment_name = PARAMS['experiment_name']
        self.grayscale = bool(PARAMS['grayscale'])

        self.experiment_data = {}
        self.experiment_loaders = {}  # we separate data from loaders because loaders are pickled objects and may broke when module name changes. If this happens, at least we preserve the data. We generally don't even need the loaders much.

        self.weblog_check_every = 5
        self.console_check_every = 100
        if self.force_cuda:
            self.use_cuda = True
        elif self.force_cuda == 0:
            print("Cuda Forcesully Disabled")
            self.use_cuda = False
        list_tags = []
        if self.additional_tags is not None:
            [list_tags.append(i) for i in self.additional_tags.split('_') if i != 'emptytag']
        list_tags.extend(additional_tags) if additional_tags is not None else None
        list_tags.append('ptvanilla') if self.pretraining == 'vanilla' else None
        list_tags.append('ptImageNet') if self.pretraining == 'ImageNet' else None
        list_tags.append(self.network_name)
        list_tags.append('lr{}'.format("{:2f}".format(self.learning_rate).split(".")[1])) if self.learning_rate is not None else None
        list_tags.append('gray') if self.grayscale else None
        if self.max_iterations is None:
            self.max_iterations = 5000 if self.use_cuda else 10

        self.finalize_init(PARAMS, list_tags)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_optimizer(selfs, optimizer):
        self.optimizer = optimizer

    def parse_arguments(self, parser):
        parser.add_argument('-seed', "--seed", type=int, default=None)
        parser.add_argument("-expname", "--experiment_name",
                            help="Name of the experiment session, used as a name in the weblogger",
                            type=str,
                            default=None)
        parser.add_argument("-r", "--num_runs",
                            help="run experiment n times",
                            type=int,
                            default=1)
        parser.add_argument("-fcuda", "--force_cuda",
                            help="Force to run it with cuda enabled or disabled (1/0). Set None to check if the GPU is available",
                            type=int,
                            default=None)
        parser.add_argument("-grayscale", "--grayscale",
                            help="Set the grayscale flag to true",
                            type=int,
                            default=0)
        parser.add_argument("-weblog", "--use_weblog",
                            help="Log stuff to the weblogger [0=none, 1=wandb, 2=neptune])",
                            type=int,
                            default=2)
        parser.add_argument("-tags", "--additional_tags",
                            help="Add additional tags. Separate them by underscore. E.g. tag1_tag2",
                            type=str,
                            default=None)
        parser.add_argument("-prjnm", "--project_name",
                            type=str,
                            default='TestProject')
        parser.add_argument("-wbg", "--wandb_group_name",
                            help="Group name for weight and biases, to organize sub experiments of a bigger project",
                            type=str,
                            default=None)
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
        parser.add_argument("-nt", "--num_iterations_testing",
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
        self.experiment_data[self.current_run] = {}
        self.experiment_loaders[self.current_run] = {'training': [],
                                                     'testing': []}
        print('Run Number {}'.format(self.current_run))

    def finalize_init(self, PARAMS, list_tags):
        print(fg.magenta)
        print('**LIST_TAGS**:')
        print(list_tags)
        print('***PARAMS***')
        if not self.use_cuda:
            list_tags.append('LOCALTEST')

        for i in sorted(PARAMS.keys()):
            print(f'\t{i} : ' + ef.inverse + f'{PARAMS[i]}' + rs.inverse)
        if self.weblogger == 1:
            a = time.time()
            wandb.init(name=self.experiment_name, project=self.project_name, tags=list_tags, group=self.group_name, config=PARAMS)
            print('Weblogger Creation: WANDB {}'.format(time.time() - a))
        if self.weblogger == 2:
            PARAMS.update({'group_name': self.group_name})
            neptune.init(f'valeriobiscione/{self.project_name}')
            neptune.create_experiment(name=self.experiment_name,
                                      params=PARAMS,
                                      tags=list_tags)
        print(rs.fg)
        self.new_run()


    @abstractmethod
    def call_run(self, loader, callbacks=None, train=True):
        pass

    @abstractmethod
    def get_net(self, num_classes):
        NotImplementedError

    def prepare_train_callbacks(self, log_text, train_loader, test_loaders=None):
        def stop(logs, cb):
            logs['stop'] = True
            print('Early Stopping')


        num_classes = self._get_num_classes(train_loader)

        all_cb = [
                  EndEpochStats(),
                  StandardMetrics(log_every=self.console_check_every, print_it=True,
                                        use_cuda=self.use_cuda,
                                        weblogger=0, log_text=log_text,
                                        metrics_prefix='cnsl'),

                  StopFromUserInput(),
                  # PlotTimeElapsed(time_every=100),
                  TotalAccuracyMetric(use_cuda=self.use_cuda,
                                      to_weblog=self.weblogger, log_text=log_text)]

        if self.stop_when_train_acc_is != np.inf:
            all_cb += (
                [TriggerActionWithPatience(min_delta=0.01, patience=800, percentage=True, mode='max',
                                      reaching_goal=self.stop_when_train_acc_is,
                                      metric_name='webl/mean_acc' if self.weblogger else 'cnsl/mean_acc',
                                      check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
                                      triggered_action=stop,
                                      action_name='Early Stopping',
                                      alpha=0.5)])  # once reached a certain accuracy
        if self.max_epochs != np.inf:
            all_cb +=([StopWhenMetricIs(value_to_reach=self.max_epochs - 1, metric_name='epoch', check_after_batch=False)])  # you could use early stopping for that

        if self.max_iterations != np.inf:
            all_cb += ([StopWhenMetricIs(value_to_reach=self.max_iterations - 1, metric_name='tot_iter')])  # you could use early stopping for that

        if self.patience_stagnation != np.inf:
            all_cb += ([TriggerActionWithPatience(min_delta=0.01, patience=self.patience_stagnation, percentage=True, mode='min',
                                            reaching_goal=None,
                                            metric_name='webl/mean_loss' if self.weblogger else 'cnsl/mean_loss',
                                            check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
                                            triggered_action=stop,
                                            action_name='Early Stopping',
                                            alpha=0.5)])  # for stagnation
        # all_cb += ([SaveModel(self.net, self.model_output_filename, self.weblogger)] if self.model_output_filename is not None else [])
        if test_loaders is not None:
            for t in test_loaders:
                all_cb += ([EndEpochTest(testing_loader=t, every_x_epochs=50, weblogger=self.weblogger, log_text=f'test {t.dataset.name_generator}', use_cuda=self.use_cuda, call_run=self.call_run)])

        if self.weblogger:
            all_cb += [StandardMetrics(log_every=self.weblog_check_every, print_it=False,
                                       use_cuda=self.use_cuda,
                                       weblogger=self.weblogger, log_text=log_text,
                                       metrics_prefix='webl')]
        #                PlotGradientWeblog(net=self.net, log_every=50, plot_every=500, log_txt=log_text, weblogger=self.weblogger)]

        return all_cb


    def _get_num_classes(self, loader):
        return loader.dataset.num_classes

    def train(self, train_loader, callbacks=None, test_loaders=None, log_text='train'):
        print(f"**Training** [{log_text}]")
        self.net = self.get_net(new_num_classes=self._get_num_classes(train_loader))
        if not self.optimizer or not self.loss_fn:
                assert False, "Optimizer or Loss Function not set"
        self.experiment_loaders[self.current_run]['training'].append(train_loader)
        all_cb = self.prepare_train_callbacks(log_text, train_loader, test_loaders=test_loaders)

        all_cb += (callbacks or [])

        if self.use_cuda:
            self.net.cuda()
        self.net.train()
        net, logs = self.call_run(train_loader,
                                  train=True,
                                  callbacks=all_cb,
                                  )
        return net

    def finalize_test(self, df_testing, conf_mat, accuracy, id_text):
        self.experiment_data[self.current_run]['df_testing'] = df_testing
        self.experiment_data[self.current_run]['conf_mat_acc'] = conf_mat
        self.experiment_data[self.current_run]['accuracy'] = accuracy
        self.experiment_data[self.current_run]['id_text'] = id_text

    def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe):
        all_cb = [
                   StandardMetrics(log_every=3000, print_it=True,
                                  use_cuda=self.use_cuda,
                                  weblogger=0, log_text=log_text,
                                  metrics_prefix='cnsl'),
                  PlotTimeElapsed(time_every=3000),

                  StopFromUserInput(),
                  StopWhenMetricIs(value_to_reach=0, metric_name='epoch', check_after_batch=False),  # you could use early stopping for that

            TotalAccuracyMetric(use_cuda=self.use_cuda,
                                      to_weblog=self.weblogger, log_text=log_text)]

        if self.max_iterations_testing != np.inf:
            all_cb += ([StopWhenMetricIs(value_to_reach=self.max_iterations_testing, metric_name='tot_iter')])
        return all_cb


    def test(self, net, test_loaders_list, callbacks=None, log_text: List[str] = None):
        self.net = net
        if self.use_cuda:
            self.net.cuda()
        self.net.eval()
        if log_text is None:
            log_text = [f'{d.dataset.name_generator}' for d in test_loaders_list]
        print(fg.yellow + f"\n**Testing Started** [{log_text}]" + rs.fg)

        self.experiment_loaders[self.current_run]['testing'].append(test_loaders_list)
        save_dataframe = True if self.output_filename is not None else False

        conf_mat_acc_all_tests = []
        accuracy_all_tests = []
        text = []
        df_testing = []
        for idx, testing_loader in enumerate(test_loaders_list):
            if self.max_iterations_testing==np.inf:
                strg = '1 epoch'
            else:
                strg =  f'{np.min((self.max_iterations_testing, len(testing_loader.dataset)))} iterations'
            print(fg.green + f'\nTesting {idx+1}/{len(test_loaders_list)}: ' + ef.inverse + ef.bold + f'[{testing_loader.dataset.name_generator}]' + rs.bold_dim + rs.inverse + f': {strg}' + rs.fg)
            all_cb = self.prepare_test_callbacks(log_text[idx] if log_text is not None else '', testing_loader, save_dataframe)
            all_cb += (callbacks or [])
            with torch.no_grad():
                net, logs = self.call_run(testing_loader,
                                          train=False,
                                          callbacks=all_cb)
            conf_mat_acc_all_tests.append(logs['conf_mat_acc'])
            accuracy_all_tests.append(logs['total_accuracy'])
            text.append(log_text[idx])
            if save_dataframe and 'dataframe' in logs:
                df_testing.append(logs['dataframe'])
            if 'dataframe' not in logs:
                print("No Dataframe Saved (probably no callback for computing the dataframe was specified")

        self.finalize_test(df_testing, conf_mat_acc_all_tests, accuracy_all_tests, text)

        return df_testing, conf_mat_acc_all_tests, accuracy_all_tests

    def save_all_runs(self):
        if self.output_filename is not None:
            result_path_data = self.output_filename
            result_path_loaders = os.path.dirname(result_path_data) + '/loaders_' + os.path.basename(result_path_data)
            pathlib.Path(os.path.dirname(result_path_data)).mkdir(parents=True, exist_ok=True)
            cloudpickle.dump(self.experiment_data, open(result_path_data, 'wb'))
            print('Saved data in {}, \nSaved loaders in [nope]'.format(result_path_data))  #result_path_loaders))
        else:
            Warning('Results path is not specified!')


class SupervisedLearningExperiment(Experiment):
    def __init__(self, **kwargs):
        self.size_canvas = None
        self.batch_size = None
        super().__init__(**kwargs)

    def parse_arguments(self, parser):
        super().parse_arguments(parser)
        parser.add_argument("-bsize", "--batch_size",
                            help="batch_size",
                            type=int,
                            default=32)
        return parser

    def finalize_init(self, PARAMS, list_tags):
        self.batch_size = PARAMS['batch_size']

        list_tags.append(f'bs{self.batch_size}') if self.batch_size != 32 else None
        super().finalize_init(PARAMS, list_tags)

    def call_run(self, loader, train=True, callbacks=None):
        return run(loader,
                   use_cuda=self.use_cuda,
                   net=self.net,
                   callbacks=callbacks,
                   loss_fn=self.loss_fn, # torch.nn.CrossEntropyLoss(),
                   optimizer=self.optimizer, #torch.optim.SGD(params_to_update, lr=0.1, momentum=0.9),
                   iteration_step=standard_net_step,
                   iteration_step_kwargs={'train': train},
                   )

    def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe):
        num_classes = self._get_num_classes(testing_loader)

        all_cb = super().prepare_test_callbacks(log_text, testing_loader, save_dataframe)
        all_cb += [ComputeConfMatrix(num_classes=num_classes,
                                     weblogger=self.weblogger,
                                     weblog_text=log_text,
                                     class_names=testing_loader.dataset.classes)]
        all_cb += ([ComputeDataFrame(num_classes,
                                     self.use_cuda,
                                     self.network_name, self.size_canvas,
                                     weblogger=self.weblogger, log_text_plot=log_text)]
                   if save_dataframe else[])
        return all_cb

    @abstractmethod
    def get_net(self, num_classes=None):
        NotImplementedError

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



class SequentialMetaLearningExp(Experiment):
    """
    Select network_name = 'matching_net_basics' for using the basic network that works for any image size, used in the paper for Omniglot.
                        'matching_net_more' to use a more complex version of the matching net, with more conv layer. It can still accept any type of input
                        'matching_net_plus' the decision is made by an evaluation network. It accepts 128x128px images.
                        #ToDo: with a adaptive average pool make it accept whatever size
    """

    def __init__(self, **kwargs):
        self.training_folder = None
        self.testing_folder = None
        self.nSc = None
        self.nSt = None
        self.nFc = None
        self.nFt = None
        self.k = None
        self.step = None
        super().__init__(**kwargs)

    def parse_arguments(self, parser):
        super().parse_arguments(parser)
        parser.add_argument("-nSc", "--nSc",
                            help="Num sequences for each candidate.",
                            type=int,
                            default=2)
        parser.add_argument("-nSt", "--nSt",
                            help="Num sequences for each training object.",
                            type=int,
                            default=2)
        parser.add_argument("-nFc", "--nFc",
                            help="Num frames for each sequence in the candidate.",
                            type=int,
                            default=2)
        parser.add_argument("-nFt", "--nFt",
                            help="Num frames for each sequence for each training object.",
                            type=int,
                            default=2)
        parser.add_argument("-k", "--k",
                            help="Num of camera sets",
                            type=int,
                            default=2)
        return parser

    def finalize_init(self, PARAMS, list_tags):
        self.nSt = PARAMS['nSt']
        self.nSc = PARAMS['nSc']
        self.nFt = PARAMS['nFt']
        self.nFc = PARAMS['nFc']
        self.k = PARAMS['k']
        list_tags.append(f'{self.k}x{self.nSt}x{self.nSc}x{self.nFt}x{self.nFc}')
        list_tags.append(self.network_name)
        super().finalize_init(PARAMS, list_tags)

    # def get_net(self, new_num_classes=None):
    #     # ToDo: Matching Learning pretrain
    #     device = torch.device('cuda' if self.use_cuda else 'cpu')
    #     if self.network_name == 'seqNt1c':
    #         assert self.nSc == 1 and self.nFc == 1, f"With the model {network_name} you need to set nSc and nFc to 1"
    #         net = SequenceNtrain1cand(grayscale=self.grayscale)
    #         self.step = sequence_net_Ntrain_1cand
    #         self.loss_fn = MSELoss()  # CrossEntropyLoss()
    #     elif self.network_name == 'relation_net':
    #         assert self.nSc <= 1 and self.nFc <= 1 and self.nSt <= 1 and self.nFt <= 1
    #         net = RelationNetSung(backbone_name='conv4', size_canvas=self.size_canvas, grayscale=self.grayscale)
    #         self.step = sequence_net_Ntrain_1cand
    #         self.loss_fn = MSELoss()
    #     else:
    #         assert False, f"network name {self.network_name} not recognized"
    #     if self.pretraining != 'vanilla':
    #         if os.path.isfile(self.pretraining):
    #             print(f"Pretraining value should be a path when used with FewShotLearning (not ImageNet, etc.). Instead is {self.pretraining}")
    #         net.load_state_dict(torch.load(self.pretraining, map_location=torch.device('cuda' if self.use_cuda else 'cpu')))
    #
    #     self.optimizer = Adam(net.parameters(), lr=0.001 if self.learning_rate is None else self.learning_rate),
    #     framework_utils.print_net_info(self.net)
    #
    #     return net

    def call_run(self, data_loader, callbacks=None, train=True):
        return run(data_loader, use_cuda=self.use_cuda, net=self.net,
                   callbacks=callbacks,
                   loss_fn=self.loss_fn,
                   optimizer=self.optimizer,
                   iteration_step=self.step,
                   iteration_step_kwargs={'train': train,
                                          'dataset': data_loader.dataset},
                   )


    def prepare_train_callbacks(self, log_text, train_loader, test_loaders=None):
        all_cb = super().prepare_train_callbacks(log_text, train_loader, test_loaders=test_loaders)
        # This ComputeConfMatrix is used for matching, that's why num_class = 2
        all_cb += [ComputeConfMatrix(num_classes=2,
                                     weblogger=self.weblogger,
                                     weblog_text=log_text,
                                     class_names=train_loader.dataset.classes)]
        return all_cb

    def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe):
        all_cb = super().prepare_test_callbacks(log_text, testing_loader, save_dataframe)
        if save_dataframe:
            all_cb += [ComputeConfMatrix(num_classes=2,
                                         weblogger=self.weblogger,
                                         weblog_text=log_text,
                                         class_names=testing_loader.dataset.classes),
                       ]
        return all_cb


def with_dataset_name(class_obj, add_tags=True):
    class BuildDatasetExp(class_obj):
        def __init__(self, **kwargs):
            self.name_dataset_training = None
            self.name_datasets_testing = None
            self.add_tags = add_tags
            super().__init__(**kwargs)

        def parse_arguments(self, parser):
            super().parse_arguments(parser)
            parser.add_argument("-name_dataset_training", "--name_dataset_training",
                                help="Select the name of the dataset used for training.",
                                type=str,
                                default=None)
            parser.add_argument("-name_dataset_testing", "--name_dataset_testing",
                                help="Select the name of the dataset used for testing.",
                                type=str,
                                default=None)
            return parser
        def finalize_init(self, PARAMS, list_tags):
            self.name_dataset_training = PARAMS['name_dataset_training']
            self.name_datasets_testing = PARAMS['name_dataset_testing']
            if self.name_datasets_testing is None and self.name_dataset_training is None:
                self.play_mode = True
                self.name_dataset_training = None
            if self.output_filename is None and self.name_datasets_testing is not None:
                assert False, "You provided some dataset for testing, but no output. This is almost always a mistake"
            if self.add_tags:
                list_tags.append(f"tr{self.name_dataset_training.split('data')[-1]}") if self.name_dataset_training is not None else None
                [list_tags.append(f"te{i.split('data')[-1]}") for i in self.name_datasets_testing.split('_')] if self.name_datasets_testing is not None else None

            if self.name_datasets_testing is not None:
                self.name_datasets_testing = str.split(self.name_datasets_testing, "_")
            else:
                self.name_datasets_testing = []

            super().finalize_init(PARAMS, list_tags)

    return BuildDatasetExp


def unity_builder_class(class_obj):
    class UnityExp(class_obj):
        def __init__(self, **kwargs):
            self.size_canvas = None
            self.play_mode = False
            super().__init__(**kwargs)

        def parse_arguments(self, parser):
            super().parse_arguments(parser)
            parser.add_argument("-sc", "--size_canvas_resize",
                                help="Change the size of the image passed by Unity. Put 0 to prevent resizing",
                                type=str,
                                default='224_224')
            parser.add_argument("-play", "--play_mode",
                                help="Execute in play mode",
                                type=lambda x: bool(int(x)),
                                default=0)
            return parser

        def finalize_init(self, PARAMS, list_tags):
            self.play_mode = PARAMS['play_mode']
            self.size_canvas = PARAMS['size_canvas_resize']
            # list_tags.append("Unity")
            list_tags.append('sc{}'.format(str(self.size_canvas).replace(', ', 'x'))) if self.size_canvas != '0' else None

            if self.size_canvas == '0':
                self.size_canvas = (128, 128)  # ToDo: this could be taken from the unity channel
            else:
                self.size_canvas = tuple([int(i) for i in self.size_canvas.split("_")])
            super().finalize_init(PARAMS, list_tags)

        def prepare_train_callbacks(self, log_text, train_loader, test_loaders=None):
            class CheckStoppingLevel:
                there_is_another_level = True

                def go_next_level(self, logs, *args, **kwargs):
                    if self.there_is_another_level:
                        self.there_is_another_level = train_loader.dataset.sampler.env_params.next_level()
                        framework_utils.weblog_dataset_info(train_loader, f'Increase {train_loader.dataset.sampler.env_params.count_increase}', weblogger=2)  # the idea is that after reaching the end level, we wait another stagnMation
                    else:
                        print("No new level, reached maximum")
                        logs['stop'] = True

            # AverageChangeMetric(loss_or_acc='acc',  use_cuda=self.use_cuda, log_every=nept_check_every, log_text='avrgChange/acc'),
            all_cb = super().prepare_train_callbacks(log_text, train_loader, test_loaders=test_loaders)
            ck = CheckStoppingLevel()
            all_cb += [PlotUnityImagesEveryOnceInAWhile(dataset=train_loader.dataset,
                                                        grayscale=self.grayscale,
                                                        plot_every=1000,
                                                        plot_only_n_times=20),
                       # TriggerActionWithPatience(min_delta=0.01, patience=400, percentage=True, mode='max',
                       #                           reaching_goal=90,
                       #                           metric_name='webl/mean_acc' if self.weblogger else 'cnsl/mean_acc',
                       #                           check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
                       #                           triggered_action=ck.go_next_level)]
            ]
            return all_cb
        def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe):
            all_cb = super().prepare_test_callbacks(log_text, testing_loader, save_dataframe)
            all_cb += [PlotUnityImagesEveryOnceInAWhile(dataset=testing_loader.dataset,
                                                        grayscale=self.grayscale,
                                                        plot_every=1000,
                                                        plot_only_n_times=5),
                       ComputeDataFrame3DsequenceLearning(k=testing_loader.dataset.sampler.k,
                                                          nSt=testing_loader.dataset.sampler.nSt,
                                                          nSc=testing_loader.dataset.sampler.nSc,
                                                          nFt=testing_loader.dataset.sampler.nFt,
                                                          nFc=testing_loader.dataset.sampler.nFc,
                                                          num_classes=self.k,
                                                          use_cuda=self.use_cuda,
                                                          network_name=self.network_name,
                                                          size_canvas=self.size_canvas,
                                                          log_text_plot=log_text,
                                                          weblogger=self.weblogger,
                                                          output_and_softmax=False)
                       ]
            return all_cb
    return UnityExp


sequence_unity_meta_learning_exp = unity_builder_class(with_dataset_name(SequentialMetaLearningExp))

