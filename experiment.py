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
from models.supervised_models import *
from models.smallCNN import smallCNNnp, smallCNNp
from train_net import *
import time
import wandb
from generate_datasets.generators.unity_metalearning_generator import UnityGenMetaLearning

# from wandb import magic

class Experiment(ABC):
    def __init__(self, experiment_class_name='default_name', parser=None, additional_tags=None):
        self.use_cuda = False
        if torch.cuda.is_available():
            print('Using cuda - you are probably on the server')
            self.use_cuda = True
        if parser is None:
            parser = argparse.ArgumentParser(allow_abbrev=False)
        parser = self.parse_arguments(parser)

        PARAMS = vars(parser.parse_known_args()[0])
        self.net = None
        self.current_run = -1
        # self.experiment_name = experiment_name
        self.num_runs = PARAMS['num_runs']  # this is not used here, and it shouldn't be here, but for now we are creating a weblogger session when an Experiment is created, and we want to save this as a parameter, so we need to do it here.
        self.pretraining = PARAMS['pretraining']
        self.stop_when_train_acc_is = PARAMS['stop_when_train_acc_is']
        self.max_iterations = PARAMS['max_iterations']
        self.model_output_filename = PARAMS['model_output_filename']
        self.output_filename = PARAMS['output_filename']
        self.num_iterations_testing = PARAMS['num_iterations_testing']
        self.network_name = PARAMS['network_name']
        self.learning_rate = PARAMS['learning_rate']
        self.force_cuda = PARAMS['force_cuda']
        self.additional_tags = PARAMS['additional_tags']
        self.weblogger = PARAMS['use_weblog']
        self.group_name = PARAMS['wandb_group_name']
        self.project_name = PARAMS['project_name']
        self.patience_stagnation = PARAMS['patience_stagnation']
        self.experiment_name = PARAMS['experiment_name']
        self.grayscale = bool(PARAMS['grayscale'])
        self.experiment_data = {}
        self.experiment_loaders = {}  # we separate data from loaders because loaders are pickled objects and may broke when module name changes. If this happens, at least we preserve the data. We generally don't even need the loaders much.


        self.weblog_check_every = 5
        self.console_check_every = 100
        if self.force_cuda:
            self.use_cuda = True

        # list_tags = [experiment_name] if experiment_name != '' else []
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


    def parse_arguments(self, parser):
        parser.add_argument("-expname", "--experiment_name",
                            help="Name of the experiment session, used as a name in the weblogger",
                            type=str,
                            default=None)
        parser.add_argument("-r", "--num_runs",
                            help="run experiment n times",
                            type=int,
                            default=1)
        parser.add_argument("-fcuda", "--force_cuda",
                            help="Force to run it with cuda enabled (for testing)",
                            type=int,
                            default=0)
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
                            help="Patience for early stopping for stagnation (num iter)",
                            type=int,
                            default=800)

        parser.add_argument("-mo", "--model_output_filename",
                            help="file name of the trained model",
                            type=str,
                            default=None)
        parser.add_argument("-o", "--output_filename",
                            help="output file name for the pandas dataframe files",
                            type=str,
                            default=None)
        parser.add_argument("-nt", "--num_iterations_testing",
                            default=300,
                            type=int)
        parser.add_argument("-lr", "--learning_rate",
                            default=None, help='learning rate. If none the standard one will be chosen',
                            type=float)
        parser.add_argument("-sa", "--stop_when_train_acc_is",
                            default=75, type=int)
        parser.add_argument("-pt", "--pretraining",
                            help="use [vanilla], [ImageNet (only for standard exp)] or a path",
                            type=str,
                            default='vanilla')
        parser.add_argument("-mi", "--max_iterations",
                            help="max number of batch iterations",
                            type=int,
                            default=None)
        parser.add_argument("-n", "--network_name", help="[vgg11] [vgg11_bn] [vgg16] [vgg16_bn]",
                            default=None,
                            type=str)
        return parser

    def new_run(self):
        self.current_run += 1
        self.experiment_data[self.current_run] = {}
        self.experiment_loaders[self.current_run] = {'training': [],
                                                     'testing': []}
        print('Run Number {}'.format(self.current_run))

    def finalize_init(self, PARAMS, list_tags):
        print('**LIST_TAGS**:')
        print(list_tags)
        print('***PARAMS***')
        if not self.use_cuda:
            list_tags.append('LOCALTEST')

        for i in sorted(PARAMS.keys()):
            print(f'\t{i} : {PARAMS[i]}')
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

        self.new_run()

    @staticmethod
    def weblogging_plot_generators_info(train_loader=None, test_loaders_list=None, weblogger=1, num_batches_to_log=2):
        if weblogger:
            if train_loader is not None:
                utils.weblog_dataset_info(train_loader, log_text=train_loader.dataset.name_generator, weblogger=weblogger, num_batches_to_log=num_batches_to_log)
            if test_loaders_list is not None:
                for loader in test_loaders_list:
                    utils.weblog_dataset_info(loader, log_text=loader.dataset.name_generator, weblogger=weblogger, num_batches_to_log=num_batches_to_log)

    @abstractmethod
    def call_run(self, train_loader, params_to_update, callbacks=None, train=True, epochs=20):
        pass

    @abstractmethod
    def get_net(self, new_num_classes=None):
        return None, None

    def prepare_train_callbacks(self, log_text, train_loader):
        def stop(logs, cb):
            logs['stop'] = True
            print('Early Stopping: {}'.format(cb.string))

        num_classes = self._get_num_classes(train_loader)

        all_cb = [StandardMetrics(log_every=self.console_check_every, print_it=True,
                                  use_cuda=self.use_cuda,
                                  weblogger=0, log_text=log_text,
                                  metrics_prefix='cnsl'),
                  TriggerActionWithPatience(min_delta=0.01, patience=800, percentage=True, mode='max',
                                            reaching_goal=self.stop_when_train_acc_is,
                                            metric_name='webl/mean_acc' if self.weblogger else 'cnsl/mean_acc',
                                            check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
                                            triggered_action=stop),  # once reached a certain accuracy
                  TriggerActionWithPatience(min_delta=0.01, patience=self.patience_stagnation, percentage=True, mode='min',
                                            reaching_goal=None,
                                            metric_name='webl/mean_loss' if self.weblogger else 'cnsl/mean_loss',
                                            check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
                                            triggered_action=stop),  # for stagnation
                  StopWhenMetricIs(value_to_reach=self.max_iterations, metric_name='tot_iter'),  # you could use early stopping for that


                  StopFromUserInput(),
                  PlotTimeElapsed(time_every=100),
                  TotalAccuracyMetric(use_cuda=self.use_cuda,
                                      to_weblog=self.weblogger, log_text=log_text)]

        all_cb += ([SaveModel(self.net, self.model_output_filename, self.weblogger)] if self.model_output_filename is not None else [])
        if self.weblogger:
            all_cb += [StandardMetrics(log_every=self.weblog_check_every, print_it=False,
                                       use_cuda=self.use_cuda,
                                       weblogger=self.weblogger, log_text=log_text,
                                       metrics_prefix='webl'),

                       PlotGradientWeblog(net=self.net, log_every=50, plot_every=500, log_txt=log_text, weblogger=self.weblogger)]

        return all_cb

    def _get_num_classes(self, loader):
        return loader.dataset.num_classes

    def train(self, train_loader, callbacks=None, log_text='train'):
        print(f"**Training** [{log_text}]");
        self.experiment_loaders[self.current_run]['training'].append(train_loader)
        self.net, params_to_update = self.get_net(new_num_classes=self._get_num_classes(train_loader))
        all_cb = self.prepare_train_callbacks(log_text, train_loader)

        all_cb += (callbacks or [])

        net, logs = self.call_run(train_loader,
                                  params_to_update=params_to_update,
                                  train=True, callbacks=all_cb)
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
                  StopWhenMetricIs(value_to_reach=self.num_iterations_testing, metric_name='tot_iter'),
                  TotalAccuracyMetric(use_cuda=self.use_cuda,
                                      to_weblog=self.weblogger, log_text=log_text)]

        return all_cb

    def test(self, net, test_loaders_list, callbacks=None, log_text: List[str] = None):
        self.net = net
        if log_text is None:
            log_text = [d.dataset.name_generator for d in test_loaders_list]
        print(f"**Testing Started** [{log_text}]")

        self.experiment_loaders[self.current_run]['testing'].append(test_loaders_list)
        save_dataframe = True if self.output_filename is not None else False

        conf_mat_acc_all_tests = []
        accuracy_all_tests = []
        text = []
        df_testing = []
        # df_testing = pd.DataFrame([])
        for idx, testing_loader in enumerate(test_loaders_list):
            # print('Testing on [{}], [{}]'.format(testing_loader.dataset.name_generator, testing_loader.dataset.translation_type_str if isinstance(testing_loader.dataset, TranslateGenerator) else 'no translation'))
            print(f'Testing {idx+1}/{len(test_loaders_list)}: [{testing_loader.dataset.name_generator}]: {np.min((self.num_iterations_testing, len(testing_loader.dataset)))}')
            # testing_loader.dataset.translation_type_str if isinstance(testing_loader.dataset, TranslateGenerator) else 'no transl'
            all_cb = self.prepare_test_callbacks(log_text[idx] if log_text is not None else '', testing_loader, save_dataframe)
            all_cb += (callbacks or [])

            net, logs = self.call_run(testing_loader,
                                      params_to_update=net.parameters(), train=False,
                                      callbacks=all_cb,
                                      epochs=1)
            conf_mat_acc_all_tests.append(logs['conf_mat_acc'])
            accuracy_all_tests.append(logs['total_accuracy'])
            text.append(log_text[idx])
            if save_dataframe:
                df_testing.append(logs['dataframe'])

        self.finalize_test(df_testing, conf_mat_acc_all_tests, accuracy_all_tests, text)

        return df_testing, conf_mat_acc_all_tests, accuracy_all_tests

    def save_all_runs(self):
        if self.output_filename is not None:
            result_path_data = self.output_filename
            result_path_loaders = os.path.dirname(result_path_data) + '/loaders_' + os.path.basename(result_path_data)
            pathlib.Path(os.path.dirname(result_path_data)).mkdir(parents=True, exist_ok=True)
            cloudpickle.dump(self.experiment_data, open(result_path_data, 'wb'))
            # cloudpickle.dump(self.experiment_loaders, open(result_path_loaders, 'wb'))
            print('Saved data in {}, \nSaved loaders in [nope]'.format(result_path_data))  #result_path_loaders))
        else:
            Warning('Results path is not specified!')


class TypeNet(Enum):
    VGG = 0
    FC = 1
    RESNET = 2
    SMALL_CNN = 3
    OTHER = 4


class SupervisedLearningExperiment(Experiment):
    def __init__(self, **kwargs):
        self.use_gap, self.feature_extraction, self.big_canvas, self.shallow_FC = False, False, False, False
        self.size_canvas = None
        self.batch_size = None
        self.scramble_fc = None
        self.scramble_conv = None
        self.freeze_fc = None
        self.size_object = None
        super().__init__(**kwargs)

    def parse_arguments(self, parser):
        super().parse_arguments(parser)
        parser.add_argument("-sFC", "--shallow_FC",
                            help='use a shallow fully connected layer (only a connection x to num_classes)',
                            default=0, type=int)
        parser.add_argument("-gap", "--use_gap",
                            help="use GAP layer at the end of the convolutional layers",
                            type=int,
                            default=0)
        # parser.add_argument("-so", "--size_object",
        #                     help="Change the size of the object (for translation exp). W_H (x, y). Set to 0 if you don't want to resize the object",
        #                     type=str,
        #                     default='50_50')
        # parser.add_argument("-sc", "--size_canvas",
        #                     help="Change the size of the image passed by Unity. Put 0 to avoid any resizing",
        #                     type=str,
        #                     default='0')
        parser.add_argument("-f", "--feature_extraction",
                            help="freeze the feature (conv) layers part of the VGG net",
                            type=int,
                            default=0)
        # parser.add_argument("-bC", "--big_canvas",
        #                     help="If true, will use 400x400 canvas (otherwise 224x224). The VGG network will be changed accordingly (we won't use the adaptive GAP)",
        #                     type=int,
        #                     default=0)
        parser.add_argument("-batch", "--batch_size",
                            help="batch_size",
                            type=int,
                            default=32)
        parser.add_argument("-freeze_fc", "--freeze_fc",
                            help="Freeze the fully connected layer",
                            type=int,
                            default=0)
        parser.add_argument("-scramble_fc", "--scramble_fc",
                            help="When using a pretrain network, do not copy the fc weights",
                            type=int,
                            default=0)
        parser.add_argument("-scramble_conv", "--scramble_conv",
                            help="When using a pretrain network, do not copy the conv weights",
                            type=int,
                            default=0)
        return parser

    def finalize_init(self, PARAMS, list_tags):
        self.use_gap = PARAMS['use_gap']
        self.feature_extraction = PARAMS['feature_extraction']
        # self.big_canvas = PARAMS['big_canvas']
        self.shallow_FC = PARAMS['shallow_FC']
        # self.size_canvas = PARAMS['size_canvas']
        self.batch_size = PARAMS['batch_size']
        self.freeze_fc = PARAMS['freeze_fc']
        self.scramble_fc = PARAMS['scramble_fc']
        self.scramble_conv = PARAMS['scramble_conv']
        # if not self.use_cuda:
        #     print(f'Not using cuda. Batch size changed from {self.batch_size} to 4')
        #     self.batch_size = 4

        ## This bit only makes sense for translation experiment - in fact, it should be changed.
        # self.size_object = PARAMS['size_object']
        # self.size_object = None if self.size_object == '0' else self.size_object
        # if self.size_object is not None:
        #     self.size_object = tuple([int(i) for i in self.size_object.split("_")])
        #     list_tags.append('so{}'.format(self.size_object).replace(', ', 'x')) if self.size_object != (50, 50) else None
        # else:
        #     list_tags.append('orsize')
        ###################

        list_tags.append('gap') if self.use_gap else None
        list_tags.append('fe') if self.feature_extraction else None
        list_tags.append('bC') if self.big_canvas else None
        list_tags.append('shFC') if self.shallow_FC else None
        list_tags.append(f'bs{self.batch_size}') if self.batch_size != 32 else None
        super().finalize_init(PARAMS, list_tags)

    def call_run(self, loader, params_to_update, train=True, callbacks=None, epochs=20):
        return run(loader,
                   use_cuda=self.use_cuda,
                   net=self.net,
                   callbacks=callbacks,
                   loss_fn=utils.make_cuda(torch.nn.CrossEntropyLoss(), self.use_cuda),
                   optimizer=torch.optim.Adam(params_to_update,
                                              lr=0.0001 if self.learning_rate is None else self.learning_rate),
                   iteration_step=standard_net_step,
                   iteration_step_kwargs={'train': train},
                   epochs=epochs
                   )

    def get_net(self, new_num_classes=None):
        net, params_to_update = self.prepare_network(network=self.network_name,
                                                     new_num_classes=new_num_classes,
                                                     is_server=self.use_cuda,
                                                     grayscale=self.grayscale,
                                                     use_gap=self.use_gap,
                                                     feature_extraction=self.feature_extraction,
                                                     pretraining=self.pretraining,
                                                     big_canvas=self.big_canvas,
                                                     shallow_FC=self.shallow_FC,
                                                     freeze_fc=self.freeze_fc,
                                                     scramble_fc=self.scramble_fc,
                                                     scramble_conv=self.scramble_conv)
        return net, params_to_update

    @staticmethod
    def get_network_structure(network, pretrain_ImageNet=False):
        if network == 'vgg11_bn':
            network = torchvision.models.vgg11_bn(pretrained=pretrain_ImageNet, progress=True)
        elif network == 'vgg11':
            network = torchvision.models.vgg11(pretrained=pretrain_ImageNet, progress=True)
        elif network == 'vgg16_bn':
            network = torchvision.models.vgg16_bn(pretrained=pretrain_ImageNet, progress=True)
        elif network == 'vgg16':
            network = torchvision.models.vgg16(pretrained=pretrain_ImageNet, progress=True)
        elif network == 'smallCNNnopool':
            if pretrain_ImageNet:
                assert False, f"No pretraining ImageNet for {network}"
            network = smallCNNnp()
        elif network == 'smallCNNpool':
            if pretrain_ImageNet:
                assert False, f"No pretraining ImageNet for {network}"

            network = smallCNNp()
        elif network == 'resnet18':
            network = torchvision.models.resnet18(pretrained=pretrain_ImageNet, progress=True)
        elif network == 'FC4':  # 2500, 2000, 1500, 1000, 500, 10
            if pretrain_ImageNet:
                assert False, f"No pretraining ImageNet for {network}"

            network = FC4()
        elif not isinstance(network, torch.nn.Module):
            assert False, 'network is neither a recognised neural network name, nor a nn.Module'

        return network


    @classmethod
    def change_net_structure(cls, network, type_net, change_to_grayscale=False, new_num_classes=None, shallow_FC=False, use_gap=False, use_big_canvas=False):
        changed_str = ''
        if change_to_grayscale:
            network.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
            changed_str += "\tInput is now in grayscale (1 channel)\n"

        resize_output = True
        if new_num_classes is None:
            new_num_classes == network.classifier[-1].out_features
            resize_output = False
        elif new_num_classes == network.classifier[-1].out_features:
            resize_output = False

        if resize_output is False and (shallow_FC or use_gap or use_big_canvas):
            changed_str += "\tResize Output is False (same number of output units), however (shallow_FC or use_gap or use_big_canvas) is True. The output weights will be lost!\n"
            resize_output = network.classifier[-1].out_features



        if type_net == TypeNet.VGG:

            if use_gap:
                changed_str += '\tPretraing model has a GAP!\n'
                network.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                if shallow_FC:
                    changed_str += '\tPretraing model has a shallow FC!\n'
                    changed_str += f"\tChanged output to {new_num_classes}\n"
                    network.classifier = torch.nn.Linear(512, new_num_classes)
                else:
                    network.classifier[0] = torch.nn.Linear(512, 4096)
            else:
                if shallow_FC:
                    changed_str += '\tPretraing model has a shallow FC!\n'
                    changed_str += f"\tChanged output to {new_num_classes}\n"
                    network.classifier = torch.nn.Linear(512 * 7 * 7, new_num_classes)

                elif resize_output:
                    changed_str += f"\tChanged output to {new_num_classes}\n"
                    network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, new_num_classes)

            if use_big_canvas and not use_gap:
                # if big_canvas, input is 400x400, and to obtain an AdaptiveAvgPool that does not
                # do anything, we need a size of 12, 12
                # ToDo: why not using a bigcanvas of 448 so everything is just double?!
                network.avgpool = torch.nn.AdaptiveAvgPool2d((12, 12))
                changed_str += '\tPretraing model is using big canvas!\n'
                if shallow_FC:
                    changed_str +='\tPretraing model has a shallow FC!\n'
                    changed_str += f"\tChanged output to {new_num_classes}\n"
                    network.classifier = torch.nn.Linear(512 * 12 * 12, new_num_classes)
                else:
                    network.classifier[0] = torch.nn.Linear(512 * 12 * 12, 4096)

        if type_net == TypeNet.SMALL_CNN:
            if resize_output:
                changed_str += f"\tChanged output to {new_num_classes}\n"
                network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, new_num_classes)
                if shallow_FC or use_gap or use_big_canvas:
                    assert False, "Parameters shallow_FC, use_gap, or use_big_canvas not implemented for SMALL_CNN net"

        if type_net == TypeNet.RESNET:
            if resize_output:
                network.fc = torch.nn.Linear(network.fc.in_features, new_num_classes)
                changed_str += f"\tChanged output to {new_num_classes}\n"
                if shallow_FC or use_gap or use_big_canvas:
                    assert False, "Parameters shallow_FC, use_gap, or use_big_canvas not implemented for RESENT net"

        if type_net == TypeNet.FC:
            if resize_output:
                changed_str += f"\tChanged output to {new_num_classes}\n"
                network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, new_num_classes)
                if shallow_FC or use_gap or use_big_canvas:
                    assert False, "Parameters shallow_FC, use_gap, or use_big_canvas not implemented for FC net"

        return network, changed_str

    @classmethod
    def prepare_load(cls, network, pretrain_path, type_net):
        try:
            find_output = int(re.findall('\d+', re.findall(r'o\d+', pretrain_path)[0])[0])
        except:
            print(f'Some problems when checking the output class num for path: {pretrain_path}')
            assert False
        find_gray = 'gray' in pretrain_path
        find_gap = 'gap1' in pretrain_path
        find_sFC = 'sFC1' in pretrain_path
        find_bC = 'bC1' in pretrain_path
        isnp = re.findall(r'nopool', pretrain_path)
        find_NP = bool(isnp[0]) if isnp else False
        network, change_str = cls.change_net_structure(network, type_net, find_gray, find_output, find_sFC, find_gap, find_bC)
        print("Preparing network before loading...")
        if change_str == '':
            print("Nothing has changed")
        else:
            print(change_str)
        return network

    @staticmethod
    def freeze_fully_connected(network):
        for param in network.classifier.parameters():
            param.requires_grad = False
        params_to_update = []
        for param in network.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        return params_to_update

    @staticmethod
    def feature_extraction_method(network, feature_extraction):
        for param in network.features.parameters():
            param.requires_grad = False
        params_to_update = []
        for param in network.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        return params_to_update

    @classmethod
    def prepare_network(cls, network, new_num_classes=None, is_server=False, grayscale=False, use_gap=False, feature_extraction=False, pretraining='vanilla', big_canvas=False, shallow_FC=False, verbose=True, freeze_fc=False, scramble_fc=False, scramble_conv=False):
        """
        @param network: this can be a string such as 'vgg16' or a torch.nn.Module
        @param pretraining: can be [vanilla], [ImageNet] or the path
        @return:
        """

        if isinstance(network, str):
            if 'vgg' in network:
                type_net = TypeNet.VGG
            elif 'FC' in network:
                type_net = TypeNet.FC
            elif 'resnet' in network:
                type_net = TypeNet.RESNET
            elif 'smallCNN' in network:
                type_net = TypeNet.SMALL_CNN
        else:
            type_net = TypeNet.OTHER

        if feature_extraction and pretraining == 'vanilla':
            assert False, 'You selected feature extraction, but you have a vanilla net! Someone gotta train those convolutions man!'

        if pretraining != 'vanilla' and not isinstance(network, str):
            assert False, "With a pretrained network, specify the network structure as a string in 'network'"

        if pretraining != 'vanilla' and big_canvas:
            Warning('If you use a big canvas, you will lose the pretraining on the first FC layers!')

        if grayscale and pretraining != 'vanilla':
            assert False, 'Cannot use pretrained network and grayscale image - you would lose the pretraining on the first conv layer'

        pretrain_path = None
        pretrain_ImageNet = False
        if pretraining != 'vanilla':
            if pretraining == 'ImageNet':
                pretrain_ImageNet = True
            else:
                pretrain_path = pretraining

        network = cls.get_network_structure(network, pretrain_ImageNet=pretrain_ImageNet)

        # Remember that the network structure just created may have a different num_classes than the pretrained state dict we are loading here. If this is the case, call prepare_load and adjust the structure accordingly to match pretrained state dict. Then in model surgery we'll put the num_classes back.
        if pretrain_path is not None:
            network = cls.prepare_load(network, pretrain_path, type_net)
            if verbose:
                print('Loaded model: {}'.format(pretrain_path))
            loaded_state_dict = torch.load(pretrain_path, map_location=torch.device('cuda' if is_server else 'cpu'))
            print('**Loading these parameters from pretrained network:') if verbose else None
            if scramble_fc:
                own_state = network.state_dict()
                if type_net == TypeNet.VGG:
                    for name, param in loaded_state_dict.items():
                        if 'classifier' not in name:
                            print(name) if verbose else None
                            own_state[name].copy_(param)
                    print('Scramble FC is ON: Fully Connected layer NOT copied when loading pretraining params') if verbose else None

                else:
                    assert False, f"Scramble_fc not implemented for [{type_net}]"
            elif scramble_conv:
                own_state = network.state_dict()
                if type_net == TypeNet.VGG:
                    for name, param in loaded_state_dict.items():
                        if 'features' not in name:
                            print(name) if verbose else None
                            own_state[name].copy_(param)
                    print('Scramble conv is ON: Conv. layer NOT copied when loading pretraining params') if verbose else None
                else:
                    assert False, f"Scramble_fc not implemented for [{type_net}]"
            else:
                print('ALL PARAMETERS') if verbose else None
                network.load_state_dict(loaded_state_dict)
        print('***')

        network, change_str = cls.change_net_structure(network, type_net, grayscale, new_num_classes, shallow_FC, use_gap, big_canvas)

        network.train_step = standard_net_step


        if feature_extraction and freeze_fc:
            assert False, 'Both feature extraction and freeze_fc are on - the network won''t learn anything!'
        if feature_extraction and (type_net == TypeNet.VGG or type_net == TypeNet.FC or type_net == TypeNet.RESNET or type_net == TypeNet.SMALL_CNN):
            params_to_update = cls.feature_extraction_method(network, feature_extraction)
        elif freeze_fc and type_net == TypeNet.VGG:
            params_to_update = cls.freeze_fully_connected(network)
        else:
            params_to_update = network.parameters()

        for m in network.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False
        # m.momentum = 0.4

        if verbose:
            print("Params to learn:")
            for name, param in network.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        if is_server:
            network.cuda()

        if verbose:
            print('***Network***')
            print(network)

        print(f"Network structure {'after loading state_dict' if pretraining != 'vanilla' else ''}: ", end="")
        if change_str == '':
            print('Nothing has changed')
        else:
            print("Changed. List of changes:")
            print(change_str)

        return network, params_to_update

    def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe):
        num_classes = self._get_num_classes(testing_loader)

        all_cb = super().prepare_test_callbacks(log_text, testing_loader, save_dataframe)
        all_cb += [TotalAccuracyMetric(use_cuda=self.use_cuda,
                                       to_weblog=self.weblogger, log_text=log_text),
                   ComputeConfMatrix(num_classes=num_classes,
                                     weblogger=self.weblogger,
                                     weblog_text=log_text)]
        all_cb += ([ComputeDataFrame2D(testing_loader.dataset.translation_type_str if isinstance(testing_loader.dataset, TranslateGenerator) else 'no transl',
                                       num_classes,
                                       self.use_cuda,
                                       self.network_name, self.size_canvas,
                                     log_density_weblog=self.weblogger, log_text_plot=log_text)]
                   if save_dataframe else[])
        return all_cb

    def prepare_train_callbacks(self, log_text, train_loader):
        num_classes = self._get_num_classes(train_loader)

        all_cb = super().prepare_train_callbacks(log_text, train_loader)
        if self.weblogger:
            all_cb += [ComputeConfMatrix(num_classes=num_classes,
                                         weblogger=self.weblogger,
                                         weblog_text=log_text,
                                         reset_every=200),
                       RollingAccEachClassWeblog(log_every=self.weblog_check_every,
                                                 num_classes=num_classes,
                                                 weblog_text=log_text,
                                                 weblogger=self.weblogger)]
        return all_cb

class InvarianceSupervisedExp(SupervisedLearningExperiment):
    def __init__(self, **kwargs):
        self.invariance_network_name = None
        self.pretraining_invariance_network = None
        super().__init__(**kwargs)

    def parse_arguments(self, parser):
        super().parse_arguments(parser)
        parser.add_argument("-invn", "--invariance_network_name",
                            help="The network structure used for invariance representation ",
                            type=str,
                            default='relation_net')
        parser.add_argument("-pti", "--pretraining_invariance_network",
                            help="The path for pretraining invariance network",
                            type=str,
                            default='vanilla')
        return parser

    def finalize_init(self, PARAMS, list_tags):
        self.invariance_network_name = PARAMS['invariance_network_name']
        self.pretraining_invariance_network = PARAMS['pretraining_invariance_network']
        super().finalize_init(PARAMS, list_tags)

    def get_net(self, new_num_classes=None):
        device = torch.device('cuda' if self.use_cuda else 'cpu')

        if self.invariance_network_name == 'relation_net':
            inv_net = RelationNetSung(size_canvas=self.size_canvas, grayscale=self.grayscale).backbone
        else:
            assert False, f"Invariance Network Name {self.network_name} not recognized"
        if self.pretraining_invariance_network != 'vanilla':
            if os.path.isfile(self.pretraining_invariance_network):
                print(f"Pretraining value should be a path when used with FewShotLearning (not ImageNet, etc.). Instead is {self.pretraining}")
            inv_net.load_state_dict(torch.load(self.pretraining_invariance_network, map_location=torch.device('cuda' if self.use_cuda else 'cpu')))

        supervised_net = self.prepare_network(network=self.network_name,
                                                     new_num_classes=new_num_classes,
                                                     is_server=self.use_cuda,
                                                     grayscale=self.grayscale,
                                                     use_gap=self.use_gap,
                                                     feature_extraction=self.feature_extraction,
                                                     pretraining=self.pretraining,
                                                     big_canvas=self.big_canvas,
                                                     shallow_FC=self.shallow_FC,
                                                     freeze_fc=self.freeze_fc,
                                                     scramble_fc=self.scramble_fc,
                                                     scramble_conv=self.scramble_conv)


        net = InvarianceSupervisedModel(inv_net, supervised_net)

        # Freeze the invariant network
        for param in inv_net.parameters():
            param.requires_grad = False
        print("Params to learn:")
        for name, param in net.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

        print('***Network***')
        print(net)
        if self.use_cuda:
            net.cuda()

        return net, net.supervised_net.parameters()


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
        self.lossfn = None
        super().__init__(**kwargs)

    # def _get_num_classes(self, loader):
    #     return self.k

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
                            help="Num of objects",
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
        # list_tags.append('Sequential')
        list_tags.append(self.network_name)
        super().finalize_init(PARAMS, list_tags)

    def get_net(self, new_num_classes=None):
        # ToDo: Matching Learning pretrain
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        if self.network_name == 'seqNt1c':
            assert self.nSc == 1 and self.nFc == 1, f"With the model {network_name} you need to set nSc and nFc to 1"
            net = SequenceNtrain1cand(grayscale=self.grayscale)
            self.step = sequence_net_Ntrain_1cand
            self.lossfn = MSELoss()  # CrossEntropyLoss()
        elif self.network_name == 'relation_net':
            assert self.nSc <= 1 and self.nFc <= 1 and self.nSt <= 1 and self.nFt <= 1
            net = RelationNetSung(size_canvas=self.size_canvas, grayscale=self.grayscale)
            self.step = sequence_net_Ntrain_1cand
            self.lossfn = MSELoss()
        else:
            assert False, f"network name {self.network_name} not recognized"
        if self.pretraining != 'vanilla':
            if os.path.isfile(self.pretraining):
                print(f"Pretraining value should be a path when used with FewShotLearning (not ImageNet, etc.). Instead is {self.pretraining}")
            net.load_state_dict(torch.load(self.pretraining, map_location=torch.device('cuda' if self.use_cuda else 'cpu')))

        print('***Network***')
        print(net)

        if self.use_cuda:
            net.cuda()
        return net, net.parameters()

    def call_run(self, data_loader, params_to_update, callbacks=None, train=True, epochs=20):
        return run(data_loader, use_cuda=self.use_cuda, net=self.net,
                   callbacks=callbacks,
                   loss_fn=utils.make_cuda(self.lossfn, self.use_cuda),
                   optimizer=Adam(params_to_update, lr=0.001 if self.learning_rate is None else self.learning_rate),
                   iteration_step=self.step,
                   iteration_step_kwargs={'train': train,
                                          'dataset': data_loader.dataset},
                   epochs=epochs)

    def prepare_train_callbacks(self, log_text, train_loader):
        all_cb = super().prepare_train_callbacks(log_text, train_loader)
        # This ComputeConfMatrix is used for matching, that's why num_class = 2
        all_cb += [ComputeConfMatrix(num_classes=2,
                                     weblogger=self.weblogger,
                                     weblog_text=log_text)]
        return all_cb

    def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe):
        all_cb = super().prepare_test_callbacks(log_text, testing_loader, save_dataframe)
        if save_dataframe:
            all_cb += [ComputeConfMatrix(num_classes=2,
                                         weblogger=self.weblogger,
                                         weblog_text=log_text),
                       ComputeDataFrame3DsequenceLearning(k=testing_loader.dataset.sampler.k,
                                                          nSt=testing_loader.dataset.sampler.nSt,
                                                          nSc=testing_loader.dataset.sampler.nSc,
                                                          nFt=testing_loader.dataset.sampler.nFt,
                                                          nFc=testing_loader.dataset.sampler.nFc,
                                                          task_type=testing_loader.dataset.sampler.place_cameras_mode,
                                                          num_classes=self.k,
                                                          use_cuda=self.use_cuda,
                                                          network_name=self.network_name,
                                                          size_canvas=self.size_canvas,
                                                          log_text_plot=log_text,
                                                          weblogger=self.weblogger,
                                                          output_and_softmax=False)]
        return all_cb



def unity_builder_class(class_obj):
    class UnityExp(class_obj):
        def __init__(self, **kwargs):
            self.name_dataset_training = None
            self.size_canvas = None
            self.play_mode = False
            super().__init__(**kwargs)

        def parse_arguments(self, parser):
            super().parse_arguments(parser)
            parser.add_argument("-name_dataset_training", "--name_dataset_training",
                                help="Select the name of the dataset used for training in Unity - set None for running play mode (and if you do, no testing will be done)",
                                type=str,
                                default=None)
            parser.add_argument("-name_dataset_testing", "--name_dataset_testing",
                                help="Select the name of the dataset used for testing in Unity. It can be more than one. Separate them with _",
                                type=str,
                                default=None)
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
            self.name_dataset_training = PARAMS['name_dataset_training']
            self.name_datasets_testing = PARAMS['name_dataset_testing']
            self.play_mode = PARAMS['play_mode']
            if self.name_datasets_testing is None and self.name_dataset_training is None:
                self.play_mode = True
                self.name_dataset_training = None
            if self.output_filename is None and self.name_datasets_testing is not None:
                assert False, "You provided some dataset for testing, but no output. This is almost always a mistake"

            list_tags.append(f"te{self.name_datasets_testing}") if self.name_datasets_testing is not None else None
            list_tags.append(f"tr{self.name_dataset_training}") if self.name_dataset_training is not None else None

            if self.name_datasets_testing is not None:
                self.name_datasets_testing = str.split(self.name_datasets_testing, "_")
            else:
                self.name_datasets_testing = []

            self.size_canvas = PARAMS['size_canvas_resize']
            # list_tags.append("Unity")
            list_tags.append('sc{}'.format(str(self.size_canvas).replace(', ', 'x'))) if self.size_canvas != '0' else None

            if self.size_canvas == '0':
                self.size_canvas = (128, 128)  # ToDo: this could be taken from the unity channel
            else:
                self.size_canvas = tuple([int(i) for i in self.size_canvas.split("_")])
            super().finalize_init(PARAMS, list_tags)

        def prepare_train_callbacks(self, log_text, train_loader):
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
            all_cb = super().prepare_train_callbacks(log_text, train_loader)
            ck = CheckStoppingLevel()
            all_cb += [PlotUnityImagesEveryOnceInAWhile(dataset=train_loader.dataset,
                                                        grayscale=self.grayscale,
                                                        plot_every=1000,
                                                        plot_only_n_times=20),
                       TriggerActionWithPatience(min_delta=0.01, patience=400, percentage=True, mode='max',
                                                 reaching_goal=90,
                                                 metric_name='webl/mean_acc' if self.weblogger else 'cnsl/mean_acc',
                                                 check_every=self.weblog_check_every if self.weblogger else self.console_check_every,
                                                 triggered_action=ck.go_next_level)]
            return all_cb
        def prepare_test_callbacks(self, log_text, testing_loader, save_dataframe):
            all_cb = super().prepare_test_callbacks(log_text, testing_loader, save_dataframe)
            all_cb += [PlotUnityImagesEveryOnceInAWhile(dataset=testing_loader.dataset,
                                                        grayscale=self.grayscale,
                                                        plot_every=1000,
                                                        plot_only_n_times=5)]
            return all_cb
    return UnityExp


sequence_unity_meta_learning_exp = unity_builder_class(SequentialMetaLearningExp)

