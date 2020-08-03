from train_net import standard_net_step
import torchvision
import cloudpickle
import glob
from models.FCnets import FC4
from enum import Enum
import re
from callbacks import *
import framework_utils as utils
from models.meta_learning_models import MatchingNetwork
from train_net import matching_net_step, run
import argparse
from torch.optim import Adam
from abc import ABC, abstractmethod


class Experiment(ABC):
    def __init__(self, name_experiment='default_name', parser=None, ):
        parser = utils.parse_experiment_arguments(parser)
        PARAMS = vars(parser.parse_known_args()[0])

        self.current_run = 0
        self.name_experiment = name_experiment
        self.num_runs = PARAMS['num_runs']  # this is not used here, and it shouldn't be here, but for now we are creating a neptune session when an Experiment is created, and we want to save this as a parameter, so we need to do it here.
        self.pretraining = PARAMS['pretraining']
        self.stop_when_train_acc_is = PARAMS['stop_when_train_acc_is']
        self.max_iterations = PARAMS['max_iterations']
        self.model_output_filename = PARAMS['model_output_filename']
        self.output_filename = PARAMS['output_filename']
        self.num_iterations_testing = PARAMS['num_iterations_testing']
        self.network_name = PARAMS['network_name']
        self.learning_rate = PARAMS['learning_rate']
        self.force_server = PARAMS['force_server']
        self.additional_tags = PARAMS['additional_tags']
        self.size_object = PARAMS['size_object']
        self.use_neptune = PARAMS['use_neptune']
        self.size_object = None if self.size_object == '0' else self.size_object
        self.experiment_data = {self.current_run: {'training_loaders': [],
                                                   'testing_loaders': []}}

        self.use_cuda = self.force_server
        if torch.cuda.is_available():
            print('Using cuda - you are probably on the server')
            self.use_cuda = True

        list_tags = [name_experiment]
        if self.additional_tags is not None:
            [list_tags.append(i) for i in self.additional_tags.split('_')]
        list_tags.append('van') if self.pretraining == 'vanilla' else None
        list_tags.append('imgn') if self.pretraining == 'ImageNet' else None
        list_tags.append(self.network_name)
        list_tags.append('lr{}'.format("{:2f}".format(self.learning_rate).split(".")[1])) if self.learning_rate is not None else None
        if self.size_object is not None:
            self.size_object = tuple([int(i) for i in self.size_object.split("_")])
            list_tags.append('so{}'.format(self.size_object)) if self.size_object != (50, 50) else None
        else:
            list_tags.append('orsize')

        self.batch_size = 32 if self.use_cuda else 4
        if self.max_iterations is None:
            self.max_iterations = 5000 if self.use_cuda else 10

        self.finalize_init(PARAMS, list_tags)

    def new_run(self):
        self.current_run += 1
        self.experiment_data[self.current_run] = {}
        self.experiment_data[self.current_run] = {'training_loaders': [],
                                                  'testing_loaders': []}
        print('Run Number {}'.format(self.current_run))

    def finalize_init(self, PARAMS, list_tags):
        print(PARAMS)
        self.initialize_neptune(PARAMS, list_tags) if self.use_neptune else None

    @staticmethod
    def initialize_neptune(PARAMS, list_tags):
        neptune.init('valeriobiscione/valerioERC')
        try:
            neptune.create_experiment(name='',
                                      params=PARAMS,
                                      tags=list_tags)
        except BaseException as e:
            print(e)

    @staticmethod
    def neptune_plot_generators_info(train_loader=None, test_loaders_list=None):
        if train_loader is not None:
            utils.neptune_log_dataset_info(train_loader, log_text='training')
        if test_loaders_list is not None:
            for loader in test_loaders_list:
                utils.neptune_log_dataset_info(loader, log_text='testing')

    @abstractmethod
    def call_train_net(self, train_loader, net, params_to_update, callbacks=None):
        pass

    @abstractmethod
    def get_net(self, network_name, num_classes, pretraining, grayscale):
        return None, None

    def train(self, train_loader, callbacks=None, log_text='train'):
        self.experiment_data[self.current_run]['training_loaders'].append(train_loader)
        net, params_to_update = self.get_net(self.network_name,
                                             num_classes=train_loader.dataset.num_classes,
                                             pretraining=self.pretraining,
                                             grayscale=train_loader.dataset.grayscale)

        nept_log = 5
        std_log = 100

        all_callbacks = [StandardMetrics(log_every=std_log, verbose=True, use_cuda=self.use_cuda),
                              EarlyStopping(min_delta=0.01, patience=150, percentage=True, mode='max',
                                            reaching_goal=self.stop_when_train_acc_is,
                                            metric_name='nept/mean_acc' if self.use_neptune else 'std/mean_acc',
                                            check_every=nept_log if self.use_neptune else std_log),
                              SaveModel(net, self.model_output_filename),
                              StopWhenMetricIs(value_to_reach=self.max_iterations, metric_name='tot_iter'),
                              MetricsNeptune(neptune_log_text=log_text, log_every=nept_log,
                                             use_cuda=self.use_cuda) if self.use_neptune else []]
        all_callbacks += (callbacks or [])

        net = self.call_train_net(train_loader, net, params_to_update, callbacks=all_callbacks)
        return net

    def finalize_test(self, df_testing, conf_mat, accuracy):
        self.experiment_data[self.current_run]['df_testing'] = df_testing
        self.experiment_data[self.current_run]['conf_mat_acc'] = conf_mat
        self.experiment_data[self.current_run]['accuracy'] = accuracy

    def test(self, net, test_loaders_list, log_text=''):
        self.experiment_data[self.current_run]['testing_loaders'].append(test_loaders_list)

        save_dataframe = True if self.output_filename is not None else False
        df_testing, conf_mat, accuracy = utils.test_loader_and_save(test_loaders_list, save_dataframe, net, self.use_cuda, self.network_name, self.num_iterations_testing, log_text=log_text)
        self.finalize_test(df_testing, conf_mat, accuracy)

    def save_all_runs(self):
        if self.output_filename is not None:
            result_path = self.output_filename
            pathlib.Path(os.path.dirname(result_path)).mkdir(parents=True, exist_ok=True)
            cloudpickle.dump(self.experiment_data, open(result_path, 'wb'))
            print('Saved in {}'.format(result_path))
            neptune.log_artifact(result_path, result_path)
        else:
            Warning('Results path is not specified!')


class TypeNet(Enum):
    VGG = 0
    FC = 1
    RESNET = 2
    OTHER = 3


class StandardTrainingExperiment(Experiment):
    def __init__(self, name_experiment='standard_training', parser=None):
        parser = utils.parse_standard_training_arguments(parser)
        self.use_gap, self.feature_extraction, self.big_canvas, self.shallow_FC = False, False, False, False
        self.size_canvas = 0
        super().__init__(name_experiment=name_experiment, parser=parser)

    def finalize_init(self, PARAMS, list_tags):
        self.use_gap = PARAMS['use_gap']
        self.feature_extraction = PARAMS['feature_extraction']
        self.big_canvas = PARAMS['big_canvas']
        self.shallow_FC = PARAMS['shallow_FC']
        self.size_canvas = (224, 224) if not self.big_canvas else (400, 400)
        list_tags.append('gap') if self.use_gap else None
        list_tags.append('fe') if self.feature_extraction else None
        list_tags.append('bC') if self.big_canvas else None
        list_tags.append('shFC') if self.shallow_FC else None
        super().finalize_init(PARAMS, list_tags)

    def call_train_net(self, train_loader, net, params_to_update, callbacks=None):
        return run(train_loader,
                   use_cuda=self.use_cuda,
                   net=net,
                   callbacks=callbacks,
                   loss_fn=utils.make_cuda(torch.nn.CrossEntropyLoss(), self.use_cuda),
                   optimizer=torch.optim.Adam(params_to_update,
                                              lr=0.0001 if self.learning_rate is None else self.learning_rate),
                   training_step=standard_net_step,
                   training_step_kwargs={'train': True}
                   )

    def get_net(self, network_name, num_classes, pretraining, grayscale=False):
        net, params_to_update = self.prepare_network(network_name,
                                                     num_classes=num_classes,
                                                     is_server=self.use_cuda,
                                                     grayscale=grayscale,
                                                     use_gap=self.use_gap,
                                                     feature_extraction=self.feature_extraction,
                                                     pretraining=pretraining,
                                                     big_canvas=self.big_canvas,
                                                     shallow_FC=self.shallow_FC)
        return net, params_to_update

    @staticmethod
    def get_network_structure(network, num_classes):
        if network == 'vgg11_bn':
            network = torchvision.models.vgg11_bn(pretrained=False, progress=True, num_classes=num_classes)
        elif network == 'vgg11':
            network = torchvision.models.vgg11(pretrained=False, progress=True, num_classes=num_classes)
        elif network == 'vgg16_bn':
            network = torchvision.models.vgg16_bn(pretrained=False, progress=True, num_classes=num_classes)
        elif network == 'vgg16':
            network = torchvision.models.vgg16(pretrained=False, progress=True, num_classes=num_classes)
        elif network == 'resnet18':
            network = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=num_classes)
        elif network == 'FC4':  # 2500, 2000, 1500, 1000, 500, 10
            network = FC4(num_classes=num_classes)
        elif not isinstance(network, torch.nn.Module):
            assert False, 'network is neither a recognised neural network name, nor a nn.Module'

        return network

    @staticmethod
    def prepare_load(network, num_classes, pretrain_path, type_net):
        find_ouptut = int(re.findall('\d+', re.findall(r'o\d+', pretrain_path)[0])[0])
        find_gap = 'gap' in pretrain_path
        find_sFC = 'sFC' in pretrain_path
        find_bC = 'bC' in pretrain_path
        # Grayscale not implemented: grayscale is assumed to be FALSE at all time
        if type_net == TypeNet.VGG:
            if find_sFC:
                network.classifier = torch.nn.Linear(512 * 7 * 7, num_classes)
            else:
                network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, find_ouptut)
            if find_gap:
                network.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                if find_sFC:
                    network.classifier = torch.nn.Linear(512, num_classes)
                else:
                    network.classifier[0] = torch.nn.Linear(512, 4096)
            if find_bC and not find_gap:
                # if big_canvas, input is 400x400, and to obtain an AdaptiveAvgPool that does not
                # do anything, we need a size of 12, 12
                network.avgpool = torch.nn.AdaptiveAvgPool2d((12, 12))
                if find_sFC:
                    network.classifier = torch.nn.Linear(512 * 12 * 12, num_classes)
                else:
                    network.classifier[0] = torch.nn.Linear(512 * 12 * 12, 4096)

        if type_net == TypeNet.RESNET:
            network.fc = torch.nn.Linear(network.fc.in_features, find_ouptut)
        if type_net == TypeNet.FC:
            network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, find_ouptut)

        return network

    @staticmethod
    def vgg_model_surgery(network, grayscale, num_classes, shallow_FC, use_gap, big_canvas):
        if grayscale:
            network.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)

        if shallow_FC:
            if isinstance(network.classifier, torch.nn.Sequential):  # if the classifier is a list, it means we need to change this
                network.classifier = torch.nn.Linear(512 * 7 * 7, num_classes)
        else:
            if network.classifier[6].out_features != num_classes:
                print('Head of network switched: it was {} classes, now it''s {}'.format(network.classifier[6].out_features, num_classes))
                network.classifier[6] = torch.nn.Linear(4096, num_classes)

        if use_gap:
            network.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if shallow_FC:
                if isinstance(network.classifier, torch.nn.Sequential):
                    network.classifier = torch.nn.Linear(512, num_classes)
            else:
                network.classifier[0] = torch.nn.Linear(512, 4096)

        if big_canvas and not use_gap:
            # if big_canvas, input is 400x400, and to obtain an AdaptiveAvgPool that does not
            # do anything, we need a size of 12, 12
            network.avgpool = torch.nn.AdaptiveAvgPool2d((12, 12))
            if shallow_FC:
                if isinstance(network.classifier, torch.nn.Sequential):
                    network.classifier = torch.nn.Linear(512 * 12 * 12, num_classes)
            else:
                network.classifier[0] = torch.nn.Linear(512 * 12 * 12, 4096)
        return network

    @staticmethod
    def feature_extraction_method(network, feature_extraction):
        params_to_update = network.parameters()
        print("Params to learn:")
        if feature_extraction:
            for param in network.features.parameters():
                param.requires_grad = False
            params_to_update = []
            for param in network.parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        return params_to_update

    @classmethod
    def prepare_network(cls, network, num_classes, is_server=False, grayscale=False, use_gap=False, feature_extraction=False, pretraining='vanilla', big_canvas=False, shallow_FC=False):
        """
        @param network: this can be a string such as 'vgg16' or a torch.nn.Module
        @param pretraining: can be [vanilla], [ImageNet]
        @return:
        """

        if isinstance(network, str):
            if 'vgg' in network:
                type_net = TypeNet.VGG
            elif 'FC' in network:
                type_net = TypeNet.FC
            elif 'resnet' in network:
                type_net = TypeNet.RESNET
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
        if pretraining != 'vanilla':
            if pretraining == 'ImageNet':
                pretrain_path = glob.glob('./models/ImageNet_{}*.pickle'.format(network))
                assert len(pretrain_path) <= 1, 'Found multiple matches for the pretraining network ImageNet {}'.format(network)
                pretrain_path = pretrain_path[0]
            else:
                pretrain_path = pretraining

        network = cls.get_network_structure(network, num_classes)

        # Remember that the network structure just created may have a different num_classes than the pretrained state dict we are loading here. If this is the case, call prepare_load and adjust the structure accordingly to match pretrained state dict. Then in model surgery we'll put the num_classes back.
        if pretrain_path is not None:
            network = cls.prepare_load(network, num_classes, pretrain_path, type_net)
            print('Loaded model: {}'.format(pretrain_path))
            network.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cuda' if is_server else 'cpu')))

        if type_net == TypeNet.VGG:
            cls.vgg_model_surgery(network, grayscale, num_classes, shallow_FC, use_gap, big_canvas)
            network.train_step = standard_net_step

        if type_net == TypeNet.FC:
            if network.classifier[-1].out_features != num_classes:
                network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, num_classes)
                network.train_step = standard_net_step

        if type_net == TypeNet.RESNET:
            if network.classifier[-1].out_features != num_classes:
                network.fc = torch.nn.Linear(network.fc.in_features, num_classes)
                network.train_step = standard_net_step

        if type_net == TypeNet.VGG or type_net == TypeNet.FC or type_net == TypeNet.RESNET:
            params_to_update = cls.feature_extraction_method(network, feature_extraction)

        for m in network.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False
        # m.momentum = 0.4

        for name, param in network.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

        if is_server:
            network.cuda()

        print('***Network***')
        print(network)
        return network, params_to_update


class MatchinNetExp(Experiment):
    def __init__(self, name_experiment):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser = utils.parse_experiment_arguments(parser)
        self.folder = None
        self.n, self.k, self.q, self.canvas_size = 0, 0, 0, 0
        super().__init__(name_experiment=name_experiment, parser=parser)

    def finalize_init(self, PARAMS, list_tags):
        self.folder = PARAMS['folder_for_training']
        self.n = PARAMS['n_shot']
        self.k = PARAMS['k_way']
        self.q = PARAMS['q_queries']
        self.canvas_size = PARAMS['canvas_size']
        list_tags.append(f'{self.k}w_{self.n}s_{self.q}q')
        list_tags.append(f'cs{self.canvas_size}')
        super().finalize_init(PARAMS, list_tags)

    def get_net(self, network_name, num_classes, pretraining, grayscale=False):
        # ToDo: Matching Learning pretrain
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        net = MatchingNetwork(self.n, self.k, self.q, fce=False,
                              num_input_channels=1 if grayscale else 3,
                              lstm_layers=0,
                              lstm_input_size=0,
                              unrolling_steps=0,
                              device=device)

        print('***Network***')
        print(net)
        return net, net.parameters()

    def call_train_net(self, train_loader, net, params_to_update, optimizer=None, callbacks=None):
        return run(train_loader, use_cuda=self.use_cuda, net=net,
                   callbacks=callbacks,
                   loss_fn=utils.make_cuda(torch.nn.NLLLoss(), self.use_cuda),
                   optimizer=Adam(params_to_update, lr=0.001 if self.learning_rate is None else self.learning_rate),
                   training_step=matching_net_step,
                   training_step_kwargs={'train': True, 'n_shot': self.n, 'k_way': self.k, 'q_queries': self.q})
