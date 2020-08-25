import torchvision
import cloudpickle
import glob
from enum import Enum
import re
import argparse
from torch.optim import Adam
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Union
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss, NLLLoss

from callbacks import *
import framework_utils as utils
from models.meta_learning_models import MatchingNetwork, MatchingNetPlus, RelationNetSung, get_few_shot_encoder, get_few_shot_encoder_basic, get_few_shot_evaluator
from models.FCnets import FC4
from train_net import *


class Experiment(ABC):
    def __init__(self, name_experiment='default_name', parser=None, ):
        parser = utils.parse_experiment_arguments(parser)
        PARAMS = vars(parser.parse_known_args()[0])

        self.current_run = -1
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
        self.force_cuda = PARAMS['force_cuda']
        self.additional_tags = PARAMS['additional_tags']
        self.size_object = PARAMS['size_object']
        self.use_neptune = PARAMS['use_neptune']
        self.size_object = None if self.size_object == '0' else self.size_object
        self.experiment_data = {}
        self.experiment_loaders = {}  # we separate data from loaders because loaders are pickled objects and may broke when module name changes. If this happens, at least we preserve the data. We generally don't even need the loaders much.


        self.use_cuda = self.force_cuda
        if torch.cuda.is_available():
            print('Using cuda - you are probably on the server')
            self.use_cuda = True

        list_tags = [name_experiment] if name_experiment != '' else []
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
        self.experiment_loaders[self.current_run] = {'training': [],
                                                     'testing': []}
        print('Run Number {}'.format(self.current_run))

    def finalize_init(self, PARAMS, list_tags):
        print(PARAMS)
        self.initialize_neptune(PARAMS, list_tags) if self.use_neptune else None
        self.new_run()

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
            utils.neptune_log_dataset_info(train_loader, log_text=train_loader.dataset.name_generator)
        if test_loaders_list is not None:
            for loader in test_loaders_list:
                utils.neptune_log_dataset_info(loader, log_text=loader.dataset.name_generator)

    @abstractmethod
    def call_run(self, train_loader, net, params_to_update, callbacks=None, train=True, epochs=20):
        pass

    @abstractmethod
    def get_net(self, network_name, num_classes, pretraining, grayscale):
        return None, None

    def prepare_train_callbacks(self, net, log_text, num_classes):
        nept_check_every = 5
        console_check_every = 100
        all_cb = [StandardMetrics(log_every=console_check_every, print_it=True,
                                  use_cuda=self.use_cuda,
                                  to_neptune=False, log_text=log_text,
                                  metrics_prefix='cnsl'),
                  EarlyStopping(min_delta=0.01, patience=150, percentage=True, mode='max',
                                reaching_goal=self.stop_when_train_acc_is,
                                metric_name='nept/mean_acc' if self.use_neptune else 'cnsl/mean_acc',
                                check_every=nept_check_every if self.use_neptune
                                else console_check_every),
                  EarlyStopping(min_delta=0.01, patience=300, percentage=True, mode='min',
                                reaching_goal=None,
                                metric_name='nept/mean_loss' if self.use_neptune else 'cnsl/mean_loss',
                                check_every=nept_check_every if self.use_neptune
                                else console_check_every),  # for stagnation
                  StopWhenMetricIs(value_to_reach=self.max_iterations, metric_name='tot_iter'),
                  TotalAccuracyMetric(use_cuda=self.use_cuda,
                                      to_neptune=self.use_neptune, log_text=log_text),
                  ComputeConfMatrix(num_classes=num_classes,
                                    send_to_neptune=self.use_neptune,
                                    neptune_text=log_text,
                                    reset_every=200),

                  StopFromUserInput(),
                  PlotTimeElapsed(time_every=100)]

        all_cb += ([SaveModel(net, self.model_output_filename, self.use_neptune)] if self.model_output_filename is not None else [])
        if self.use_neptune:
            all_cb += [StandardMetrics(log_every=nept_check_every, print_it=False,
                                       use_cuda=self.use_cuda,
                                       to_neptune=True, log_text=log_text,
                                       metrics_prefix='nept'),
                       RollingAccEachClassNeptune(log_every=nept_check_every,
                                                  num_classes=num_classes,
                                                  neptune_text=log_text),
                       PlotGradientNeptune(net=net, log_every=50, plot_every=500, log_txt=log_text),
                       ]

        return all_cb

    def _get_num_classes(self, loader):
        return loader.dataset.num_classes

    def train(self, train_loader, callbacks=None, log_text='train'):
        self.experiment_loaders[self.current_run]['training'].append(train_loader)
        net, params_to_update = self.get_net(self.network_name,
                                             num_classes=self._get_num_classes(train_loader),
                                             pretraining=self.pretraining,
                                             grayscale=train_loader.dataset.grayscale)
        all_cb = self.prepare_train_callbacks(net, log_text, self._get_num_classes(train_loader))

        all_cb += (callbacks or [])

        net, logs = self.call_run(train_loader, net, params_to_update, train=True, callbacks=all_cb)
        return net

    def finalize_test(self, df_testing, conf_mat, accuracy):
        self.experiment_data[self.current_run]['df_testing'] = df_testing
        self.experiment_data[self.current_run]['conf_mat_acc'] = conf_mat
        self.experiment_data[self.current_run]['accuracy'] = accuracy

    def prepare_test_callbacks(self, num_classes, log_text, translation_type_str, save_dataframe):
        all_cb = [StopWhenMetricIs(value_to_reach=self.num_iterations_testing, metric_name='tot_iter'),
                  TotalAccuracyMetric(use_cuda=self.use_cuda,
                                      to_neptune=self.use_neptune, log_text=log_text),
                  ComputeConfMatrix(num_classes=num_classes,
                                    send_to_neptune=self.use_neptune,
                                    neptune_text=log_text),
                  ]
        size_canvas = self.size_canvas if hasattr(self, 'size_canvas') else (224, 224)
        all_cb += ([ComputeDataframe(num_classes,
                                     self.use_cuda,
                                     translation_type_str,
                                     self.network_name, size_canvas,
                                     log_density_neptune=True if self.use_neptune else False, log_text_plot=log_text)]
                   if save_dataframe else [])
        return all_cb

    def test(self, net, test_loaders_list, callbacks=None, log_text: List[str] = None):
        if log_text is None:
            log_text = [d.dataset.name_generator for d in test_loaders_list]
        self.experiment_loaders[self.current_run]['testing'].append(test_loaders_list)
        save_dataframe = True if self.output_filename is not None else False

        conf_mat_acc_all_tests = []
        accuracy_all_tests = []

        print('*TESTS')
        df_testing = pd.DataFrame([])
        for idx, testing_loader in enumerate(test_loaders_list):
            print('Testing on [{}], [{}]'.format(testing_loader.dataset.name_generator, testing_loader.dataset.translation_type_str))
            all_cb = self.prepare_test_callbacks(self._get_num_classes(testing_loader), log_text[idx] if log_text is not None else '', testing_loader.dataset.translation_type_str, save_dataframe)
            all_cb += (callbacks or [])

            net, logs = self.call_run(testing_loader, net,
                                      params_to_update=net.parameters(), train=False,
                                      callbacks=all_cb,
                                      epochs=1)
            conf_mat_acc_all_tests.append(logs['conf_mat_acc'])
            accuracy_all_tests.append(logs['total_accuracy'])
            if save_dataframe:
                df_testing = pd.concat((df_testing, logs['dataframe']))

        self.finalize_test(df_testing, conf_mat_acc_all_tests, accuracy_all_tests)

        return df_testing, conf_mat_acc_all_tests, accuracy_all_tests

    def save_all_runs(self):
        if self.output_filename is not None:
            result_path_data = self.output_filename
            result_path_loaders = os.path.dirname(result_path_data) + '/loaders_' + os.path.basename(result_path_data)
            pathlib.Path(os.path.dirname(result_path_data)).mkdir(parents=True, exist_ok=True)
            cloudpickle.dump(self.experiment_data, open(result_path_data, 'wb'))
            cloudpickle.dump(self.experiment_loaders, open(result_path_loaders, 'wb'))
            print('Saved data in {}, \nSaved loaders in {}'.format(result_path_data, result_path_loaders))
            # if self.use_neptune:
            #     neptune.log_artifact(result_path_data, result_path_data)
            #     neptune.log_artifact(result_path_loaders, result_path_loaders)
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

    def call_run(self, loader, net, params_to_update, train=True, callbacks=None, epochs=20):
        return run(loader,
                   use_cuda=self.use_cuda,
                   net=net,
                   callbacks=callbacks,
                   loss_fn=utils.make_cuda(torch.nn.CrossEntropyLoss(), self.use_cuda),
                   optimizer=torch.optim.Adam(params_to_update,
                                              lr=0.0001 if self.learning_rate is None else self.learning_rate),
                   iteration_step=standard_net_step,
                   iteration_step_kwargs={'train': train},
                   epochs=epochs
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
        find_gap = 'gap1' in pretrain_path
        find_sFC = 'sFC1' in pretrain_path
        find_bC = 'bC1' in pretrain_path
        # Grayscale not implemented: grayscale is assumed to be FALSE at all time
        if type_net == TypeNet.VGG:
            if find_sFC:
                print('**Pretraing model has a shallow FC!')
                network.classifier = torch.nn.Linear(512 * 7 * 7, num_classes)
            else:
                network.classifier[-1] = torch.nn.Linear(network.classifier[-1].in_features, find_ouptut)
            if find_gap:
                print('**Pretraing model has a GAP!')
                network.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                if find_sFC:
                    print('**Pretraing model has a shallow FC!')
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


class FewShotLearningExp(Experiment):
    '''
    Select network_name = 'matching_net_basics' for using the basic network that works for any image size, used in the paper for Omniglot.
                        'matching_net_more' to use a more complex version of the matching net, with more conv layer. It can still accept any type of input
                        'matching_net_plus' the decision is made by an evaluation network. It accepts 128x128px images.
                        #ToDo: with a adaptive average pool make it accept whatever size
    '''
    def __init__(self, name_experiment):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser = utils.parse_few_shot_learning_parameters(parser)
        self.training_folder = None
        self.testing_folder = None
        self.n, self.k, self.q, self.size_canvas = 0, 0, 0, 0
        self.step = None
        self.lossfn = None
        super().__init__(name_experiment=name_experiment, parser=parser)

    def finalize_init(self, PARAMS, list_tags):
        self.training_folder = PARAMS['folder_for_training']
        self.testing_folder = PARAMS['folder_for_testing']
        self.n = PARAMS['n_shot']
        self.k = PARAMS['k_way']
        self.q = PARAMS['q_queries']
        self.size_canvas = PARAMS['size_canvas']
        list_tags.append(f'{self.k}w_{self.n}s_{self.q}q')
        list_tags.append(f'sc{self.size_canvas}')
        super().finalize_init(PARAMS, list_tags)

    def get_net(self, network_name, num_classes, pretraining, grayscale=False):
        # ToDo: Matching Learning pretrain
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        if network_name == 'matching_net_basic':
            net = MatchingNetwork(self.n, self.k, self.q, fce=False,
                                  num_input_channels=1 if grayscale else 3,
                                  lstm_layers=0,
                                  lstm_input_size=0,
                                  unrolling_steps=0,
                                  device=device,
                                  encoder=get_few_shot_encoder_basic)
            self.step = matching_net_step
            self.lossfn = NLLLoss()
        if network_name == 'matching_net_more':
            net = MatchingNetwork(self.n, self.k, self.q, fce=False,
                                  num_input_channels=1 if grayscale else 3,
                                  lstm_layers=0,
                                  lstm_input_size=0,
                                  unrolling_steps=0,
                                  device=device,
                                  encoder=get_few_shot_encoder)
            self.step = matching_net_step
            self.lossfn = torch.nn.NLLLoss()

        if network_name == 'matching_net_plus':
            net = MatchingNetPlus(self.n, self.k, self.q,
                                  num_input_channels=1 if grayscale else 3)
            self.step = matching_net_step_plus
            self.lossfn = CrossEntropyLoss()

        if network_name == 'relation_net':
            net = RelationNetSung(size_canvas=self.size_canvas)
            self.step = relation_net_step
            self.lossfn = MSELoss()
        print('***Network***')
        print(net)
        return net, net.parameters()

    def call_run(self, data_loader, net, params_to_update, callbacks=None, train=True, epochs=20):
        if data_loader.dataset.num_classes < self.k:
            warnings.warn(f'Experiment: Number of classes in the folder {data_loader.dataset.folder} < k ({self.k}. K will be changed to {data_loader.dataset.num_classes}')
            self.k = data_loader.dataset.num_classes

        return run(data_loader, use_cuda=self.use_cuda, net=net,
                   callbacks=callbacks,
                   loss_fn=utils.make_cuda(self.lossfn, self.use_cuda),
                   optimizer=Adam(params_to_update, lr=0.001 if self.learning_rate is None else self.learning_rate),
                   iteration_step=self.step,
                   iteration_step_kwargs={'train': train, 'n_shot': self.n, 'k_way': self.k, 'q_queries': self.q},
                   epochs=epochs)

    def prepare_train_callbacks(self, net, log_text, num_classes):
        all_cb = super().prepare_train_callbacks(net, log_text, num_classes)
        idx_ccm = [idx for idx, i in enumerate(all_cb) if isinstance(i, ComputeConfMatrix)]
        assert len(idx_ccm) == 1
        all_cb[idx_ccm[0]] = CompConfMatrixFewShot(num_classes=num_classes,
                                                   send_to_neptune=self.use_neptune,
                                                   neptune_text=log_text,
                                                   reset_every=200)
        return all_cb

    def prepare_test_callbacks(self, num_classes, log_text, translation_type_str, save_dataframe):
        all_cb = super().prepare_test_callbacks(num_classes, log_text, translation_type_str, save_dataframe)
        if save_dataframe:
            idx_cd = [idx for idx, i in enumerate(all_cb) if isinstance(i, ComputeDataframe)]
            assert len(idx_cd) <= 1
            all_cb[idx_cd[0]] = ComputeDataframe(self.k,
                                                self.use_cuda,
                                                translation_type_str,
                                                self.network_name, self.size_canvas,
                                                log_density_neptune=True if self.use_neptune else False,
                                                log_text_plot=log_text,
                                                output_and_softmax=False)

        idx_ccm = [idx for idx, i in enumerate(all_cb) if isinstance(i, ComputeConfMatrix)]
        assert len(idx_ccm) == 1
        all_cb[idx_ccm[0]] = CompConfMatrixFewShot(num_classes=num_classes,
                                                   send_to_neptune=self.use_neptune,
                                                   neptune_text=log_text,
                                                   reset_every=None)

        return all_cb
