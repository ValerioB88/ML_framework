import torch
from framework_utils import make_cuda
import sty
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from tqdm import tqdm
from typing import List


class RecordActivations:
    def __init__(self, net, use_cuda=None, only_save: List[str] = None, detach_tensors=True):
        if only_save is None:
            self.only_save = ['Conv2d', 'Linear']
        else:
            self.only_save = only_save
        self.cuda = False
        if use_cuda is None:
            if torch.cuda.is_available():
                self.cuda = True
            else:
                self.cuda = False
        else:
            self.cuda = use_cuda
        self.net = net
        self.detach_tensors = detach_tensors
        self.activation = {}
        self.last_linear_layer = ''
        self.all_layers_name = []
        self.setup_network()


    def setup_network(self):
        self.was_train = self.net.training
        self.net.eval()  # a bit dangerous
        print(sty.fg.yellow + "Network put in eval mode in Record Activation" + sty.rs.fg)
        all_layers = self.group_all_layers()
        self.hook_lists = []
        for idx, i in enumerate(all_layers):
            name = '{}: {}'.format(idx, str.split(str(i), '(')[0])
            if np.any([ii in name for ii in self.only_save]):
                self.all_layers_name.append(name)
                self.hook_lists.append(i.register_forward_hook(self.get_activation(name)))
        self.last_linear_layer = self.all_layers_name[-1]

    def get_activation(self, name):
        def hook(model, input, output):
                self.activation[name] = (output.detach() if self.detach_tensors else output)
        return hook

    def group_all_layers(self):
        all_layers = []

        def recursive_group(net):
            for layer in net.children():
                if not list(layer.children()):  # if leaf node, add it to list
                    all_layers.append(layer)
                else:
                    recursive_group(layer)

        recursive_group(self.net)
        return all_layers

    def remove_hooks(self):
        for h in self.hook_lists:
            h.remove()
        if self.was_train:
            self.net.train()


class DistanceActivation(RecordActivations, ABC):
    class CompareWith(Enum):
        ANY_OBJECT = 0
        SAME_OBJECT = 1
        SAME_CLASS_DIFF_OBJ = 2
        DIFF_CLASS = 3

    def __init__(self, dataset=None, distance='cossim', compare_with=CompareWith.SAME_OBJECT, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance
        self.dataset = dataset
        self.compare_with = compare_with

    @abstractmethod
    def get_base_and_other_canvasses(self, class_num, name_class, **kwargs):
        raise NotImplementedError

    def finalize_each_class(self, name_class, cossim, cossim_imgs,  x_values):
        """
        Use for plotting, analysis, etc.
        """
        return name_class, cossim, cossim_imgs,  x_values

    def compute_distance(self, a, b):
        if self.distance == 'cossim':
            return torch.nn.CosineSimilarity(dim=0)(a.flatten(), b.flatten()).item()
        if self.distance == 'euclidean':
            diff =a.flatten() - b.flatten()
            return torch.sqrt(torch.dot(diff, diff)).item()

    def get_cosine_similarity_from_images(self, base_canvas, other_canvasses, **kwargs):
        distance_net = {}
        predictions = []
        prediction_base = torch.argmax(self.net([base_canvas.unsqueeze(0)])[0]).item()
        base_activation = {}
        for name, features in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            base_activation[name] = features

        for idx, canvas_comparison in enumerate(other_canvasses):
            canvas_comparison_activation = {}
            self.net([canvas_comparison.unsqueeze(0)])
            for name, features in self.activation.items():
                if not np.any([i in name for i in self.only_save]):
                    continue
                canvas_comparison_activation[name] = features
                if name not in distance_net:
                    distance_net[name] = []
                distance_net[name].append(self.compute_distance(base_activation[name], canvas_comparison_activation[name]))

        if isinstance(base_canvas, list):
            distance_image = [self.compute_distance(b, c) for b, c in zip(base_canvas, other_canvasses)]
        else:
            distance_image = [self.compute_distance(base_canvas, c) for c in other_canvasses]
        return distance_net, distance_image, (prediction_base, predictions)

    @staticmethod
    def get_average_cossim_across_classes_and_values(cossim_net):
        all_layers = list(cossim_net[0].keys())

        # global cossim across all values for each layer
        mean = [np.mean(np.array([v[l] for k, v in cossim_net.items()])) for l in all_layers]
        std = [np.std(np.array([v[l] for k, v in cossim_net.items()])) for l in all_layers]
        return mean, std, all_layers

    @staticmethod
    def get_average_cossim_across_classes(cossim_net):
        if isinstance(cossim_net, list):
            all_layers = list(cossim_net[0].keys())
            mean = [np.mean(np.array([v[l] for v in cossim_net]), axis=0) for l in all_layers]
            std = [np.std(np.array([v[l] for v in cossim_net]), axis=0) for l in all_layers]
            return mean, std, all_layers
        if isinstance(cossim_net[0], dict):
            all_layers = list(cossim_net[0].keys())
            mean = [np.mean(np.array([v[l] for k, v in cossim_net.items()]), axis=0) for l in all_layers]
            std = [np.std(np.array([v[l] for k, v in cossim_net.items()]), axis=0) for l in all_layers]
            return mean, std, all_layers
        elif isinstance(cossim_net, dict):
            mean = np.mean(np.array([v for k, v in cossim_net.items()]), axis=0)
            std = np.std(np.array([v for k, v in cossim_net.items()]), axis=0)
            return mean, std, None




    def get_cosine_similarity_one_class_random_img(self, class_num=None, dataset=None, **kwargs):
        """
        cossim: [group class idx][name feature][list of cossim for each step]
        """

        name_class = dataset.idx_to_class[class_num] if class_num is not None else None

        base_canvas, other_canvasses, x_values = self.get_base_and_other_canvasses(class_num, name_class, **kwargs)
        cossim_net, cossim_images,  p = self.get_cosine_similarity_from_images(base_canvas, other_canvasses, **kwargs)
        name_class, cossim_net, cossim_images, x_values = self.finalize_each_class(name_class, cossim_net, cossim_images, x_values)
        return cossim_net, cossim_images, x_values

    def calculate_distance_dataloader_each_class(self, dataset=None, compare_with=None, **kwargs):
            # self.setup_network()
            if compare_with is not None:
                self.compare_with = compare_with
            if dataset is None:
                dataset = self.dataset
            else:
                self.dataset = dataset
            cossim_net = {}
            cossim_img = {}
            for c in range(dataset.num_classes):
                cossim_net[c], cossim_img[c], x_values = self.get_cosine_similarity_one_class_random_img(c, dataset, **kwargs)

            return cossim_net, cossim_img, x_values


    def calculate_distance_dataloader(self, N=20,  dataset=None, compare_with=None, **kwargs):
            # self.setup_network()
            if compare_with is not None:
                self.compare_with = compare_with
            if dataset is None:
                dataset = self.dataset
            else:
                self.dataset = dataset
            cossim_net = []
            cossim_img = []
            x_values = []
            for i in tqdm(range(N)):
                cn, ci, xv = self.get_cosine_similarity_one_class_random_img(None, dataset, **kwargs)
                cossim_net.append(cn)
                cossim_img.append(ci)
                x_values.append(xv)
            return cossim_net, cossim_img, x_values

##

