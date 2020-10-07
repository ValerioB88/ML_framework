import torch
import framework_utils
from framework_utils import make_cuda
from generate_datasets.generators.utils_generator import get_background_color
import numpy as np
import PIL.Image as Image
from abc import ABC, abstractmethod


class CosSim(ABC):
    def __init__(self, net, dataset):
        if torch.cuda.is_available():
            self.cuda = True
        self.net = net
        self.dataset = dataset
        self.only_save = ['Conv2d', 'Linear']
        self.detach_this_step = True
        self.activation = {}

    @abstractmethod
    def get_base_and_other_canvasses(self, class_num, name_class, image: Image):
        raise NotImplementedError

    def get_activation(self, name):
        def hook(model, input, output):
            if self.detach_this_step:
                self.activation[name] = output.detach()

        return hook

    def group_all_layers(self):
        all_layers = []

        def recursive_group():
            for layer in self.net.children():
                if not list(layer.children()):  # if leaf node, add it to list
                    all_layers.append(layer)
                else:
                    recursive_group(layer)

        recursive_group()
        return all_layers

    def get_cosine_similarity_one_class_random_img(self, class_num):
        """
        cossim: [group class idx][name feature][list of cossim for each step]
        """

        name_class = self.dataset.map_num_to_name[class_num]
        num = np.random.choice(len(self.dataset.samples[name_class]))
        image_name = self.dataset.samples[name_class][num]

        image = Image.open(self.dataset.folder + '/' + image_name)
        base_canvas, other_canvasses = self.get_base_and_other_canvasses(class_num, name_class, image)
        self.net(make_cuda(base_canvas.unsqueeze(0), self.cuda))
        base_activation = {}
        # base_activation = activation['one_to_last']
        for name, features in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            base_activation[name] = features

        # cos_fun = torch.nn.CosineSimilarity(dim=1)
        cossim = {}
        for canvas_comparison in other_canvasses:
            canvas_comparison_activation = {}
            self.net(make_cuda(canvas_comparison.unsqueeze(0), self.cuda))
            for name, features in self.activation.items():
                if not np.any([i in name for i in self.only_save]):
                    continue
                canvas_comparison_activation[name] = features
                if name not in cossim:
                    cossim[name] = []
                # canvas_comparison_activation = features
                cossim[name].append(torch.nn.CosineSimilarity(dim=0)(base_activation[name].flatten(), canvas_comparison_activation[name].flatten()).item())
        return cossim

    def calculate_cossim_network(self):
        ## HOOK NETWORK ACTIVATION
        all_layers = self.group_all_layers()
        hook_lists = []
        for idx, i in enumerate(all_layers):
            hook_lists.append(i.register_forward_hook(self.get_activation('{}: {}'.format(idx, str.split(str(i), '(')[0]))))

        ## NOW GET COSINE SIM
        cossim = {}
        for c in range(self.dataset.num_classes):
            cossim[c] = self.get_cosine_similarity_one_class_random_img(c)

        ## REMOVE HOOKS
        for h in hook_lists:
            h.remove()
        return cossim


class CosSimTranslation(CosSim):
    def get_base_and_other_canvasses(self, class_num, name_class, image: Image):
        def get_canvas(center):
            # random_center = dataset._get_translation(label, image_name, idx)
            canvas = framework_utils.copy_img_in_canvas(image, self.dataset.size_canvas, center, color_canvas=get_background_color(self.dataset.background_color_type))
            canvas, label, more = self.dataset._finalize_get_item(canvas, class_num, {'center': center})
            canvas = self.dataset.transform(canvas)
            return canvas

        other_canvasses = []
        image, *_ = self.dataset._resize(image)
        minX, maxX, minY, maxY = self.dataset.translations_range[name_class]
        stepsX = self.dataset.symmetric_steps(minX, maxX)
        stepY = self.dataset.size_canvas[1] // 2
        base_center = (stepsX[0], stepY)

        base_canvas = get_canvas(base_center)
        for stX in stepsX:
            center = (stX, stepY)
            other_canvasses.append(get_canvas(center))

        return base_canvas, other_canvasses