import argparse
import os
import cv2
import neptune
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import interpolate

from generate_datasets.generators.translate_generator import TranslateGenerator
import wandb

desired_width = 320
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 4)
pd.set_option('display.width', desired_width)


def make_cuda(fun, is_cuda):
    return fun.cuda() if is_cuda else fun


def remap_value(x, range_source, range_target):
    return range_target[0] + (x - range_source[0]) * (range_target[1] - range_target[0]) / (range_source[1] - range_source[0])


def parse_experiment_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(allow_abbrev=False)
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
    parser.add_argument("-weblog", "--use_weblog",
                        help="Log stuff to the weblogger (wandb, neptune, etc.)",
                        type=int,
                        default=1)
    parser.add_argument("-tags", "--additional_tags",
                        help="Add additional tags. Separate them by underscore. E.g. tag1_tag2",
                        type=str,
                        default=None)
    parser.add_argument("-prjnm", "--project_name",
                        type=str,
                        default='RandomProject')
    parser.add_argument("-wbg", "--wandb_group_name",
                        help="Group name for weight and biases, to organize sub experiments of a bigger project",
                        type=str,
                        default=None)
    parser.add_argument("-pat1", "--patience_stagnation",
                        help="Patience for early stopping for stagnation (num iter)",
                        type=int,
                        default=800)
    parser.add_argument("-so", "--size_object",
                        help="Change the size of the object. W_H (x, y). Set to 0 if you don't want to resize the object",
                        type=str,
                        default='50_50')
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


def parse_few_shot_learning_parameters(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("-trainf", "--folder_for_training",
                        help="Select the folder for training this network.",
                        type=str,
                        default=None)
    parser.add_argument("-testf", "--folder_for_testing",
                        help="Select the folder(s) for testing this network.",
                        type=str,
                        nargs='+',
                        default=None)
    parser.add_argument("-n_shot", "--n_shot",
                        help="images for each classes for meta-training.",
                        type=int,
                        default=2)
    parser.add_argument("-k_way", "--k_way",
                        help="number of classes for meta-training",
                        type=int,
                        default=5)
    parser.add_argument("-q_query", "--q_queries",
                        help="number of query for each class for meta-training.",
                        type=int,
                        default=1)
    parser.add_argument("-sc", "--size_canvas",
                        help="Change the size of the canvas. Canvas can be smaller than object (object will be cropped). "
                             "Use only for purely convolutional networks will not support this (the FC layer needs to be specified",
                        type=lambda x: tuple([int(i) for i in x.split("_")]),
                        default='224_224')
    return parser

def parse_standard_training_arguments(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("-sFC", "--shallow_FC",
                        help='use a shallow fully connected layer (only a connection x to num_classes)',
                        default=0, type=int)
    parser.add_argument("-gap", "--use_gap",
                        help="use GAP layer at the end of the convolutional layers",
                        type=int,
                        default=0)
    parser.add_argument("-f", "--feature_extraction",
                        help="freeze the feature (conv) layers part of the VGG net",
                        type=int,
                        default=0)
    parser.add_argument("-bC", "--big_canvas",
                        help="If true, will use 400x400 canvas (otherwise 224x224). The VGG network will be changed accordingly (we won't use the adaptive GAP)",
                        type=int,
                        default=0)
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

def weblog_dataset_info(dataloader, log_text='', dataset_name=None):
    compute_my_generator_info = False
    if isinstance(dataloader.dataset, TranslateGenerator):
        dataset = dataloader.dataset
        translation_type_str = dataset.translation_type_str
        compute_my_generator_info = True
        dataset_name = dataset.name_generator
        mean = dataloader.dataset.stats['mean']
        std = dataloader.dataset.stats['std']
    else:
        dataset_name = 'no_name' if dataset_name is None else dataset_name
        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]
        Warning('MEAN, STD AND DATASET_NAME NOT SET FOR NEPTUNE LOGGING. This message is not referring to normalizing in PyTorch')

    wandb.run.summary['{} mean'.format(dataset_name)] = mean
    wandb.run.summary['{} std'.format(dataset_name)] = std

    size_object = dataset.size_object if dataset.size_object is not None else (0, 0)

    def draw_rect(canvas, range):
        canvas = cv2.rectangle(canvas, (range[0], range[2]),
                                       (range[1] - 1, range[3] - 1), (0, 0, 255), 2)
        canvas = cv2.rectangle(canvas, (range[0] - size_object[0] // 2, range[2] - size_object[0] // 2),
                                       (range[1] + size_object[1] // 2 - 1, range[3] + size_object[1] // 2 - 1), (0, 0, 240), 2)
        return canvas

    break_after_one = False
    if compute_my_generator_info:
        if all(value == list(dataset.translations_range.values())[0] for value in dataset.translations_range.values()):
            break_after_one = True
        for groupID, rangeC in dataset.translations_range.items():
            canvas = np.zeros(dataset.size_canvas) + 254
            if isinstance(rangeC[0], tuple):
                for r in rangeC:
                    canvas = draw_rect(canvas, r)
            else:
                canvas = draw_rect(canvas, rangeC)
            wandb.log({f'Debug/{log_text} AREA [{dataset_name}] ': [wandb.Image(canvas)]}, step=0)

            # neptune.log_image('{} AREA, [{}] '.format(log_text, dataset_name, translation_type_str), canvas.astype(np.uint8))
            if break_after_one:
                break

    iterator = iter(dataloader)
    nc = dataloader.dataset.name_classes
    for i in range(1):
        try:
            images, labels, more = next(iterator)
            images = images[0:np.max((4, len(images)))]
            labels = labels[0:np.max((4, len(labels)))]
            # more = more[0:np.max((4, len(more)))]

            add_text = [''] * len(labels)
            if 'image_name' in list(more.keys()):
                add_text = more['image_name']

            wandb.log({'Debug/{} example images: [{}]'.format(log_text, dataset_name):
                               [wandb.Image(convert_normalized_tensor_to_plottable_array(im, mean, std,
                                                                             text=f'{lb}' +
                                                                                  os.path.splitext(n)[0]).astype(np.uint8))
             for im, lb, n in zip(images, labels, add_text)]}, step=0)

        except StopIteration:
            Warning('Iteration stopped when plotting [{}] on Neptune'.format(dataset_name))


def convert_pil_img_to_tensor(imageT, normalize):
    """

    @param imageT: a PIL image (HxWxC)
    @param normalize: Normalize is a transform operation in pytorch
    @return: a Tensor
    """
    t = torch.tensor(np.array(imageT))
    t = t.permute(2, 0, 1)
    t = t / 255.0
    t = normalize(t)
    t = t.unsqueeze(0)
    return t


def copy_img_in_center_canvas(image: Image, size_canvas):
    """
    Place an image in the center of a bigger canvas
    @param image:
    @param size_canvas:
    @return:
    """
    # Put the image in a big canvas
    canvas = Image.new('RGBA', tuple(size_canvas), 'white')
    # Resize the target image
    size_image = np.array(image.size)
    img_resized = image.resize(size_image)
    left, up = (size_canvas / 2 - size_image / 2).astype(int)
    bottom, right = np.array([left, up]) + size_image
    canvas.paste(img_resized, (left, up, bottom, right))
    return canvas


def copy_img_in_canvas(image: Image, size_canvas: tuple, position, color_canvas='white'):
    """
    Place an image in the center of a bigger canvas
    @param image:
    @param size_canvas:
    @return:
    """
    if isinstance(color_canvas, int):
        color_canvas = (color_canvas, ) * 3
    # Put the image in a big canvas
    canvas = Image.new('RGBA', size_canvas, color_canvas)
    # Resize the target image
    size_image = np.array(image.size)
    left, up = position - size_image // 2
    bottom, right = np.array([left, up]) + size_image
    canvas.paste(image, (left, up, bottom, right), image if np.shape(image)[-1] == 4 else None)
    canvas = canvas.convert('RGB')
    return canvas

def convert_normalized_tensor_to_plottable_figure(tensor, mean, std, title_lab=None, title_more=''):
    tensor = tensor.numpy().transpose((1, 2, 0))
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    fig = plt.figure(1, facecolor='gray')
    if len(np.shape(image)) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    if title_lab is not None:
        plt.title(str(title_lab) + ' ' + title_more if title_more != '' else '')
    plt.axis('off')
    plt.tight_layout()
    plt.close(fig)
    return fig


def convert_normalized_tensor_to_plottable_array(tensor, mean, std, text):
    image = conver_tensor_to_plot(tensor, mean, std)
    canvas_size = np.shape(image)
    font_scale = np.ceil(canvas_size[1])/150
    font = cv2.QT_FONT_NORMAL
    umat = cv2.UMat(image * 255)
    umat = cv2.putText(cv2.UMat(umat), text=text, org=(0, int(canvas_size[1] - 3)), fontFace=font, fontScale=font_scale, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=6)
    umat = cv2.putText(img=cv2.UMat(umat), text=text, org=(0, int(canvas_size[1] -3)),
                fontFace=font, fontScale=font_scale, color=[255, 255, 255], lineType=cv2.LINE_AA, thickness=1)
    # cv2.imshow('ciao', image)
    image = cv2.UMat.get(umat)
    image = np.array(image, np.int8)
    # plt.imshow(image)
    return image


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image


def imshow_batch(inp, mean=None, std=None, title_lab=None, title_more=''):
    if mean is None:
        mean = np.array([0.5, 0.5, 0.5])
    if std is None:
        std = np.array([0.5, 0.5, 0.5])
    """Imshow for Tensor."""
    fig = plt.figure(1, facecolor='gray')
    for idx, image in enumerate(inp):
        cols = np.min([5, len(inp)])
        image = conver_tensor_to_plot(image, mean, std)
        plt.subplot(int(np.ceil(np.shape(inp)[0]/cols)), cols, idx+1)
        plt.axis('off')
        if len(np.shape(image)) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        if title_lab is not None:
            plt.title(str(title_lab[idx].item()) + ' ' + (title_more[idx] if title_more != '' else ''))

    plt.pause(0.1)
    plt.tight_layout()
    return fig


def interpolate_grid(canvas):
    x = np.arange(0, canvas.shape[1])
    y = np.arange(0, canvas.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(canvas)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    canvas = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                  method='cubic')
    return canvas


def compute_density(values, plot_args=None):
    do_interpolate = False
    if plot_args is None:
        plot_args = {}
    if 'interpolate' in list(plot_args.keys()) and plot_args['interpolate']:
        do_interpolate = True
    if 'size_canvas' in list(plot_args.keys()):
        size_canvas = plot_args['size_canvas']
    else:
        size_canvas = (224, 224)

    canvas = np.empty(size_canvas)
    canvas[:] = np.nan
    x_values = np.array(values.index.get_level_values('transl_X'), dtype=int)
    y_values = np.array(values.index.get_level_values('transl_Y'), dtype=int)
    canvas[y_values, x_values] = values
    # negative values are black, nan are white
    # ax.imshow(canvas, vmin=lim[0], vmax=lim[1], cmap='viridis')
    # plt.colorbar(ax=ax)
    try:
        if do_interpolate:
            canvas = interpolate_grid(canvas)
    except:
        pass
    return canvas


def imshow_density(values, ax=None, plot_args=None, **kwargs):
    fig = None
    cmap = 'viridis' if 'cmap' not in kwargs else kwargs['cmap']
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    canvas = compute_density(values, plot_args)
    cm = plt.get_cmap(cmap)
    canvas = cm(canvas)
    im = ax.imshow(canvas, **kwargs)

    # TODO: This hasattr has to be done because the dataset structure changed. If you re-run all the experiments then you can delete the first part, hasattr(.., 'minX')
    if 'dataset' in list(plot_args.keys()):
        dataset = plot_args['dataset']
        if hasattr(dataset, 'minX'):
            ax.add_patch(Rectangle((dataset.minX, dataset.minY), dataset.maxX-dataset.minX, dataset.maxY-dataset.minY, edgecolor='r', facecolor='none', linewidth=2))
        elif hasattr(dataset, 'translations_range'):
            for groupID, rangeC in dataset.translations_range.items():
                ax.add_patch(Rectangle((rangeC[0], rangeC[2]), rangeC[1] - rangeC[0], rangeC[3] - rangeC[2], edgecolor='r', facecolor='none', linewidth=2))


    plt.show()
    return ax, fig, im