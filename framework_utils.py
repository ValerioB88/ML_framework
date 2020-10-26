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
from generate_datasets.generators.input_image_generator import InputImagesGenerator
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


def weblog_dataset_info(dataloader, log_text='', dataset_name=None, weblogger=1):
    compute_my_generator_info = False
    if isinstance(dataloader.dataset, InputImagesGenerator):
        dataset = dataloader.dataset
        compute_my_generator_info = True
        dataset_name = dataset.name_generator
        mean = dataloader.dataset.stats['mean']
        std = dataloader.dataset.stats['std']
    else:
        dataset_name = 'no_name' if dataset_name is None else dataset_name
        mean = [0.5, 0.5, 0.5]
        std = [0.2, 0.2, 0.2]
        Warning('MEAN, STD AND DATASET_NAME NOT SET FOR NEPTUNE LOGGING. This message is not referring to normalizing in PyTorch')

    if weblogger == 1:
        wandb.run.summary['Log: {} mean'.format(dataset_name)] = mean
        wandb.run.summary['Log: {} std'.format(dataset_name)] = std
    if weblogger == 2:
        neptune.log_text('Log/{} mean'.format(dataset_name), str(mean))
        neptune.log_text('Log/{} std'.format(dataset_name), str(std))
    if isinstance(dataset, TranslateGenerator):

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
                metric_str = f'Debug/{log_text} AREA [{dataset_name}] '
                if weblogger == 1:
                    wandb.log({metric_str: [wandb.Image(canvas)]}, step=0)
                if weblogger == 2:
                    neptune.log_image(metric_str, canvas.astype(np.uint8))
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
            metric_str = 'Debug/{} example images: [{}]'.format(log_text, dataset_name)
            if weblogger == 1:
                wandb.log({metric_str:
                               [wandb.Image(convert_normalized_tensor_to_plottable_array(im, mean, std,
                                                                             text=f'{lb}' +
                                                                                  os.path.splitext(n)[0]).astype(np.uint8))
             for im, lb, n in zip(images, labels, add_text)]}, step=0)
            if weblogger == 2:
                [neptune.log_image(metric_str,
                                   convert_normalized_tensor_to_plottable_array(im, mean, std, text=f'{lb}' + os.path.splitext(n)[0]).astype(np.uint8)) for im, lb, n in zip(images, labels, add_text)]
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


def imshow_batch(inp, stats=None, title_lab=None, title_more=''):
    if stats is None:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
    else:
        mean = stats['mean']
        std = stats['std']
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