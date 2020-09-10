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

    parser.add_argument("-r", "--num_runs",
                        help="run experiment n times",
                        type=int,
                        default=1)
    parser.add_argument("-fcuda", "--force_cuda",
                        help="Force to run it with cuda enabled (for testing)",
                        type=int,
                        default=0)
    parser.add_argument("-neptune", "--use_neptune",
                        help="Log stuff to neptune",
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
                        help="freeze the feature layer part of the VGG net",
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

    return parser

def neptune_log_dataset_info(dataloader, log_text='', dataset_name=None):
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

    # neptune.log_text('{} mean'.format(dataset_name), str(mean))
    # neptune.log_text('{} std'.format(dataset_name), str(std))
    # wandb.log({'{} mean'.format(dataset_name): str(mean),
    #           '{} std'.format(dataset_name) : str(std)})
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

            wandb.log({'Debug / {} AREA, [{}] '.format(log_text, dataset_name, translation_type_str): [wandb.Image(canvas)]}, step=0)
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

            wandb.log({'Debug / {} example images: [{}]'.format(log_text, dataset_name):
                               [wandb.Image(convert_normalized_tensor_to_plottable_array(im, mean, std,
                                                                             text=f'{str(lb)}' +
                                                                                  (f':"{nc[lb]}"' if nc[lb] != str(lb) else '') +
                                                                                  os.path.splitext(n)[0]).astype(np.uint8))
             for im, lb, n in zip(images, labels.numpy(), add_text)]}, step=0)

            # [neptune.log_image('{} example images: [{}]'.format(log_text, dataset_name),
            #                    (convert_normalized_tensor_to_plottable_array(im, mean, std,
            #                                                                         text=f'{str(lb)}' +
            #                                                                              (f':"{nc[lb]}"' if nc[lb] != str(lb) else '') +
            #                                                                              os.path.splitext(n)[0])).astype(np.uint8))
            #  for im, lb, n in zip(images, labels.numpy(), add_text)]

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

# class SaveInfoInDF(Callbacks):
#     def __init__():
#         super().__init__()
#         index_dataframe = ['net', 'class_name', 'transl_X', 'transl_Y', 'tested_area', 'is_correct', 'class_output']

#
# def run_test_loader(net, is_server, network_name, data_loader, num_iterations, num_classes=None, name_dataset=None, translation_type_str=None, compute_dataframe=True, log_text=''):
#     torch.cuda.empty_cache()
#     index_dataframe = ['net', 'class_name', 'transl_X', 'transl_Y', 'tested_area', 'is_correct', 'class_output']
#     if num_classes is None:
#         num_classes = data_loader.dataset.num_classes
#     if translation_type_str is None:
#         translation_type_str = data_loader.dataset.translation_type_str
#     if name_dataset is None:
#         name_dataset = data_loader.dataset.name_generator
#
#     column_names = build_columns(['class {}'.format(i) for i in range(num_classes)])
#
#     net.eval()
#     correct_tot = 0
#     total_samples = 0
#     rows_frames = []
#     confusion_matrix = torch.zeros(num_classes, num_classes)
#     reached_max_iter = False
#     with torch.no_grad():
#         for i, data in enumerate(data_loader, 0):
#             images_batch, labels_batch_t, more = data
#             face_center_batch_t = more['center']
#             output_batch_t = net(images_batch.cuda() if is_server else images_batch)
#             max_output, predicted_batch_t = torch.max(output_batch_t, 1)
#             correct_batch_t = ((predicted_batch_t.cuda() if is_server else predicted_batch_t) == (labels_batch_t.cuda() if is_server else labels_batch_t))
#             correct_tot += correct_batch_t.sum().item()
#             for t, p in zip(labels_batch_t.view(-1), predicted_batch_t.view(-1)):
#                 confusion_matrix[t.long(), p.long()] += 1
#             # images_batch, lb, fc = next(iter(data_loader))
#             # vis.imshow_batch(images_batch, data_loader.dataset.stats['mean'], data_loader.dataset.stats['std'])
#             total_samples += labels_batch_t.size(0)
#
#             if compute_dataframe:
#                 softmax_batch_t = torch.softmax(output_batch_t.cuda() if is_server else output_batch_t, 1)
#                 softmax_batch = np.array(softmax_batch_t.tolist())
#                 output_batch = np.array(output_batch_t.tolist())
#                 labels = labels_batch_t.tolist()
#                 predicted_batch = predicted_batch_t.tolist()
#                 correct_batch = correct_batch_t.tolist()
#                 face_center_batch = np.array([np.array(i) for i in face_center_batch_t]).transpose()
#
#                 for c, softmax_all_cat in enumerate(softmax_batch):
#                     output = output_batch[c]
#                     softmax = softmax_batch[c]
#                     softmax_correct_category = softmax[labels[c]]
#                     output_correct_category = output[labels[c]]
#                     max_softmax = np.max(softmax)
#                     max_output = np.max(output)
#                     correct = correct_batch[c]
#                     label = labels[c]
#                     predicted = predicted_batch[c]
#                     face_center = face_center_batch[c]
#
#                     assert softmax_correct_category == max_softmax if correct else True, 'softmax values: {}, is correct? {}'.format(softmax, correct)
#                     assert softmax_correct_category != max_softmax if not correct else True, 'softmax values: {}, is correct? {}'.format(softmax, correct)
#                     assert predicted == label if correct else predicted != label, 'softmax values: {}, is correct? {}'.format(softmax, correct)
#
#                     rows_frames.append([network_name, label, face_center[0], face_center[1], translation_type_str, correct, predicted, max_softmax, softmax_correct_category, *softmax, max_output, output_correct_category, *output])
#             if i >= num_iterations - 1:
#                 reached_max_iter = True
#                 print('Max iterations reached')
#                 break
#
#         if not reached_max_iter:
#             print('Dataset Exhausted')
#         accuracy = 100.0 * correct_tot / total_samples
#         print('*Dataset: {} on {} test images - Accuracy: {}%'.format(name_dataset, total_samples, accuracy))
#         conf_mat_acc = (confusion_matrix / confusion_matrix.sum(1)[:, None]).numpy()
#         # conf_mat_acc = np.zeros((5,5)) + 255
#         if is_server:
#             # Plot confidence Matrix
#             neptune.log_metric('{} Acc'.format(name_dataset), accuracy)
#             figure = plt.figure(figsize=(10, 7))
#             sn.heatmap(conf_mat_acc, annot=True, annot_kws={"size": 16})  # font size
#             plt.ylabel('truth')
#             plt.xlabel('predicted')
#             plt.title(name_dataset)
#             neptune.log_image('{} Confusion Matrix'.format(log_text), figure)
#
#
#     if compute_dataframe:
#         data_frame = pd.DataFrame(rows_frames)
#         data_frame = data_frame.set_index([i for i in range(len(index_dataframe))])
#         data_frame.index.names = index_dataframe
#         data_frame.columns = column_names
#         data_frame.reset_index(level='is_correct', inplace=True)
#
#     else:
#         data_frame = None
#
#     if is_server and compute_dataframe:
#         mean_accuracy = data_frame.groupby(['transl_X', 'transl_Y']).mean()['is_correct']
#         ax, fig, im = imshow_density(mean_accuracy, lim=[1 / data_loader.dataset.num_classes - 1 / data_loader.dataset.num_classes * 0.2, 1], plot_args={'interpolate': True})
#         plt.title(name_dataset)
#         cbar = fig.colorbar(im)
#         cbar.set_label('Mean Accuracy (%)', rotation=270, labelpad=25)
#         neptune.log_image('{} Density Plot Accuracy'.format(log_text), fig)
#
#     return data_frame, conf_mat_acc, accuracy

#
# def test_loader_and_save(testing_loader_list, compute_dataframe, net, is_server, network_name, num_iterations_testing, log_text=''):
#     conf_mat_acc_all_tests = []
#     accuracy_all_tests = []
#
#     print('Running the tests')
#     df_testing = pd.DataFrame([])
#     for testing_loader in testing_loader_list:
#         df_testing_one, conf_mat_acc, accuracy = run_test_loader(net, is_server, network_name, testing_loader, num_iterations_testing, compute_dataframe=compute_dataframe, log_text=log_text)
#         conf_mat_acc_all_tests.append(conf_mat_acc)
#         accuracy_all_tests.append(accuracy)
#         # df_post_priming['run no.'] = run  # saved from another file, can be useful here
#         # [df_post_priming.set_index(i, append=True, inplace=True) for i in ['run no.']]
#         df_testing = pd.concat((df_testing, df_testing_one))
#
#     return df_testing, conf_mat_acc_all_tests, accuracy_all_tests


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
    # std = np.array([0.229, 0.224, 0.225])
    # mean = np.array([0.969, 0.969, 0.969])
    # std = np.array([0.126, 0.126, 0.126])
    # mean = np.array([0.969])
    # std = np.array([0.138])
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