from neptune.new.types import File
from sty import fg, bg, rs, ef
import argparse
import os
import neptune.new as neptune
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from collections import namedtuple
import matplotlib as mpl
import matplotlib
import shutil
import re
import glob
import pathlib
from pathlib import Path
import pprint


def add_record(self, record):
    new_cols = list(set(record.keys()) - set(self.columns))
    for c in new_cols:
        self[c] = np.NAN
    ss = " & ".join([f'((self["{k}"] == ' + (f'"{v}"' if isinstance(v, str) else (f'np.NAN' if v is np.NAN else f'{v}')) + f') | pd.isnull(self["{k}"]))' for k, v in record.items()])
    if self.loc[eval(ss)].empty: #or np.all([True if pd.isnull(self.loc[eval(ss)][i]).values[0] else False for i in record.keys()]):
        self = self.append(record, ignore_index=True)
    else:
        for cc in self.columns[self.loc[eval(ss)].isnull().any()]:
            if cc in list(record.keys()):
                self.loc[eval(ss), cc] = record[cc]
    return self


import pandas as pd
pd.DataFrame.add_record = add_record

plt.rcParams['figure.facecolor'] = (0.9, 0.9, 0.9)
# matplotlib.use('TkAgg')
if not torch.cuda.is_available():
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('agg')


plt.rcParams['figure.facecolor'] = (0.9, 0.9, 0.9)
# matplotlib.use('TkAgg')
# matplotlib.use('QtAgg')
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)
desired_width = 423
np.set_printoptions(linewidth=desired_width)
torch.set_printoptions(linewidth=desired_width)
pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 4)
pd.set_option('display.width', desired_width)
pd.set_option("expand_frame_repr", False) # print cols side by side as it's supposed to b
pd.set_option("display.max_rows", None, "display.max_columns", None)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#

def get_all_dir_tokens(folder, delimiters=None):
    if delimiters is None:
        delimiters = os.path.sep, "_"
    all_folders = [r for r, d, f in os.walk(folder) if not d]
    all_subdirs = [r.split(folder)[1] for r in all_folders]

    regexPattern = '|'.join(map(re.escape, delimiters))

    all_dirs_tokens = [[i for i in re.split(regexPattern, d) if i] for d in all_subdirs]
    return np.array(all_dirs_tokens).T, all_folders  # check if dimension is ok


def rearrange_folders(folder, string, tokens, all_folders):
    folder = folder.strip('//').strip('/')
    # shutil.copytree(folder, folder + '_tmp')
    # shutil.rmtree(folder)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    tokens = tokens.T.tolist()
    new_folders = [folder + '_tmp//' + f'{string}'.format(*i) for i in tokens]
    # new_folders = [folder + '//{0}//{2}//{1}_{4}'.format(*i) for i in tokens]

    for i in range(len(all_folders)):
        old = all_folders[i]
        new = new_folders[i]
        pathlib.Path(new).mkdir(parents=True, exist_ok=True)
        [shutil.copy(i, new) for i in glob.glob(old + '/**')]
    shutil.move(folder, folder + '_bk')
    shutil.move(folder + '_tmp', folder)

# folder = './results/pomerantz_stimuli'
# tokens, all_folders = get_all_dir_tokens(folder)
# rearrange_folders(folder, '{0}//{2}//{1}_{4}', tokens, all_folders)


def get_all_dir_tokens(folder, delimiters=None):
    if delimiters is None:
        delimiters = os.path.sep, "_", "."
    all_files = [(r, f) for r, d, f in os.walk(folder) if not d]
    all_files = sum(*[[["\\".join([i[0], j]) for j in i[1]] for i in all_files]], [])
    all_subdirs = [r.split(folder)[1] for r in all_files]

    regexPattern = '|'.join(map(re.escape, delimiters))
    d = all_subdirs[0]
    import itertools as it
    cnt = it.count()
    pattern = re.sub(r'[a-zA-Z\-]+', lambda x: f'{{{next(cnt)}}}', d)
    all_dirs_tokens = [[i for i in re.split(regexPattern, d) if i] for d in all_subdirs]
    tokens = np.array(all_dirs_tokens).T
    if not len(np.unique([len(t) for t in tokens])) == 1:
        assert False, fg.yellow + "Different tokens' length! Can't reformart" + rs.fg

    tt = {idx: list(t[0:np.min([3, len(t)])]) for idx, t in enumerate(tokens)}
    print(fg.blue)
    print(f"EXAMPLE TOKENS: \n")
    import json
    print(json.dumps(tt, indent=4))
    print(rs.fg)
    return tokens, all_files, pattern


def rearrange_folders(folder, string, tokens, all_files):
    from datetime import datetime

    today = datetime.now()
    # check if dimension is ok
    assert len(np.unique([len(t) for t in tokens])) == 1, "Different tokens' length! Can't reformart"
    folder = folder.strip('//').strip('/')
    # shutil.copytree(folder, folder + '_tmp')
    # shutil.rmtree(folder)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    tokens = tokens.T.tolist()
    date = today.strftime("%H%M%S_%d%m%y")

    new_folders = [folder + f'_rearranged_{date}//' + eval('f"' + re.sub(r"(\d)", r"i[\1]", string) + '"') for i in tokens]
    # new_folders = [folder + '//{0}//{2}//{1}_{4}'.format(*i) for i in tokens]
    # shutil.copytree(folder, folder + '_bk')
    for i in range(len(all_files)):
        old = all_files[i]
        new = new_folders[i]
        Path(new).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(old, new)
    # shutil.move(folder + '_tmp', folder)

# tokens, all_files, pattern = get_all_dir_tokens('./results/pomerantz_stimuli/special')
# rearrange_folders('./results/pomerantz_stimuli/special', "\\{0}\\{1}_{2}\\{(3 + '-ts' if 3 == 'parentheses' or 3 == 'parentheses-crossed' else 3)}\\{4}.{5}", tokens, all_files)

# rearrange_folders('./results/pomerantz_stimuli/special', '//{0}_{2}_{1}_{3}_"({4}).upper()")', tokens, all_files)

# folder = './results/pomerantz_stimuli'
# tokens, all_folders = get_all_dir_tokens(folder)
# rearrange_folders(folder, '{0}//{2}//{1}_{4}', tokens, all_folders)

class Logs():
    value = None

    def __repl__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __copy__(self):
        return self.value

    def __deepcopy__(self, memodict={}):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rfloordiv__(self, other):
        return other // self.value

    def __rtruediv__(self, other):
        return other / self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __floordiv__(self, other):
        return self.value // other

    def __truediv__(self, other):
        return self.value / other

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __int__(self):
        return int(self.value)

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other

    def __float__(self):
        return float(self.value)

    def __pow__(self, power, modulo=None):
        return self.value ** power

    def __format__(self, format_spec):
        return format(self.value, format_spec)

class ExpMovingAverage(Logs):
    value = None
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def add(self, *args):
        if self.value is None:
            self.value = args[0]
        else:
            self.value = self.alpha * args[0] + (1 -    self.alpha) * self.value
        return self


class CumulativeAverage(Logs):
    value = None
    n = 0

    def add(self, *args):
        if self.value is None:
            self.value = args[0]

        else:
            self.value = (args[0] + self.n*self.value) / (self.n+1)
        self.n += 1
        return self


def scatter_plot_on_sphere(points, correct, title):
    import plotly.express as px
    import plotly.graph_objects as go
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z']).join(pd.DataFrame({'correct': correct}))

    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(theta), np.sin(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.ones(100), np.cos(phi))
##
    data_sphere = go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.2,
            colorscale=[[0, 'rgb(220, 220, 220)'], [1, 'rgb(190, 190, 190)']],
            hoverinfo='skip'
    )
    layout = go.Layout(
        title=title,
        height=800,
        width=1000
    )
    plotlyfig = go.Figure(data=[go.Scatter3d(
                                x=df['X'],
                                y=df['Y'],
                                z=df['Z'],
                                mode='markers',
                                marker=dict(
                                    size=5,
                                    color=['red' if i is False else 'blue' for i in df['correct']],                # set color to an array/list of desired values
                                    opacity=0.2)
                                )], layout=layout)
    # px.scatter_3d(df, x='X', y='Y', z='Z', color='correct', opacity=0.5)
    plotlyfig.add_trace(go.Scatter3d(x=[1], y=[0], z=[0], marker=dict(size=12, color=['green'])))
    plotlyfig.add_trace(data_sphere)
    return plotlyfig



def align_vectors(a, b):
    def get_incl_and_azi(a):
        # must be unit vector
        incl = np.arccos(a[2])
        azi = np.arctan2(a[1], a[0])
        return incl, azi

    def get_cart_coord(incl, azi, r=1):
        x = r * np.sin(incl) * np.cos(azi)
        y = r * np.sin(incl) * np.sin(azi)
        z = r * np.cos(incl)
        return x, y, z

    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    incl1, azi1 = get_incl_and_azi(a)
    incl2, azi2 = get_incl_and_azi(b)

    di = incl2 - incl1
    da = azi2 - azi1
    c = get_cart_coord(di + np.pi / 2, da)
    return np.array(c)


def from_dataframe_to_3D_scatter(dataframe, title):
    # Compute Distance Between Query and average support
    candidate_campos = np.array([np.mean(i.reshape(-1, 3), axis=0) for i in dataframe['candidate_campos_XYZ']])
    training_campos = np.array([np.mean(i.reshape(-1, 3), axis=0) for i in dataframe['training_campos_XYZ']])
    correct = [i for i in dataframe['is_correct']]

    aligned_all_vectors = np.array([align_vectors(q, s) for q, s in zip(candidate_campos, training_campos)])
    aligned_norm_vect = np.array([i / np.linalg.norm(i) for i in aligned_all_vectors])
    mplt_colors = ['r' if i is False else 'b' for i in correct]
    plotly_colors = ['y' if i is False else 'b' for i in correct]
    # MATPLOTLIB version
    mplt_fig, ax = create_sphere(color='k')
    plt.title(title, size=24)
    add_norm_vector([2, 0, 0], ax=ax, col='k', norm=False)
    ax.scatter(aligned_norm_vect[:, 0], aligned_norm_vect[:, 1], aligned_norm_vect[:, 2], c=mplt_colors)

    # PLOTLY version
    plotly_fig = scatter_plot_on_sphere(aligned_norm_vect, correct, title)
    return plotly_fig, mplt_fig


def create_sphere(color='r', r=1):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x*r, y*r, z*r, color=color, alpha=0.1)

    # draw a point
    ax.scatter([0], [0], [0], color="b", s=100)

    vv = np.array([1, 0, 0])
    # add_norm_vector(vv * r, 'b', ax=ax, norm=False)

    plt.tight_layout()
    # plt.gca().set_aspect('equal', adjustable='box')
    return fig, ax


def add_norm_vector(u, col="k", ax=None, norm=True, lw=2, **kwargs):
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    if norm:
        u = u / np.linalg.norm(u)
    vh = Arrow3D([0, u[0]], [0, u[1]], [0, u[2]], mutation_scale=20, arrowstyle="-|>", color=col, lw=lw, **kwargs)
    ax.add_artist(vh)
    return vh

def get_all_files_with_ext(path, ext):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if (file.endswith(ext)):
                all_files.append(os.path.join(root, file))
    return all_files

def make_cuda(fun, is_cuda):
    return fun.cuda() if is_cuda else fun


def remap_value(x, range_source, range_target):
    return range_target[0] + (x - range_source[0]) * (range_target[1] - range_target[0]) / (range_source[1] - range_source[0])


def print_net_info(net):
    """
    Get net must be reimplemented for any non abstract base class. It returns the network and the parameters to be updated during training
    """
    num_trainable_params = 0
    tmp = ''
    print(fg.yellow)
    print("Params to learn:")
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            tmp += "\t" + name + "\n"
            print("\t" + name)
            num_trainable_params += len(param.flatten())
    print(f"Trainable Params: {num_trainable_params}")

    print('***Network***')
    print(net)
    print(ef.inverse + f"Network is in {('~train~' if net.training else '~eval~')} mode." + rs.inverse)
    print(rs.fg)
    print()


def weblogging_plot_generators_info(loader_list, weblogger=1, num_batches_to_log=2):
    if weblogger:
        for loader in loader_list:
            weblog_dataset_info(loader, log_text=loader.dataset.name_generator, weblogger=weblogger, num_batches_to_log=num_batches_to_log)


def weblog_dataset_info(dataloader, log_text='', dataset_name=None, weblogger=1, num_batches_to_log=2):
    stats = {}
    compute_my_generator_info = False
    if 'stats' in dir(dataloader.dataset):
        dataset = dataloader.dataset
        compute_my_generator_info = True
        dataset_name = dataset.name_generator
        stats = dataloader.dataset.stats
    else:
        dataset_name = 'no_name' if dataset_name is None else dataset_name
        stats['mean'] = [0.5, 0.5, 0.5]
        stats['std'] = [0.2, 0.2, 0.2]
        Warning('MEAN, STD AND DATASET_NAME NOT SET FOR NEPTUNE LOGGING. This message is not referring to normalizing in PyTorch')

    if isinstance(weblogger, neptune.run.Run):
        weblogger['Logs'] = f'{dataset_name} mean: {stats["mean"]}, std: {stats["std"]}'

    nc = dataset.classes
    for idx, data in enumerate(dataloader):
        images, labels, more = data
        plot_images_on_weblogger(dataset_name, stats, images, labels, more, log_text, weblogger)
        if idx + 1 >= num_batches_to_log:
            break


def plot_images_on_weblogger(dataset_name, stats, images, labels, more, log_text, weblogger=2):
    plot_images = images[0:np.max((4, len(images)))]
    if labels is None:
        labels = np.repeat('', len(plot_images))
    labels = labels[0:np.max((4, len(labels)))]
    add_text = [''] * len(labels)
    if isinstance(more, dict) and 'image_name' in list(more.keys()):
        add_text = more['image_name']
    metric_str = 'Debug/{} example images: [{}]'.format(log_text, dataset_name)

    if isinstance(weblogger, neptune.run.Run):
        [weblogger[metric_str].log
                           (File.as_image(convert_normalized_tensor_to_plottable_array(im, stats['mean'], stats['std'], text=f'{lb}' + os.path.splitext(n)[0])/255))
         for im, lb, n in zip(plot_images, labels, add_text)]


class TwoWaysDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


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


LAYOUT = []
def save_figure_layout(filename=None):
    import cloudpickle

    global LAYOUT
    figt = namedtuple('figure', 'num size position')
    fignums = plt.get_fignums()
    LAYOUT = []
    for i in fignums:
        fig = plt.figure(i)
        size = plt.get_current_fig_manager().window.size()
        pos = plt.get_current_fig_manager().window.pos()
        LAYOUT.append(figt(i, [size.width(), size.height()], [pos.x(), pos.y()]))
    if filename is not None:
        cloudpickle.dump(LAYOUT, open(filename, 'wb'))
    return LAYOUT


def load_figure_layout(filename=None, offset=0):
    import cloudpickle

    global LAYOUT
    if filename is not None:
        LAYOUT = cloudpickle.load(open(filename, "rb"))

    for i in range(len(LAYOUT)):
        fig = plt.figure(LAYOUT[i].num + offset)
        plt.get_current_fig_manager().window.setGeometry(*LAYOUT[i].position, *LAYOUT[i].size)
        plt.get_current_fig_manager().window.activateWindow()
        plt.get_current_fig_manager().window.show()


def convert_normalized_tensor_to_plottable_array(tensor, mean, std, text):
    image = conver_tensor_to_plot(tensor, mean, std)
    canvas_size = np.shape(image)
    font_scale = np.ceil(canvas_size[1])/150
    font = cv2.QT_FONT_NORMAL
    umat = cv2.UMat(image * 255)
    umat = cv2.putText(cv2.UMat(umat), text=text, org=(0, int(canvas_size[1] - 3)), fontFace=font, fontScale=font_scale, color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=6)
    umat = cv2.putText(img=cv2.UMat(umat), text=text, org=(0, int(canvas_size[1] - 3)),
                fontFace=font, fontScale=font_scale, color=[255, 255, 255], lineType=cv2.LINE_AA, thickness=1)
    image = cv2.UMat.get(umat)
    image = np.array(image, np.uint8)
    return image


def conver_tensor_to_plot(tensor, mean, std):
    tensor = tensor.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    image = std * tensor + mean
    image = np.clip(image, 0, 1)
    if np.shape(image)[2] == 1:
        image = np.squeeze(image)
    return image

def get_random_filepath_from_folder(folder, ext='png', N=1):
    import glob
    import re
    import numpy as np
    all_files = glob.glob(folder + f'/*.{ext}')
    paths = np.random.choice(all_files, N)
    return paths

def get_highest_number_file(folder, ext='[a-zA-Z]+'):
    import glob
    import re
    all_files = glob.glob(folder + '*')
    a = [re.findall(rf"([0-9]+){ext}$", i) for i in all_files]
    a = [int(i[0]) if i else -np.inf for i in a]
    if len(a) == 0:
        assert False, f'Folder {folder} doest not contain the right type of file'
    return all_files[np.argmax(a)]

def maximize():
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

def imshow_batch(inp, stats=None, labels=None, title_more='', maximize=True, ax=None):
    if stats is None:
        mean = np.array([0, 0, 0])
        std = np.array([1, 1, 1])
    else:
        mean = stats['mean']
        std = stats['std']
    """Imshow for Tensor."""

    cols = np.min([5, len(inp)])
    if ax is None:
        fig, ax = plt.subplots(int(np.ceil(np.shape(inp)[0] / cols)), cols)
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)
    ax = ax.flatten()
    mng = plt.get_current_fig_manager()
    try:
        mng.window.showMaximized() if maximize else None
    except AttributeError:
        print("Tkinter can't maximize. Skipped")
    for idx, image in enumerate(inp):
        image = conver_tensor_to_plot(image, mean, std)
        ax[idx].clear()
        ax[idx].axis('off')
        if len(np.shape(image)) == 2:
            ax[idx].imshow(image, cmap='gray', vmin=0, vmax=1)
        else:
            ax[idx].imshow(image)
        if labels is not None and len(labels) > idx:
            if isinstance(labels[idx], torch.Tensor):
                t = labels[idx].item()
            else:
                t = labels[idx]
            ax[idx].set_title(str(labels[idx]) + ' ' + (title_more[idx] if title_more != '' else ''))
    plt.pause(0.1)
    plt.tight_layout()
    return ax


def interpolate_grid(canvas):
    from scipy import interpolate

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
    if isinstance(values, pd.Series) or isinstance(values, pd.DataFrame):
        x_values = np.array(values.index.get_level_values('transl_X'), dtype=int)
        y_values = np.array(values.index.get_level_values('transl_Y'), dtype=int)
        canvas[y_values, x_values] = values
    else:
        # here we assume that values are relative (e.g. 0.3 of the full canvas)
        canvas[np.array(values[0]).astype(int),
               np.array(values[1]).astype(int)] = values[2]
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
    cmap = 'viridis' if 'cmap' not in kwargs else kwargs['cmap']
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    canvas = compute_density(values, plot_args)
    cm = plt.get_cmap(cmap)
    canvas = cm(canvas)
    im = ax.imshow(canvas, **kwargs)
    ax.set_xlim([0, canvas.shape[0]])
    ax.set_ylim([0, canvas.shape[1]])
    if 'dataset' in list(plot_args.keys()):
        dataset = plot_args['dataset']
        if hasattr(dataset, 'minX'):
            ax.add_patch(Rectangle((dataset.minX, dataset.minY), dataset.maxX-dataset.minX, dataset.maxY-dataset.minY, edgecolor='r', facecolor='none', linewidth=2))
        elif hasattr(dataset, 'translations_range'):
            for groupID, rangeC in dataset.translations_range.items():
                ax.add_patch(Rectangle((rangeC[0], rangeC[2]), rangeC[1] - rangeC[0], rangeC[3] - rangeC[2], edgecolor='r', facecolor='none', linewidth=2))
    return ax, im
##

import time
import sys

term_width = 80
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    def format_time(seconds):
        days = int(seconds / 3600 / 24)
        seconds = seconds - days * 3600 * 24
        hours = int(seconds / 3600)
        seconds = seconds - hours * 3600
        minutes = int(seconds / 60)
        seconds = seconds - minutes * 60
        secondsf = int(seconds)
        seconds = seconds - secondsf
        millis = int(seconds * 1000)

        f = ''
        i = 1
        if days > 0:
            f += str(days) + 'D'
            i += 1
        if hours > 0 and i <= 2:
            f += str(hours) + 'h'
            i += 1
        if minutes > 0 and i <= 2:
            f += str(minutes) + 'm'
            i += 1
        if secondsf > 0 and i <= 2:
            f += str(secondsf) + 's'
            i += 1
        if millis > 0 and i <= 2:
            f += str(millis) + 'ms'
            i += 1
        if f == '':
            f = '0ms'
        return f

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

