import copy

import numpy as np
from matplotlib import pyplot as plt
from neptune import new as neptune

import framework_utils
from callbacks import Callback, GenericDataFrameSaver


class PlotUnityImagesEveryOnceInAWhile(Callback):
    counter = 0

    def __init__(self, dataset, plot_every=100, plot_only_n_times=5):
        self.dataset = dataset
        self.plot_every = plot_every
        self.plot_only_n_times = plot_only_n_times

    def on_training_step_end(self, batch, logs=None):
        if logs['tot_iter'] % self.plot_every == self.plot_every - 1 and self.counter < self.plot_only_n_times:
            framework_utils.plot_images_on_weblogger(self.dataset, self.dataset.name_generator, self.dataset.stats,
                                                     images=logs['images'], labels=None, more=None,
                                                     log_text=f"ITER {logs['tot_iter']}")
            self.counter += 1


class SequenceLearning3dDataFrameSaver(GenericDataFrameSaver):
    def __init__(self, k, nSt, nSc, nFt, nFc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.nSt = nSt
        self.nSc = nSc
        self.nFt = nFt
        self.nFc = nFc
        self.additional_logs_names = ['task_num', 'objC', 'objT',  'candidate_campos_XYZ', 'training_campos_XYZ', 'rel_score']
        self.column_names.extend(self.additional_logs_names)
        self.camera_positions_batch = None
        self.is_support = None
        self.task_num = None


    def _get_additional_logs(self, logs, sample_index):
        self.camera_positions_batch = np.array(logs['camera_positions'])
        self.task_num = logs['tot_iter']

        def unity2python(v):
            v = copy.deepcopy(v)
            v.T[[1, 2]] = v.T[[2, 1]]
            return v

        camera_positions_candidates = self.camera_positions_batch[sample_index][:self.nFc * self.nSc].reshape(self.nSc, self.nFc, 3)
        camera_positions_trainings = self.camera_positions_batch[sample_index][self.nFc * self.nSc:].reshape(self.nSt, self.nFt, 3)

        #################################~~~~~~DEBUG~~~~~~###############################################
        # _, self.ax = framework_utils.create_sphere()
        #
        # import matplotlib.pyplot as plt
        # plt.show()
        # import copy
        # def unity2python(v):
        #     v = copy.deepcopy(v)
        #     v.T[[1, 2]] = v.T[[2, 1]]
        #     return v
        #
        # for idx, c in enumerate(self.camera_positions_batch):
        #     if vh1:
        #         # [i.remove() for i in vh1]
        #         # [i.remove() for i in vh2]
        #         vh1 = []
        #         vh2 = []
        #     for i in range(len(self.camera_positions_batch[0]) - 1):
        #         vh2.append(framework_utils.add_norm_vector(unity2python(c[i + 1]), 'r', ax=self.ax))
        #         vh1.append(framework_utils.add_norm_vector(unity2python(c[0]), 'k', ax=self.ax))
        #################################################################
        add_logs = [self.task_num,
                    logs['labels'][sample_index][0].item(), logs['labels'][sample_index][1].item(),
                    np.array([unity2python(i) for i in camera_positions_candidates]), np.array([unity2python(i) for i in camera_positions_trainings]),
                    logs['output'][sample_index].item()]

        return add_logs


class MetaLearning3dDataFrameSaver(GenericDataFrameSaver):
    def _get_additional_logs(self, logs, sample_index):
        # each row is a camera query. It works even for Q>1
        self.camera_positions_batch = np.array(logs['more']['camera_positions'])
        self.task_num = logs['tot_iter']
        additional_logs = [self.task_num, self.camera_positions_batch[self.n*self.k:][sample_index], self.camera_positions_batch[self.n * int(sample_index / self.q):self.n * int(sample_index / self.q) + self.n]]
        return additional_logs

    def _compute_and_log_metrics(self, data_frame):
        plotly_fig, mplt_fig = framework_utils.from_dataframe_to_3D_scatter(data_frame, title=self.log_text_plot)
        metric_str = '3D Sphere'
        if self.weblogger == 1:
            pass
        if isinstance(self.weblogger, neptune.run.Run):
            self.weblogger[metric_str].log(mplt_fig)
            # log_chart(f'{self.log_text_plot} {metric_str}', plotly_fig)
        return data_frame


class TranslationDataFrameSaver(GenericDataFrameSaver):
    def __init__(self, translation_type_str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translation_type_str = translation_type_str
        self.additional_logs_names = ['transl_X', 'transl_Y', 'size_X', 'size_Y', 'rotation', 'tested_area']
        self.index_dataframe.extend(self.additional_logs_names)
        self.face_center_batch = None
        self.size_object_batch = None
        self.rotation_batch = None


    def _get_additional_logs(self, logs, sample_index):
        face_center_batch_t = logs['more']['center']
        size_object_batch_t = logs['more']['size']
        rotation_batch_t = logs['more']['rotation']

        self.face_center_batch = np.array([np.array(i) for i in face_center_batch_t]).transpose()
        self.size_object_batch = np.array([np.array(i) for i in size_object_batch_t]).transpose()
        self.rotation_batch = np.array([np.array(i) for i in rotation_batch_t]).transpose()

        additional_logs = [self.face_center_batch[sample_index][0],
                           self.face_center_batch[sample_index][1],
                           self.size_object_batch[sample_index][0],
                           self.size_object_batch[sample_index][1],
                           self.rotation_batch[sample_index],
                           self.translation_type_str]
        return additional_logs

    def _compute_and_log_metrics(self, data_frame):
        if self.weblogger:
            # Plot Density Translation
            mean_accuracy_translation = data_frame.groupby(['transl_X', 'transl_Y']).mean()['is_correct']
            ax, im = framework_utils.imshow_density(mean_accuracy_translation, plot_args={'interpolate': True, 'size_canvas': self.size_canvas}, vmin=1 / self.num_classes - 1 / self.num_classes * 0.2, vmax=1)
            plt.title(self.log_text_plot)
            fig = ax.figure
            cbar = fig.colorbar(im)
            cbar.set_label('Mean Accuracy (%)', rotation=270, labelpad=25)
            metric_str = 'Density Plot/{}'.format(self.log_text_plot)
            if self.weblogger == 1:
                wandb.log({metric_str: fig})
            if isinstance(self.weblogger, neptune.run.Run):
                self.weblogger[metric_str].log(fig)

            plt.close()

            # Plot Scale Accuracy
            fig, ax = plt.subplots(1, 1)
            mean_accuracy_size_X = data_frame.groupby(['size_X']).mean()['is_correct']  # generally size_X = size_Y so for now we don't bother with both
            x = mean_accuracy_size_X.index.get_level_values('size_X')
            plt.plot(x, mean_accuracy_size_X * 100, 'o-')
            plt.xlabel('Size item (horizontal)')
            plt.ylabel('Mean Accuracy (%)')
            plt.title('size-accuracy')
            print(f'Mean Accuracy Size: {mean_accuracy_size_X} for sizes: {x}')
            metric_str = 'Size Accuracy/{}'.format(self.log_text_plot)
            if self.weblogger == 1:
                wandb.log({metric_str: plt})
            if isinstance(self.weblogger, neptune.run.Run):
                self.weblogger[metric_str].log(fig)
            plt.close()

            # Plot Rotation Accuracy
            fig, ax = plt.subplots(1, 1)
            mean_accuracy_rotation = data_frame.groupby(['rotation']).mean()['is_correct']  # generally size_X = size_Y so for now we don't bother with both
            x = mean_accuracy_rotation.index.get_level_values('rotation')
            plt.plot(x, mean_accuracy_rotation * 100, 'o-')
            plt.xlabel('Rotation item (degree)')
            plt.ylabel('Mean Accuracy (%)')
            plt.title('rotation-accuracy')
            print(f'Mean Accuracy Rotation: {mean_accuracy_rotation} for rotation: {x}')
            # wandb.log({'{}/Rotation Accuracy'.format(self.log_text_plot): plt})
            metric_str = 'Rotation Accuracy/{}'.format(self.log_text_plot)
            if self.weblogger == 1:
                wandb.log({metric_str: plt})
            if isinstance(self.weblogger, neptune.run.Run):
                self.weblogger[metric_str].log(fig)
            plt.close()
            plt.close()
        return data_frame