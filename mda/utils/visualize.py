import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.collections import PatchCollection
import seaborn as sns


class VisualizeMuscle(object):
    def __init__(self, data):
        self.data = data
    
    def show_motion(self, ax=None, muscle_remap=None, adjacent=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        if muscle_remap is None:
            polygons = self.data
        else:
            polygons = self.data[[muscle_remap[k] for k in muscle_remap], :]
        patches = []
        for i, p in enumerate(polygons):
            affine_transform = mtransforms.Affine2D()
            affine_transform.rotate_deg_around(x=p[3], y=p[4], degrees=p[2])
            xy = (p[3] - p[0]/2, p[4] - p[1]/2)
            width = p[0]
            length = p[1]
            patches.append(mpatches.Rectangle(xy, width, length, transform=affine_transform))

        plot_collection = PatchCollection(patches, alpha=0.4)
        colors = 0.1 * np.arange(len(patches))
        plot_collection.set_array(colors)
        ax.add_collection(plot_collection)
        xlim = [polygons[..., 3].min(), polygons[..., 3].max()]
        ylim = [polygons[..., 4].min(), polygons[..., 4].max()]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis('equal')


class VisualizeCurve(object):
    def __init__(self, fake_curves, real_curves, col_wrap=5):
        self.fake_curves = fake_curves
        self.real_curves = real_curves
        self.num_curves = self.fake_curves.shape[0]
        self.time_steps = self.fake_curves.shape[1]
        self.ncols = min(self.fake_curves.shape[0], col_wrap)
        self.nrows = int(np.ceil(self.fake_curves.shape[0] / col_wrap))
    
    def show_curve(self, ax=None):
        if ax is None:
            fig, axes = plt.subplots(self.nrows, self.ncols, 
                                     sharex=True, sharey=False,
                                     figsize=(self.nrows * 3, self.ncols*4))
        # TODO: axes condition when nrows == 1
        x = np.arange(self.time_steps)
        for i in range(self.num_curves):
            axes[i // self.ncols, i % self.ncols].plot(x, self.real_curves[i], label='real')
            axes[i // self.ncols, i % self.ncols].plot(x, self.fake_curves[i], label='fake')
            axes[i // self.ncols, i % self.ncols].legend()
        plt.tight_layout()
        return fig


class VisualizeHeatmap(object):
    def __init__(self, heatmap, col_wrap=5):
        self.heatmap = heatmap
        self.num_heatmap = heatmap.shape[0]
        self.num_nodes = heatmap.shape[1]
        self.ncols = min(self.num_heatmap, col_wrap)
        self.nrows = int(np.ceil(self.num_heatmap / col_wrap))
    
    def show_heatmap(self, ax=None):
        fig, axes = plt.subplots(self.nrows, self.ncols,
                                 figsize=(self.ncols * self.num_nodes // 2,
                                          self.nrows * self.num_nodes // 2))
        for i in range(self.num_heatmap):
            axes[i].set_xticks(np.arange(self.heatmap[i].shape[0]))
            axes[i].set_yticks(np.arange(self.heatmap[i].shape[1]))
            heatmap = self.heatmap[i]
            im = axes[i].imshow(heatmap)
            # TODO: generate the labels for these ticks
            # axes[i].set_xticklabels(x_ticklabels)
            # axes[i].set_yticklabels(y_ticklabels)
            # plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right',
            #          rotation_mode='anchor')
            # for i in range(len(x_ticklabels)):
            #     for j in range(len(y_ticklabels)):
            #         text = axes[i].text(j, i, heatmap[i, j],
            #                             ha='center', va='center', color='w')
            # ax.set_title("Muscle Relation strength")
        fig.tight_layout()
        return fig
        