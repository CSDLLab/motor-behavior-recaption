import json
import logging
import os
import pathlib
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.collections import PatchCollection
import numpy as np
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_logger(cfg, cfg_name, phase='train'):
    """Create log and checkpoint output directories
    :param cfg: Config
    :type cfg: object
    :param cfg_name: config name
    :type cfg_name: str
    :param phase: Be ``train`` or ``test``, defaults to 'train'
    :type phase: str, optional
    """
    root_output_dir = Path(cfg.OUTPUT_DIR)
    
    # Set up the logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace('->', '2')
    dataset = dataset.replace('/', '-')
    model = cfg.MODEL.NAME
    cfg_name = Path(cfg_name).stem

    # Choose the dirname naming style
    final_output_dir = root_output_dir / model / dataset / cfg_name
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger(cfg_name)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)
    file_stream = logging.FileHandler(final_log_file)
    logger.addHandler(file_stream)

    # Create logger file
    dataset_root_name = Path(cfg.DATASET.ROOT).stem
    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset_root_name / dataset / model / (cfg_name + '_' + phase + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def save_to_video(tensor, filename, fps=20):
    """Create video from tensor array and save to filename

    :param tensor: A sequence of pose, NxCxTxV shape
    :type tensor: np.ndarray 
    :param filename: Filename of output video file
    :type filename: str
    """
    tensor = tensor.transpose((0, 2, 3, 1))
    batch_size = tensor.shape[0]
    time_steps = tensor.shape[1]
    xlim = [tensor[..., 3].min(), tensor[..., 3].max()]
    ylim = [tensor[..., 4].min(), tensor[..., 4].max()]
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    def make_frame(t):
        ax.clear()
        muscles = seq[int(t * fps)]
        patches = []
        for rect in muscles:
            affine_transform = mtransforms.Affine2D()
            affine_transform.rotate_deg_around(x=rect[3], y=rect[4], degrees=rect[2])
            xy = (rect[3] - rect[0]/2, rect[4] - rect[1]/2)
            width = rect[0]
            height = rect[1]
            patches.append(mpatches.Rectangle(xy, width, height, transform=affine_transform))
        plot_collection = PatchCollection(patches, alpha=0.4)
        colors = 0.1 * np.arange(len(patches))
        plot_collection.set_array(colors)
        ax.add_collection(plot_collection)
        ax.set_title('ID: {:.2f}'.format(t * fps))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()
        return mplfig_to_npimage(fig)
    
    for i, seq in enumerate(tensor):
        animation = VideoClip(make_frame, duration=time_steps / fps)
        animation.write_videofile(f'{filename}-{i}.mp4', fps=time_steps)
