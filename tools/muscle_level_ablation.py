import os
from re import A
import torch
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from scipy.spatial import procrustes
from tqdm.notebook import tqdm
import __init_paths

from config.st_dynamic_system import config, update_config
from models.st_gcn import STDynamicSystem
from dataset.muscle_sequence import get_muscle_sequences
from utils.visualize import VisualizeMuscle
from multiprocessing import Pool, Manager
from itertools import repeat
from functools import partial

logger = logging.getLogger("MuscleAblation")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def load_model(model_file, config_file):
    params = {
        'cfg': config_file
    }
    update_config(config, params)
    _, adjacency, _ = get_muscle_sequences(config, is_train=False)

    model = STDynamicSystem(config, adjacency)
    model.cuda()
    model.load_checkpoint(model_file)
    return model


def inference_pose(config_file, model, initial_x, subsequent_u):
    params = {
        'cfg': config_file,
    }
    update_config(config, params)

    # normalize the input
    x_std = np.asarray(config.DATASET.X_STD)
    x_mean = np.asarray(config.DATASET.X_MEAN)
    u_std = np.asarray(config.DATASET.U_STD)
    u_mean = np.asarray(config.DATASET.U_MEAN)
    initial_x = (initial_x - x_mean[:, np.newaxis]) / x_std[:, np.newaxis]
    subsequent_u = (subsequent_u - u_mean[:, np.newaxis, np.newaxis]) / u_std[:, np.newaxis, np.newaxis]
    
    initial_x = torch.from_numpy(initial_x.astype(np.float32))
    subsequent_u = torch.from_numpy(subsequent_u.astype(np.float32))
    
    pred = model.predict(initial_x, subsequent_u)
    pred = pred.detach().cpu().numpy()
    pred = pred * x_std[:, np.newaxis, np.newaxis] + x_mean[:, np.newaxis, np.newaxis]
    return pred


def to_cartesian_coords(data):
    """Transform the [length, width, angle, cx, cy] representation to cartesian coordinates
    
    :param data: [V, C]
    """
    polygons = data
    pose = []
    for i, p in enumerate(polygons):
        affine_transform = mtransforms.Affine2D()
        affine_transform.rotate_deg_around(x=p[3], y=p[4], degrees=p[2])
        xy = (p[3] - p[0]/2, p[4] - p[1]/2)
        width = p[0]
        length = p[1]
        rect = mpatches.Rectangle(xy, width, length, transform=affine_transform)
        coords = np.array([rect.get_xy(),
                           [rect.get_x() + rect.get_width(), rect.get_y()],
                           [rect.get_x() + rect.get_width(), rect.get_y() + rect.get_height()],
                           [rect.get_x(), rect.get_y() + rect.get_height()]])
        pose.append(coords)
    return np.array(pose)


def mae(p1, p2):
    """Calculate the pose mse based on their cartesian coordinates
    
    :param p1: [C, V]
    :param p2: [C, V]
    """
    p1_cartesian = to_cartesian_coords(p1)
    p2_cartesian = to_cartesian_coords(p2)
    mtx1, mtx2, disparity = procrustes(p1_cartesian.reshape(-1, 2), p2_cartesian.reshape(-1, 2))
    return disparity


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def muscle_ablation(model_file,
                    config_file,
                    selected_muscle,
                    x_dataset,
                    u_dataset,
                    errs_dict):
    # dict for saving overall error according to the individual muscle ablation
    logger.info("Computing {}-th muscle ablation...".format(selected_muscle))
    overall_errs = []

    model = load_model(model_file, config_file)
    for data_idx, (x_data, u_data) in enumerate(zip(x_dataset, u_dataset)):
        num_steps = len(x_data)
        if num_steps < 48:
            continue
        for s in range(num_steps - 48):
            x = x_data.copy().values[s:s+49, :]
            u = u_data.copy().values[s:s+49, :]
            if selected_muscle != 'No':
                u[:, selected_muscle] = 0.0
            x = np.reshape(x, (len(x), -1, 5)).transpose((2, 0, 1))
            u = u[np.newaxis, :, :]
            x0 = x[:, 0]
            x = x[:, 1:]
            u = u[:, 1:]
            pred = inference_pose(config_file, model, x0, u)
            errs = [mae(p1, p2) for (p1, p2) in zip(pred.transpose(1, 2, 0), x.transpose(1, 2, 0))]
            overall_errs.extend(errs)
    errs_dict[selected_muscle] = np.asarray(overall_errs)

    logger.info("Finish computing {}-th muscle ablation.".format(selected_muscle))
    return np.asarray(overall_errs)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    model_file = './outputs/STDynamicSystem/Calcium2Muscle/ventral/checkpoint_latest.pth'
    config_file = './experiments/st_dynamic_system/calcium2muscle/ventral.yaml'

    logger.info("Begin loading the dataset...")
    # load the dataset
    with open('./data/larva/ventral.pkl', 'rb') as fp:
        ventral_dataset = pickle.load(fp)
    logger.info("Finish loading the dataset.")

    logger.info("Begin extracting the dataset...")
    ventral_calcium_dfs = []
    ventral_pose_dfs = []
    for raw_item in ventral_dataset:
        df = raw_item['data']
        x_cols = [c for c in df.columns if c[2] != 'calcium']
        u_cols = [c for c in df.columns if c[2] == 'calcium']
        ventral_calcium_dfs.append(df[u_cols])
        ventral_pose_dfs.append(df[x_cols])
    logger.info("Finish extracting the dataset.")

    logger.info("Begin compute the muscle ablated errors...")
    num_muscles = ventral_calcium_dfs[0].shape[1]
    manager = Manager()
    ablated_errs_dict = manager.dict()
    muscle_ablation(
        model_file, config_file, 'No', ventral_pose_dfs, ventral_calcium_dfs, errs_dict=ablated_errs_dict
    )

    logger.info("Begin parallel muscle ablated errors computing...")
    with Pool(6) as p:
        p.starmap(muscle_ablation, zip(
            repeat(model_file), 
            repeat(config_file), 
            range(num_muscles), 
            repeat(ventral_pose_dfs), 
            repeat(ventral_calcium_dfs),
            repeat(ablated_errs_dict)
        ))
    logger.info("Finsh computing the muscle ablated errors.")
    
    logger.info("Begin writing errors into file...")
    with open('./results/ventral_individual_muscle_ablation_errs.pkl', 'wb') as fp:
        pickle.dump(dict(ablated_errs_dict), fp)
    logger.info("Finish writing errors.")
