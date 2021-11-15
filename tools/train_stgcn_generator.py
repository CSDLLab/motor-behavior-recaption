import os
import argparse
import pprint
import shutil
import tqdm
from numpy import arange
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import __init_paths
from config.stgcn_generator import config, update_config
from models.stgcn_generator import STGCNGenerator
from dataset.muscle_sequence import get_muscle_sequences
from utils.utils import create_logger, save_to_video


def train(params: dict):
    """General training scripts
    :param params: The custom training config
    :type params: dict
    """
    update_config(config, params)

    logger, final_out_dir, tb_log_dir = create_logger(
        config, params['cfg'], 'train'
    )
    logger.info(pprint.pformat(params))
    logger.info(config)

    torch.cuda.manual_seed(config.TRAIN.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enable = config.CUDNN.ENABLE

    if config.SAVE_CHECKPOINT:
        this_dir = os.path.dirname(__file__)
        shutil.copy(
            os.path.join(this_dir, '../mda/models/stgcn_generator.py'),
            final_out_dir
        )
        shutil.copy(
            os.path.abspath(__file__),
            final_out_dir
        )
    
    # Set dataloader
    if 'Calcium2Muscle' in config.DATASET.DATASET:
        train_loader, adjacency, norm_params = get_muscle_sequences(config, is_train=True)
        val_loader, adjacency, norm_params = get_muscle_sequences(config, is_train=False)
    elif 'Muscle2Calcium' in config.DATASET.DATASET:
        train_loader, adjacency, norm_params = get_muscle_sequences(config, is_train=True, reverse=True)
        val_loader, adjacency, norm_params = get_muscle_sequences(config, is_train=False, reverse=True)
    update_config(config, norm_params)

    # Set model
    if config.MODEL.NAME == 'STGCNGenerator':
        model = STGCNGenerator(config, adjacency)
    else:
        raise NotImplementedError()
    model.set_optimizer(config.TRAIN.OPTIM, lr=config.TRAIN.LR)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        model.data_parallel()
    elif len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) != 1:
        model.data_parallel()
    model.cuda()
    
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_step': 0
    }

    for epoch in tqdm.tqdm(range(config.TRAIN.END_EPOCH), 
                           desc="Epoch", ncols=80):
        model.train(epoch, train_loader, writer_dict)

        if (epoch+1) % config.PRINT_FREQ == 0:
            if config.SAVE_CHECKPOINT:
                model.save_checkpoint(epoch, '{}/checkpoint_latest.pth'.format(final_out_dir, epoch))
            # generated_tensor = model.eval(epoch, val_loader, writer_dict)
            # save_to_video(generated_tensor, f'{final_out_dir}/epoch-{epoch}')
    
    writer_dict['writer'].close()


if __name__ == '__main__':
    from pathlib import Path

    parser = argparse.ArgumentParser('Train Template machine learning project')
    parser.add_argument('--cfg', help='experiment configuration filepath',
                        default='experiments/lstm_dynamic_system/calcium2muscle/config.yaml',
                        type=str)
    parser.add_argument('--dataset', help='Train dataset root path',
                        type=str, required=False)
    shortcut = {
        'cfg': 'cfg',
        'dataset': 'DATASET.ROOT',
    }
    argparams = parser.parse_args()
    argparams = vars(argparams)
    params = {}
    for key in argparams:
        if argparams[key] is not None:
            params[shortcut[key]] = argparams[key]
    train(params)
