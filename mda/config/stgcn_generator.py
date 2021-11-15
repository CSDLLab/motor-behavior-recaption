from yacs.config import CfgNode

_C = CfgNode()

# BASIC
_C.OUTPUT_DIR = 'outputs'
_C.LOG_DIR = 'logs'
_C.WORKERS = 4
_C.SAVE_CHECKPOINT = False
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.USE_APEX = False
_C.PRINT_FREQ = 200

# CUDNN
_C.CUDNN = CfgNode()
_C.CUDNN.ENABLE = True
_C.CUDNN.DETERMINISTIC = True
_C.CUDNN.BENCHMARK = True

# Dataset
_C.DATASET = CfgNode()
_C.DATASET.DATASET = 'Calcium2Muscle'
_C.DATASET.ROOT = 'data/larva/muscle_sequenece/ventral'
_C.DATASET.TIME_STEPS = 16
_C.DATASET.DILATION = 3
_C.DATASET.X_CHANNELS = 5
_C.DATASET.X_STD = [ 17.71321356, 1.44263055, 107.53236155, 148.91354573, 77.56978701 ]
_C.DATASET.X_MEAN = [ 49.36877278, 6.58087856, 6.59548827, 578.32888227, 321.56623788 ]
_C.DATASET.U_CHANNELS = 1
_C.DATASET.U_STD = [49.59902561]
_C.DATASET.U_MEAN = [24.38892674]
_C.DATASET.Z_CHANNELS = 1
_C.DATASET.NUM_NODES = 38

# MODEL
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'STGCNGenerator'
_C.MODEL.BN = False
_C.MODEL.ST_KERNEL_G = [3, 5]
_C.MODEL.ST_KERNEL_D = [3, 5]
_C.MODEL.LAMBDA_RECONSTRUCTION = 0.1
_C.MODEL.LAMBDA_ANGLE = 1.0

# Train
_C.TRAIN = CfgNode()
_C.TRAIN.TEACHER_FORCE_RATIO = 0.5
_C.TRAIN.SEED = 2021
_C.TRAIN.BATCH_SIZE_PER_GPU = 4
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 20000
_C.TRAIN.OPTIM = 'RMSprop'
_C.TRAIN.LR = 0.0001
_C.TRAIN.WEIGHT_DECAY = 0.001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV = True
_C.TRAIN.BETA1 = 0.9
_C.TRAIN.BETA2 = 0.99
_C.TRAIN.CLIP_VALUE = 0.1
_C.TRAIN.NOISE_LEVEL = 0.01
_C.TRAIN.MAX_NUM_SHOW = 2

# VAL
_C.VAL = CfgNode()
_C.VAL.BATCH_SIZE_PER_GPU = 1

# DEBUG
_C.DEBUG = CfgNode()
_C.DEBUG.DEBUG = False

config = _C


def update_config(cfg, args):
    if 'cfg' in args:
        cfg.merge_from_file(args['cfg'])
    arg_list = []
    for key in args:
        if key != 'cfg':
            arg_list.append(key)
            arg_list.append(args[key])
    cfg.merge_from_list(arg_list)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as fp:
        print(config, file=fp)
