from models_bank import mask_rcnn
from models_bank.efficient_ps import EfficientPS
import os.path
import json
import constants
def get_model_by_name(model_name):

    if model_name is "FuseNet":
        return FuseNet(config_kitti.K_NUMBER, config_kitti.N_NUMBER)

import config_kitti
from models_bank.fuseblock_2d_3d import FuseNet
    if model_name is "EfficientPS":
        return EfficientPS2(config_kitti.BACKBONE,
                            config_kitti.BACKBONE_OUT_CHANNELS,
                            config_kitti.NUM_THING_CLASSES,
                            config_kitti.NUM_STUFF_CLASSES,
                            config_kitti.CROP_OUTPUT_SIZE,
                            config_kitti.MIN_SIZE,
                            config_kitti.MAX_SIZE)
def get_model():
    models_map = {
        "EfficientPS": EfficientPS(config.BACKBONE, 
            config.BACKBONE_OUT_CHANNELS, 
            config.NUM_THING_CLASSES, 
            config.NUM_STUFF_CLASSES, 
            config.ORIGINAL_INPUT_SIZE_HW,
            config.MIN_SIZE,
            config.MAX_SIZE)
    }
    return models_map[config.MODEL]