from models_bank import mask_rcnn
from models_bank.efficient_ps import EfficientPS
import os.path
import json
import constants
from models_bank.fuseblock_2d_3d import FuseNet
import config_kitti


def get_model_by_name(model_name):

    if model_name is "FuseNet":
        return FuseNet(config_kitti.K_NUMBER, config_kitti.N_NUMBER)

    if model_name is "EfficientPS":
        return EfficientPS(config_kitti.BACKBONE,
                            config_kitti.BACKBONE_OUT_CHANNELS,
                            config_kitti.NUM_THING_CLASSES,
                            config_kitti.NUM_STUFF_CLASSES,
                            config_kitti.CROP_OUTPUT_SIZE,
                            config_kitti.MIN_SIZE,
                            config_kitti.MAX_SIZE)
def get_model():
    models_map = {
        "EfficientPS": EfficientPS(config_kitti.BACKBONE, 
            config_kitti.BACKBONE_OUT_CHANNELS, 
            config_kitti.NUM_THING_CLASSES, 
            config_kitti.NUM_STUFF_CLASSES, 
            config_kitti.CROP_OUTPUT_SIZE,
            config_kitti.MIN_SIZE,
            config_kitti.MAX_SIZE)
    }
    return models_map[config_kitti.MODEL]