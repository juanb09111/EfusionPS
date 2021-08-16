# from models_bank import mask_rcnn
# from models_bank.efficient_ps import EfficientPS
# from models_bank.efficient_ps_no_instance import EfficientPS as EfficientPS_no_instance
import os.path
import json
import constants
# from models_bank.fuseblock_2d_3d import FuseNet
# from models_bank.efusion_ps import EfusionPS
# from models_bank.efusion_ps_no_instance import EfusionPS as EfusionPS_no_instance
# from models_bank.efusion_ps_v2 import EfusionPS as EfusionPS_V2
from models_bank.efusion_ps_v3_instance import EfusionPS as EfusionPS_V3
# from models_bank.efusion_ps_v3_depth_head import EfusionPS as EfusionPS_V3_depth
from models_bank.efusion_ps_v3_depth_head_2 import EfusionPS as EfusionPS_V3_depth
from models_bank.efficient_ps_full import EfficientPS_Full
import config_kitti


def get_model_by_name(model_name):

    # if model_name is "FuseNet":
    #     return FuseNet(config_kitti.K_NUMBER, config_kitti.N_NUMBER)
    # if model_name is "EfusionPS":
    #     return EfusionPS(config_kitti.K_NUMBER,
    #                      config_kitti.NUM_THING_CLASSES,
    #                      config_kitti.NUM_STUFF_CLASSES,
    #                      config_kitti.CROP_OUTPUT_SIZE,
    #                      min_size=config_kitti.MIN_SIZE,
    #                      max_size=config_kitti.MAX_SIZE,
    #                      n_number=config_kitti.N_NUMBER)

    # if model_name is "EfusionPS_V2":
    #     return EfusionPS_V2(config_kitti.BACKBONE,
    #                      config_kitti.BACKBONE_OUT_CHANNELS,
    #                      config_kitti.K_NUMBER,
    #                      config_kitti.NUM_THING_CLASSES,
    #                      config_kitti.NUM_STUFF_CLASSES,
    #                      config_kitti.CROP_OUTPUT_SIZE,
    #                      min_size=config_kitti.MIN_SIZE,
    #                      max_size=config_kitti.MAX_SIZE,
    #                      n_number=config_kitti.N_NUMBER)
                         
    if model_name is "EfusionPS_V3":
        return EfusionPS_V3(config_kitti.BACKBONE,
                         config_kitti.BACKBONE_OUT_CHANNELS,
                         config_kitti.K_NUMBER,
                         config_kitti.NUM_THING_CLASSES,
                         config_kitti.NUM_STUFF_CLASSES,
                         config_kitti.CROP_OUTPUT_SIZE,
                         min_size=config_kitti.MIN_SIZE,
                         max_size=config_kitti.MAX_SIZE,
                         n_number=config_kitti.N_NUMBER)

    
    if model_name is "EfusionPS_V3_depth":
        return EfusionPS_V3_depth(config_kitti.BACKBONE,
                         config_kitti.BACKBONE_OUT_CHANNELS,
                         config_kitti.K_NUMBER,
                         config_kitti.NUM_THING_CLASSES,
                         config_kitti.NUM_STUFF_CLASSES,
                         config_kitti.CROP_OUTPUT_SIZE,
                         min_size=config_kitti.MIN_SIZE,
                         max_size=config_kitti.MAX_SIZE,
                         n_number=config_kitti.N_NUMBER)

    if model_name is "EfficientPS_Full":
        return EfficientPS_Full(config_kitti.BACKBONE,
                         config_kitti.BACKBONE_OUT_CHANNELS,
                         config_kitti.K_NUMBER,
                         config_kitti.NUM_THING_CLASSES,
                         config_kitti.NUM_STUFF_CLASSES,
                         config_kitti.CROP_OUTPUT_SIZE,
                         min_size=config_kitti.MIN_SIZE,
                         max_size=config_kitti.MAX_SIZE,
                         n_number=config_kitti.N_NUMBER)

    # if model_name is "EfusionPS_no_instance":
    #     return EfusionPS_no_instance(config_kitti.K_NUMBER,
    #                                  config_kitti.NUM_THING_CLASSES,
    #                                  config_kitti.NUM_STUFF_CLASSES,
    #                                  config_kitti.CROP_OUTPUT_SIZE,
    #                                  min_size=config_kitti.MIN_SIZE,
    #                                  max_size=config_kitti.MAX_SIZE,
    #                                  n_number=config_kitti.N_NUMBER)

    # if model_name is "EfficientPS_no_instance":
    #     return EfficientPS_no_instance(config_kitti.BACKBONE,
    #                                    config_kitti.BACKBONE_OUT_CHANNELS,
    #                                    config_kitti.NUM_THING_CLASSES,
    #                                    config_kitti.NUM_STUFF_CLASSES,
    #                                    config_kitti.CROP_OUTPUT_SIZE,
    #                                    config_kitti.MIN_SIZE,
    #                                    config_kitti.MAX_SIZE)

    # if model_name is "EfficientPS":
    #     return EfficientPS(config_kitti.BACKBONE,
    #                        config_kitti.BACKBONE_OUT_CHANNELS,
    #                        config_kitti.NUM_THING_CLASSES,
    #                        config_kitti.NUM_STUFF_CLASSES,
    #                        config_kitti.CROP_OUTPUT_SIZE,
    #                        config_kitti.MIN_SIZE,
    #                        config_kitti.MAX_SIZE)


# def get_model():
#     models_map = {
#         "EfficientPS": EfficientPS(config_kitti.BACKBONE,
#                                    config_kitti.BACKBONE_OUT_CHANNELS,
#                                    config_kitti.NUM_THING_CLASSES,
#                                    config_kitti.NUM_STUFF_CLASSES,
#                                    config_kitti.CROP_OUTPUT_SIZE,
#                                    config_kitti.MIN_SIZE,
#                                    config_kitti.MAX_SIZE)
#     }
#     return models_map[config_kitti.MODEL]
