# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.pose_hrnet
import models.hpe_net

import torch

from .baseline_model import BaselineModel


def get_model_3d(config, num_joints):
    """Get model.

    Args:
        config (yacs.config.CfgNode): Configuration.

    Returns:
        (torch.nn.Module): Model.
    """
    print("Loading model for 3D HPE...")

    model = BaselineModel(
        linear_size=config.MODEL.LINEAR_SIZE,
        num_stages=config.MODEL.NUM_STAGES,
        p_dropout=config.MODEL.DROPOUT_PROB,
        predict_14=config.MODEL.PREDICT_14,
        num_joints=num_joints
    )

    weight_path = config.MODEL.WEIGHT_3D
    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")), strict=False)
        print(f"MODEL_3D Loaded weight from {weight_path}.")

    print("Done!")

    return model


def get_model_2d(config, is_train=False):
    print("Loading model for 2D HPE...")
    
    model = models.pose_hrnet.get_pose_net(config, is_train=is_train)
    
    weight_path = config.MODEL.WEIGHT_2D
    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")), strict=False)
        print(f"MODEL_2D Loaded weight from {weight_path}.")
    
    print("Done!")
    
    return model


def get_hpe_model(config, stats, device_id=0):
    print("Loading Combined HPE model...")
    
    if config.USE_CUDA:
        assert torch.cuda.is_available(), "CUDA is not available."
    if device_id:
        device = torch.device(f"cuda:{device_id}" if config.USE_CUDA else "cpu")    
    else:
        device = torch.device("cuda" if config.USE_CUDA else "cpu")
    
    model = models.hpe_net.HPE_NET(config, stats, device)
    
    model = model.to(device)
    
    print("Done!")
    
    return model
