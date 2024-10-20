import os
import numpy as np

import torch
import torch.nn as nn

from core.inference_2d import get_final_preds
from utils.transforms import flip_back, get_scale
from utils.camera_utils_panoptic import transform_camera_to_world

from lib.models import get_model_2d
from lib.models import get_model_3d

class HPE_NET(nn.Module):
    def __init__(self, config, stats, device):
        super(HPE_NET, self).__init__()
        self.config = config
        self.device = device
        self.output_3d_world = config.MODEL.OUTPUT_3D_WORLD
        
        self.flip_pairs = [[14, 8], [13, 7], [12, 6], [11, 5], [10, 4], [9, 3]]
        self.image_size = config.MODEL.IMAGE_SIZE
        if config.DATASET.CAMERA_TYPE == 'hd':
            self.center = [960., 540.]
            self.scale = [9.6, 7.2]
        elif config.DATASET.CAMERA_TYPE == 'vga':
            self.center = [320., 240.]
            self.scale = [3.2, 2.4]
        
        self.mean_2d, self.std_2d = stats[0], stats[1]
        self.mean_3d, self.std_3d = stats[2], stats[3]

        self.model_2d = get_model_2d(config, is_train=False).to(self.device)
        self.model_3d = get_model_3d(config, num_joints=self.config.MODEL.NUM_JOINTS).to(self.device)
        self.model_2d.eval()
        self.model_3d.eval()

    
    def forward(self, input, gt2d=None, no_grad=True, R=None, T=None):
        # 2D HPE
        if no_grad:
            with torch.no_grad():
                if gt2d is not None:    
                    preds_2d = gt2d
                    heatmap = None
                    maxvals = None
                else:
                    heatmaps = self.model_2d(input)
                    if isinstance(heatmaps, list):
                        heatmap = heatmaps[-1]
                    else:
                        heatmap = heatmaps
                    preds_2d, maxvals = get_final_preds(self.config, heatmap.clone().cpu().numpy(),
                                                        np.repeat([self.center], heatmap.shape[0], axis=0),
                                                        np.repeat([self.scale], heatmap.shape[0], axis=0))
                
                # 3D HPE
                normalized_preds = self.normalize_pose(preds_2d, self.mean_2d, self.std_2d)
                preds_3d = self.model_3d(torch.tensor(normalized_preds.reshape(-1, self.config.MODEL.NUM_JOINTS*2)).to(self.device))
                preds_3d = preds_3d.detach().cpu().numpy().reshape(-1, self.config.MODEL.NUM_JOINTS, 3)
                preds_3d = self.unnormalize_pose(preds_3d, self.mean_3d, self.std_3d)
                if self.output_3d_world:
                    preds_3d = self.camera_to_world(preds_3d, R, T)
        else:
            if gt2d is not None:
                preds_2d = gt2d
                heatmap = None
                maxvals = None
            else:
                heatmaps = self.model_2d(input)
                if isinstance(heatmaps, list):
                    heatmap = heatmaps[-1]
                else:
                    heatmap = heatmaps
                preds_2d, maxvals = get_final_preds(self.config, heatmap.clone().cpu().numpy(),
                                                    np.repeat([self.center], heatmap.shape[0], axis=0),
                                                    np.repeat([self.scale], heatmap.shape[0], axis=0))
            
            # 3D HPE
            normalized_preds = self.normalize_pose(preds_2d, self.mean_2d, self.std_2d)
            preds_3d = self.model_3d(torch.tensor(normalized_preds.reshape(-1, self.config.MODEL.NUM_JOINTS*2)).to(self.device))
            preds_3d = preds_3d.detach().cpu().numpy().reshape(-1, self.config.MODEL.NUM_JOINTS, 3)
            preds_3d = self.unnormalize_pose(preds_3d, self.mean_3d, self.std_3d)
            if self.output_3d_world:
                preds_3d = self.camera_to_world(preds_3d, R, T)

        return heatmap, preds_2d, preds_3d, maxvals
    
    # pose normalization
    def normalize_pose(self, pose, mean, std):
        return np.divide((pose - mean), std+1e-8)
    
    # unnormalize pose
    def unnormalize_pose(self, pose, mean, std):
        return np.multiply(pose, std) + mean
    
    def camera_to_world(self, pose_set, R, T):
        '''
        pose_set: (N, 15, 3)
        
        return: (N, 15, 3)
        '''
        t3d_world = []
        for i in range(len(pose_set)):
            t3d_camera = pose_set[i]
            t3d_camera = t3d_camera.reshape((-1, 3))
            
            world_coord = transform_camera_to_world(t3d_camera, R[i], T[i])
            world_coord = world_coord.reshape((self.config.MODEL.NUM_JOINTS, 3))
            
            t3d_world.append(world_coord)
            
        return torch.stack(t3d_world)