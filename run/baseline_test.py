import argparse
import os
import pprint
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from collections import deque
import re
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms

import _init_paths

from config import update_config
import dataset
import models

# for simple baseline model
from config.default import get_default_config
# from models import get_hpe_model
from models.hpe_net import HPE_NET
from utils.utils import create_logger

from rl_ddpg.panoptic_env_baseline import Panoptic_Env_Baseline

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--gpu_id',
                        help='gpu id for multiprocessing training',
                        type=int,
                        default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    # reinforcement learning
    parser.add_argument('--mode', required=True, default='random', type=str)
    parser.add_argument('--repeat', default=10, type=int)
    parser.add_argument('--vis', default=False, action='store_true', help='Visualize the result')
    parser.add_argument('--world', default=False, action='store_true', help='Convert to world coordinates')
    parser.add_argument('--procrustes_align', default=False, action='store_true')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--episodes', default=2000, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--lr_actor', default=1e-5, type=float)
    parser.add_argument('--lr_critic', default=1e-4, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--replay_memory_size', default=30000, type=int)
    parser.add_argument('--action_range', default=150, type=int)
    parser.add_argument('--state_dim', default=(15, 30), type=tuple)
    parser.add_argument('--action_dim', default=2, type=int)
    parser.add_argument('--vis_theta', default=30, help='Penalty weight for invisible joint ratio', type=int)
    parser.add_argument('--mpjpe_theta', default=10, help='Penalty/Reward weights for MPJPE', type=int)
    parser.add_argument('--mpjpe_threshold', default=60, help='MPJPE Reward threshold (mm)', type=int)
    parser.add_argument('--clockwise', default=True, action='store_true', help='Clockwise rotation')
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = get_default_config()
    update_config(config, args)
    config.freeze()
    
    # # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
    config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        camera_type=config.DATASET.CAMERA_TYPE,
    )
    panoptic_stats = pickle.load(open(os.path.join(config.DATASET.ROOT, 'panoptic_stats.pkl'), 'rb'))[config.DATASET.CAMERA_TYPE]
    mean_2d, std_2d = panoptic_stats['mean_2d'], panoptic_stats['std_2d']
    mean_3d, std_3d = panoptic_stats['mean_3d'], panoptic_stats['std_3d']
    stats = (mean_2d, std_2d, mean_3d, std_3d)
    
    if config.DATASET.CAMERA_TYPE == 'hd':
        args.action_range = 140
    elif config.DATASET.CAMERA_TYPE == 'vga':
        args.action_range = 35
    
    
    # Combined HPE model load (HR-Net + Simple-Baseline)
    device = torch.device(f"cuda:{args.gpu_id}" if config.USE_CUDA else "cpu")
    model_hpe = HPE_NET(config, stats, device=device)
    model_hpe = model_hpe.to(device)
    
    output_dir = os.path.join(config.OUTPUT_DIR, config.DATASET.DATASET, f'Baselines_{config.DATASET.CAMERA_TYPE}', args.mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Mode: {args.mode}")
    for seed in range(args.repeat):
        if args.mode == 'rotation':
            if seed % 2 == 0:
                args.clockwise = True
            else:
                args.clockwise = False
        env = Panoptic_Env_Baseline(valid_dataset, model_hpe, args)
        print(f"Seed: {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        results = {'mpjpes':[], 'cams':[], 'camera_idx':[]}
        end = False
        start_time = time.time()
        while True:   # 1 epoch 동안만 테스트
            if end:
                break
            state, mpjpe = env.reset()
            results['mpjpes'].append(mpjpe)
            results['camera_idx'].append(env.cam_idx)
            while True:
                action = env.select_action(state)
                next_state, _, done, mpjpe = env.step(action[0])
                results['mpjpes'].append(mpjpe)
                results['camera_idx'].append(env.cam_idx)
                
                state = next_state
                if env.end_of_epoch:
                    end = True
                    break
                if done:
                    break
        results['time'] = time.time() - start_time
        pickle.dump(results, open(f'{output_dir}/sup_results_{seed}.pkl', 'wb'))
    
    
if __name__ == '__main__':
    main()