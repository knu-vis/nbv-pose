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
from utils.nbv_visualization import save_image, image2video

from rl_ddpg.ddpg_agent import DDPGAgent
from rl_ddpg.panoptic_env import Panoptic_Env

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
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--test_episode', default=0, help='0: test all episodes, n: test n-th episode', type=int)
    parser.add_argument('--vis', default=False, action='store_true', help='Visualize the result')
    parser.add_argument('--vis_only', default=False, action='store_true', help='Visualize the result')
    parser.add_argument('--world', default=False, action='store_true', help='Convert to world coordinates')
    parser.add_argument('--procrustes_align', default=False, action='store_true')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--episodes', default=30000, type=int)
    parser.add_argument('--gamma', default=0.9, type=float)
    parser.add_argument('--lr_actor', default=1e-5, type=float)
    parser.add_argument('--lr_critic', default=1e-4, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--replay_memory_size', default=30000, type=int)
    parser.add_argument('--action_range', default=150, type=int)
    parser.add_argument('--state_dim', default=(15, 8), type=tuple)
    parser.add_argument('--action_dim', default=2, type=int)
    parser.add_argument('--vis_theta', default=30, help='Penalty weight for invisible joint ratio', type=int)
    parser.add_argument('--mpjpe_theta', default=10, help='Penalty/Reward weights for MPJPE', type=int)
    parser.add_argument('--mpjpe_threshold', default=60, help='MPJPE Reward threshold (mm)', type=int)
    parser.add_argument('--pose_buffer_maxlen', default=8, help='Maximum length of the pose buffer', type=int)
    parser.add_argument('--error_2d_theta', default=20, help='Penalty weight for 2D HPE error', type=int)
    parser.add_argument('--option', default='both', help='Policy network input type (both, heatmap, depthmap)', type=str)
    
    args = parser.parse_args()

    return args

def find_max_episode(output_dir, start_string='checkpoint_actor'):
    max_episode = 0
    for file_name in sorted(os.listdir(output_dir)):
        if file_name.startswith(start_string):
            temp = re.findall(r'\d+', file_name)
            if temp:
                episode_num = int(re.findall(r'\d+', file_name)[0])
            else:
                continue
            if episode_num > max_episode:
                max_episode = episode_num
    return max_episode

def find_best_episode(output_dir):
    best_mpjpe = 100000
    for file_name in sorted(os.listdir(output_dir)):
        if not file_name.startswith('results'):
            continue
        episode_num = int(re.findall(r'\d+', file_name)[0])
        results = pickle.load(open(os.path.join(output_dir, file_name), 'rb'))
        test_mpjpe = np.mean(results['mpjpes'])
        if test_mpjpe < best_mpjpe:
            best_mpjpe = test_mpjpe
            best_episode = episode_num
    return best_episode

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
    if config.DATASET.CAMERA_TYPE == 'hd':
        args.action_range = 140
    elif config.DATASET.CAMERA_TYPE == 'vga':
        args.action_range = 35
    
    panoptic_stats = pickle.load(open(os.path.join(config.DATASET.ROOT, 'panoptic_stats.pkl'), 'rb'))[config.DATASET.CAMERA_TYPE]
    mean_2d, std_2d = panoptic_stats['mean_2d'], panoptic_stats['std_2d']
    mean_3d, std_3d = panoptic_stats['mean_3d'], panoptic_stats['std_3d']
    stats = (mean_2d, std_2d, mean_3d, std_3d)
    
    # Combined HPE model load (HR-Net + Simple-Baseline)
    device = torch.device(f"cuda:{args.gpu_id}" if config.USE_CUDA else "cpu")
    model_hpe = HPE_NET(config, stats, device=device)
    model_hpe = model_hpe.to(device)
    
    output_dir = os.path.join(args.output_dir, 'test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.vis:
        vis_output_dir = os.path.join(output_dir, 'vis_image')
        if not os.path.exists(vis_output_dir):
            os.makedirs(vis_output_dir)
    
    
    if not args.vis_only:
        if args.test_episode > 0:  
            start_episode, end_episode = args.test_episode, args.test_episode
        else:       
            end_episode = find_max_episode(os.path.join(args.output_dir, 'models'))  
            start_episode = find_max_episode(output_dir, start_string='results') + 100
        if end_episode == 0:
            print('No checkpoint found')
            raise ValueError('No checkpoint found')
        for trained_episode in range(start_episode, end_episode+1, 100):
            actor_path = os.path.join(args.output_dir, 'models', f'checkpoint_actor_{trained_episode:05d}.pth')
            env = Panoptic_Env(valid_dataset, model_hpe, args)
            agent = DDPGAgent(args.state_dim, args.action_dim, args, device=device, option=args.option)
            agent.actor.load_state_dict(torch.load(actor_path))
            agent.actor.eval()
            
            results = {'mpjpes':[], 'scores':[], 'action':[], 'camera_idx':[]}
            end = False
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            while True:   # test during 1 epoch
                if end:
                    break
                state, mpjpe = env.reset()
                results['mpjpes'].append(mpjpe)
                results['camera_idx'].append(env.cam_idx)
                score = 0
                while True:
                    action = agent.select_action(state, noise=0.0)  # set noise-0 during inference
                    next_state, reward, done, mpjpe = env.step(action[0])
                    
                    results['action'].append(action[0])
                    results['mpjpes'].append(mpjpe)
                    results['camera_idx'].append(env.cam_idx)
                    
                    state = next_state
                    score += reward
                    
                    if env.end_of_epoch:
                        end = True
                        break
                    if done:
                        break
                results['scores'].append(score)
            end_time.record()
            torch.cuda.synchronize()
            results['time'] = (start_time.elapsed_time(end_time) / 1000) / len(env.dataset)
            
            pickle.dump(results, open(f'{output_dir}/results_{trained_episode:05d}.pkl', 'wb'))
            print(f'Episode [{trained_episode:05d}/{end_episode:05d}]\tDone')
    
    # Visualize the result
    if args.vis:
        print('Visualizing the result...')
        if args.test_episode > 0:
            best_episode = args.test_episode
        else:
            best_episode = find_best_episode(output_dir)   
        video_path = os.path.join(output_dir, f'nbv_video_test_{best_episode:05d}.mp4')
        if not os.path.exists(video_path):
            actor_path = os.path.join(args.output_dir, 'models', f'checkpoint_actor_{best_episode:05d}.pth')
            env = Panoptic_Env(valid_dataset, model_hpe, args)
            agent = DDPGAgent(args.state_dim, args.action_dim, args, device=device, option=args.option)
            agent.actor.load_state_dict(torch.load(actor_path))
            agent.actor.eval()
            end = False
            start_time = time.time()
            while True:   # test during 1 epoch
                if end:
                    break
                state, mpjpe = env.reset()
                save_image(state[2], state[3]['joints_3d'].reshape(-1, 15, 3), state[3]['pelvis'], state[3]['camera'][env.cam_idx],
                        vis_output_dir, mpjpe, env.cam_idx, env.index, env.episode_index, env.dataset[env.index][1]['image'], world=args.world)
                while True:
                    action = agent.select_action(state, noise=0.0)  # set noise-0 during inference 
                    next_state, reward, done, mpjpe = env.step(action[0])
                    
                    state = next_state
                    
                    save_image(state[2], state[3]['joints_3d'].reshape(-1, 15, 3), state[3]['pelvis'], state[3]['camera'][env.cam_idx],
                            vis_output_dir, mpjpe, env.cam_idx, env.index, env.episode_index, env.dataset[env.index][1]['image'], world=args.world)
                    if env.end_of_epoch:
                        end = True
                        break
                    if done:
                        break
            image2video(vis_output_dir, video_path, fps=10) 
        
    
if __name__ == '__main__':
    main()