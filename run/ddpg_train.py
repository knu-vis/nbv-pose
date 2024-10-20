import argparse
import os
import pprint
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from collections import deque

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
                        help='Modify config options using the command-line',
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
    parser.add_argument('--world', default=False, action='store_true', help='Convert to world coordinates')
    parser.add_argument('--procrustes_align', default=False, action='store_true')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--episodes', default=30000, type=int)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--lr_actor', default=1e-5, type=float)
    parser.add_argument('--lr_critic', default=1e-4, type=float)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--replay_memory_size', default=30000, type=int)
    parser.add_argument('--action_range', default=150, type=int)
    parser.add_argument('--state_dim', default=(15, 8), type=tuple)
    parser.add_argument('--action_dim', default=2, type=int)
    parser.add_argument('--noise', default=1.5, type=float)
    parser.add_argument('--vis_theta', default=30, help='Penalty weight for invisible joint ratio', type=int)
    parser.add_argument('--mpjpe_theta', default=10, help='Penalty/Reward weights for MPJPE', type=int)
    parser.add_argument('--mpjpe_threshold', default=60, help='MPJPE Reward threshold (mm)', type=int)
    parser.add_argument('--pose_buffer_maxlen', default=4, help='Maximum length of the pose buffer', type=int)
    parser.add_argument('--error_2d_theta', default=20, help='Penalty weight for 2D HPE error', type=int)
    parser.add_argument('--option', default='both', help='Policy network input type (both, heatmap, depthmap)', type=str)
    parser.add_argument('--weight_decay', default=0.0, help='weight decay for critic optimizer', type=float)
    parser.add_argument('--lr_decay', default=False, action='store_true', help='learning rate decay')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training')
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = get_default_config()
    update_config(config, args)
    config.freeze()
    
    logger, final_output_dir, tensorboard_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)
    if not os.path.exists(os.path.join(final_output_dir, 'models')):
        os.makedirs(os.path.join(final_output_dir, 'models'))        
    
    # # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config, config.DATASET.ROOT, config.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        camera_type=config.DATASET.CAMERA_TYPE,
    )
    mean_2d, std_2d = train_dataset.mean_2d, train_dataset.std_2d
    mean_3d, std_3d = train_dataset.mean_3d, train_dataset.std_3d
    stats = (mean_2d, std_2d, mean_3d, std_3d)
    if config.DATASET.CAMERA_TYPE == 'hd':
        args.action_range = 140
        args.mpjpe_threshold = 100
    elif config.DATASET.CAMERA_TYPE == 'vga':
        args.action_range = 35
        args.mpjpe_threshold = 150
    
    # Combined HPE model load (HR-Net + Simple-Baseline)
    device = torch.device(f"cuda:{args.gpu_id}" if config.USE_CUDA else "cpu")
    model_hpe = HPE_NET(config, stats, device=device)
    model_hpe = model_hpe.to(device)
        
    env = Panoptic_Env(train_dataset, model_hpe, args)
    agent = DDPGAgent(args.state_dim, args.action_dim, args, device=device, option=args.option, weight_decay=args.weight_decay)
    
    if args.resume:
        agent.actor.load_state_dict(torch.load(f'{final_output_dir}/models/checkpoint_actor_final.pth'))
        agent.critic.load_state_dict(torch.load(f'{final_output_dir}/models/checkpoint_critic_final.pth'))
        agent.actor_target.load_state_dict(torch.load(f'{final_output_dir}/models/checkpoint_actor_target_final.pth'))
        agent.critic_target.load_state_dict(torch.load(f'{final_output_dir}/models/checkpoint_critic_target_final.pth'))
        agent.actor_optimizer.load_state_dict(torch.load(f'{final_output_dir}/models/checkpoint_actor_optimizer.pth'))
        agent.critic_optimizer.load_state_dict(torch.load(f'{final_output_dir}/models/checkpoint_critic_optimizer.pth'))
        
        train_info = pickle.load(open(f'{final_output_dir}/models/train_info.pkl', 'rb'))
        start_episode = train_info['episode']
        args.noise = train_info['noise']
        
        mpjpe_list = pickle.load(open(f'{final_output_dir}/mpjpe_list.pkl', 'rb'))
        epoch_scores = np.load(f'{final_output_dir}/scores.npy').tolist()
    else:
        start_episode = 1
        mpjpe_list = {}
        epoch_scores = []
    
    # for lr decay
    lr_minimum = 1e-6   # critic minimum lr
    
    # Training 
    scores_deque = deque(maxlen=100)
    scores_episodes, mpjpe_episodes = [], []
    epoch_score = 0
    for i_episode in range(start_episode, args.episodes+1):
        mpjpe_episode = []
        state, mpjpe = env.reset()
        mpjpe_episode.append(mpjpe)
        mpjpe_list.setdefault(env.epoch, []).append(mpjpe)
        score = 0
        while True:
            action = agent.select_action(state, noise=args.noise)
            next_state, reward, done, mpjpe = env.step(action[0])
            mpjpe_list.setdefault(env.epoch, []).append(mpjpe)
            mpjpe_episode.append(mpjpe)
            
            agent.memory.push(state[0].cpu(), state[5].cpu(), action, reward, 
                              next_state[0].cpu(), next_state[5].cpu(), done)
                
            state = next_state
            score += reward
            agent.train()
            if done:
                break
        
        epoch_score += score
        scores_deque.append(score)
        scores_episodes.append(score)
        mpjpe_episodes.append(np.mean(mpjpe_episode))
        
        np.save(f'{final_output_dir}/scores_episodes.npy', np.array(scores_episodes))
        np.save(f'{final_output_dir}/mpjpe_episodes.npy', np.array(mpjpe_episodes))
        
        msg = f'Episode {i_episode:05d}\tAverage Score: {np.mean(scores_deque):.2f}  Noise: {args.noise * agent.noise_epsilon:.2f}'
        logger.info(msg)
        if env.end_of_epoch:
            epoch_scores.append(epoch_score)
            epoch_score = 0
            pickle.dump(mpjpe_list, open(f'{final_output_dir}/mpjpe_list.pkl', 'wb'))
            np.save(f'{final_output_dir}/scores.npy', np.array(epoch_scores))
        if i_episode % 100 == 0:
            torch.save(agent.actor.state_dict(), f'{final_output_dir}/models/checkpoint_actor_{i_episode:05d}.pth')
            torch.save(agent.critic.state_dict(), f'{final_output_dir}/models/checkpoint_critic_{i_episode:05d}.pth')
            
        if i_episode % 10 == 0:
            torch.save(agent.actor.state_dict(), f'{final_output_dir}/models/checkpoint_actor_final.pth')
            torch.save(agent.critic.state_dict(), f'{final_output_dir}/models/checkpoint_critic_final.pth')
            torch.save(agent.actor_target.state_dict(), f'{final_output_dir}/models/checkpoint_actor_target_final.pth')
            torch.save(agent.critic_target.state_dict(), f'{final_output_dir}/models/checkpoint_critic_target_final.pth')
            torch.save(agent.actor_optimizer.state_dict(), f'{final_output_dir}/models/checkpoint_actor_optimizer.pth')
            torch.save(agent.critic_optimizer.state_dict(), f'{final_output_dir}/models/checkpoint_critic_optimizer.pth')
            
            args.noise * agent.noise_epsilon
            train_info = {
                'episode': i_episode,
                'noise': args.noise * agent.noise_epsilon
            }
            pickle.dump(train_info, open(f'{final_output_dir}/models/train_info.pkl', 'wb'))

        # # lr decay
        if args.lr_decay and i_episode % 1000 == 0:
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.5, 0.1*lr_minimum)
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.5, lr_minimum)
    
if __name__ == '__main__':
    main()