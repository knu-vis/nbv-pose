import numpy as np 
import os
import torch
import pickle
from collections import deque

from evaluators.evaluate_3d import compute_joint_distances, compute_joint_distances_2d
from utils.camera_utils_panoptic import transform_camera_to_world, transform_world_to_camera
from utils.data_utils_panoptic import generate_target_depth, normalize_depth

class Panoptic_Env(object):
    def __init__(self, dataset, hpe_model, args):
        """환경 초기화 및 설정"""
        self.dataset = dataset
        self.index = 0  
        self.epoch = 0  
        self.hpe_model = hpe_model
        self.device = self.hpe_model.device
        self.args = args
        self.state_dim = (15, 8)
        self.action_dim = 2
        self.cam_dim = self.dataset.camera_num
        self.cameras = None
        self.current_cam = {}
        self.cam_distance = 0
        self.num_joints = 15
        self.procrustes_align = args.procrustes_align
        self.world = args.world
        
        self.error_2d_theta = args.error_2d_theta  
        self.vis_theta = args.vis_theta            
        self.mpjpe_theta = args.mpjpe_theta        
        self.mpjpe_threshold = args.mpjpe_threshold
        self.final_sequences = pickle.load(open(os.path.join(dataset.root, 'panoptic_final_sequences.pkl'), 'rb'))[dataset.camera_type][dataset.image_set]
        
        self.end_of_epoch = False 
        if dataset.camera_type == 'hd':
            self.action_interval = 4
            self.width, self.height = 1920, 1080
        elif dataset.camera_type == 'vga':
            self.action_interval = 1
            self.width, self.height = 640, 480
                    
        
    def reset(self):
        """환경 리셋"""
        self.episode_index = 0
        if self.end_of_epoch:
            self.epoch += 1
            self.index = 0
            self.end_of_epoch = False
        
        vis_cams = []
        for idx, joints_2d in enumerate(self.dataset.db[self.index]['joints_2d']):
            x_check = np.bitwise_and(joints_2d[:, 0] >= 0,joints_2d[:, 0] <= self.width - 1)
            y_check = np.bitwise_and(joints_2d[:, 1] >= 0,joints_2d[:, 1] <= self.height - 1)
            joints_vis = np.bitwise_and(x_check, y_check)
            if joints_vis.sum() >= self.num_joints:
                vis_cams.append(idx)
                
        while True:   
            self.cam_idx = np.random.choice(vis_cams)
            self.dataset.set_next_cam(self.cam_idx)
            if not self.dataset[self.index][1]['image_broken']:
                break
            
        self.cameras = self.dataset[self.index][1]['camera']
        self.current_cam = self.cameras[self.cam_idx]
        self.state = self.get_state(self.index)
            
        _, mpjpe = self._compute_reward(self.state[1], self.state[2], self.state[3])
        self.index += 1
        
        return self.state, mpjpe
    
    def step(self, action):
        """action에 따른 state 변화 및 reward 계산"""
        if self.episode_index % self.action_interval == 0: 
            self._apply_action(self.state, action)
        
        self.state = self.get_state(self.index)
        _, preds_2d, preds_3d, meta, _, _ = self.state
        reward, mpjpe = self._compute_reward(preds_2d, preds_3d, meta)
        done = self._check_done(self.state[3]['image'])
        
        self.index += 1
        self.episode_index += 1
        if self.index == len(self.dataset):  
            self.end_of_epoch = True
        
        return self.state, reward, done, mpjpe
    
    
    def get_state(self, index, reset=False):
        """현재 state 반환"""
        image, meta = self.dataset[index]
        heatmap, preds_2d, preds_3d, maxvals = self.hpe_model(image.unsqueeze(0).to(self.device), no_grad=True)
        
        preds_depth = normalize_depth(preds_3d[0, :, 2]).reshape(-1, 1)
        depthmap = generate_target_depth(preds_2d[0], np.array([self.width, self.height]), depth=preds_depth)
        depthmap = torch.tensor(depthmap, dtype=torch.float32).to(self.device).unsqueeze(0)

        return heatmap, preds_2d, preds_3d, meta, maxvals, depthmap
    
    
    def _apply_action(self, state, action):
        """action에 따른 state 변화 적용"""
        move_vector = np.array([[action[0]], [action[1]], [0]])
        next_T = self.current_cam['T'] + move_vector
        next_position = -np.dot(self.current_cam['R'].T, next_T)
        
        broken_cameras = []   
        while True:
            min_dist = 100000
            for i, cam in enumerate(self.cameras):
                if i in broken_cameras:
                    continue
                dist = np.linalg.norm(cam['position'] - next_position)
                if dist < min_dist:
                    min_dist = dist
                    next_cam_idx = i
                    
            next_cam_position = -np.dot(self.cameras[next_cam_idx]['R'].T, self.cameras[next_cam_idx]['T'])
            current_cam_position = -np.dot(self.current_cam['R'].T, self.current_cam['T'])
            self.cam_distance = np.linalg.norm(current_cam_position - next_cam_position)
            
            self.cam_idx = next_cam_idx
            self.dataset.set_next_cam(self.cam_idx)
            self.current_cam = self.cameras[self.cam_idx]
            
            if self.dataset[self.index][1]['image_broken']: 
                broken_cameras.append(self.cam_idx)
            else:
                break
    
    
    def _compute_reward(self, preds_2d, preds_3d, meta):
        """state와 action에 따른 reward 계산"""
        
        mpjpe = compute_joint_distances(preds_3d, meta['joints_3d'], procrustes=self.procrustes_align).mean() * 10.0
        num_outside = self.num_out_of_frame_joints(meta['joints_2d'], meta['center'])
        
        person_size = self.compute_person_size(meta['joints_2d'])
        error_2d = compute_joint_distances_2d(preds_2d, meta['joints_2d'].reshape(1, -1, 2))[0]
        failed_pose_2d = sum(error_2d > person_size/10)
        
        error_2d_penalty = - (failed_pose_2d / self.num_joints) * self.error_2d_theta
        joint_vis_penalty = - (num_outside / self.num_joints) * self.vis_theta
        mpjpe_penalty = (1 - (mpjpe / self.mpjpe_threshold)) * self.mpjpe_theta
        
        reward = mpjpe_penalty + joint_vis_penalty
        
        return reward, mpjpe
    
    
    def _check_done(self, image_path):
        """state에 따른 done 여부 확인"""
        scene_name = '/'.join(image_path.split('/')[:3])
        frame_num = os.path.basename(image_path).split('.')[0].split('_')[-1]
        if (scene_name, frame_num) in self.final_sequences:
            done = True
        else:
            done = False
        
        return done
    
    
    def num_out_of_frame_joints(self, joints_2d, center):
        """프레임 밖으로 벗어난 2D 관절 개수 반환"""
        width, height = center * 2
        out_of_frame_count = 0 
        for x, y in joints_2d:
            if x < 0 or x > width or y < 0 or y > height:
                out_of_frame_count += 1
        
        return out_of_frame_count
    
    def compute_person_size(self, joints_2d):
        # joints_2d: [n_joints, 2]
        x_min, y_min = np.min(joints_2d, axis=0)
        x_max, y_max = np.max(joints_2d, axis=0)
        box_width = x_max - x_min
        box_height = y_max - y_min
        
        return max(box_width, box_height)