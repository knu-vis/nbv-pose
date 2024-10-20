import numpy as np 
import os
import torch
import pickle

from evaluators.evaluate_3d import compute_joint_distances

class Panoptic_Env_Baseline(object):
    def __init__(self, dataset, hpe_model, args):
        """환경 초기화 및 설정"""
        self.mode = args.mode
        if args.mode == 'rotation':
            self.clockwise = args.clockwise
        self.dataset = dataset
        self.index = 0   
        self.epoch = 0   
        self.hpe_model = hpe_model
        self.device = self.hpe_model.device
        self.args = args
        self.state_dim = (15, 72, 96)
        self.action_dim = 2
        self.cam_dim = 31
        self.cameras = None
        self.current_cam = {}
        self.cam_distance = 0
        self.num_joints = 15
        self.action_range = args.action_range
        self.procrustes_align = args.procrustes_align
        self.world = args.world
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
        _, preds_2d, preds_3d, meta, _ = self.state
        reward, mpjpe = self._compute_reward(preds_2d, preds_3d, meta)
        done = self._check_done(self.state[3]['image'])
        
        self.index += 1
        self.episode_index += 1
        if self.index == len(self.dataset): 
            self.end_of_epoch = True
        
        return self.state, reward, done, mpjpe
    
    
    def get_state(self, index):
        """현재 state 반환"""
        image, meta = self.dataset[index]
        heatmap, preds_2d, preds_3d, maxvals = self.hpe_model(image.unsqueeze(0).to(self.device), no_grad=True)
        
        return heatmap, preds_2d, preds_3d, meta, maxvals
    
    
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

        joint_vis_penalty = - (num_outside / self.num_joints) * self.vis_theta
        mpjpe_penalty = (1 - (mpjpe / self.mpjpe_threshold)) * self.mpjpe_theta
        
        reward = joint_vis_penalty + mpjpe_penalty
        
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
    
    
    def select_action(self, state):
        if self.mode == 'random':
            action = np.random.uniform(-self.action_range, self.action_range, 2).reshape(1, -1)
        elif self.mode == 'rotation':
            horizontal_move = self.action_range if self.clockwise else -self.action_range
            vertical_move = np.random.uniform(-self.action_range/2, self.action_range/2)
            action = np.array([horizontal_move, vertical_move]).reshape(1, -1)
        elif self.mode == 'random_fixed':
            action = np.array([0, 0]).reshape(1, -1)
            
        return action
    