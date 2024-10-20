import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt
from .camera_utils_panoptic import transform_camera_to_world

body_edges = np.array([[0,1],[0,2],[0,3],[0,9],[3,4],[9,10],[4,5],
                       [10,11],[2,6],[2,12],[6,7],[12,13],[7,8],[13,14]])

def image2video(image_dir, video_path, fps=10):
    images = [img for img in sorted(os.listdir(image_dir)) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))
    cv2.destroyAllWindows()
    video.release()
    print('Video saved at', video_path)
    

def save_image(preds_3d, gt_3d, pelvis, camera, vis_image_output_dir, mpjpe, cam_idx, frame_index, episode_index, image_path, world=False):
    if world:
        world_pred = preds_3d - preds_3d[:, 2:3, :]
        world_gt = gt_3d - gt_3d[:, 2:3, :]
    else:
        world_pred = camera_to_world(preds_3d, camera['R'], camera['T'])
        world_gt = camera_to_world(gt_3d.reshape(-1, 15, 3), camera['R'], camera['T'])
        
        world_pred = world_pred - world_pred[:, 2:3, :]
        world_gt = world_gt - world_gt[:, 2:3, :]
    
    world_pred = world_pred + pelvis
    world_gt = world_gt + pelvis

    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
    labeling = True
    for edge in body_edges:
        ax.plot(world_pred[0][edge, 0], world_pred[0][edge, 1], world_pred[0][edge, 2], color='b', linewidth=2, 
                label='pred' if labeling else '')
        ax.plot(world_gt[0][edge, 0], world_gt[0][edge, 1], world_gt[0][edge, 2], color='r', linewidth=2, 
                label='GT' if labeling else '')
        labeling = False
    ax.scatter(*camera['position'], color='black', marker='^', s=100)
    ax.quiver(*camera['position'], *camera['R'][2], color='black', length=100, arrow_length_ratio=0.2)
        
    ax.set_xlim(-250, 250); ax.set_ylim(-250, 250); ax.set_zlim(-250, 250);ax.legend()
    ax.set_title(f'[{episode_index:04d}] MPJPE: {mpjpe:.2f},  cam index: {cam_idx}\n{image_path}')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_image_output_dir, f"pose_{frame_index:05d}.jpg"))
    plt.close()
    
    
def camera_to_world(pose_set, R, T):
    '''
    pose_set: (N, 15, 3)
    return: (N, 15, 3)
    '''
    t3d_world = []
    for i in range(len(pose_set)):
        t3d_camera = pose_set[i]
        t3d_camera = t3d_camera.reshape((-1, 3))
        
        world_coord = transform_camera_to_world(t3d_camera, R, T)
        world_coord = world_coord.reshape((15, 3))
        
        t3d_world.append(world_coord)
        
    return np.array(t3d_world)