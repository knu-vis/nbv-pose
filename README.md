# nbv-pose
Implementation of "Real-Time Reinforcement Learning for Optimal Viewpoint Selection in Monocular 3D Human Pose Estimation"

This repository contains code for producing results of our NBV selection model following our experimental setup on the [CMU Panoptic dataset](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox).

### Overview of Our NBV Selection Model
![image](https://github.com/hyeon0819/nbv-pose/assets/153258272/89ddb67e-1d0e-4af9-b235-0b786811b2c1)

### Setup
1. Clone this repository `{ROOT}`.
2. Create and activate a `nbv` conda environment using the provided environment:
   ```
   conda env create -f nbv.yaml
   conda activate nbv
   ```   
3. To download the dataset, run the code:
     ```
     bash scripts/getData_single-person_hd.sh
     bash scripts/getData_single-person_vga.sh
     ```

### Train & Test the NBV selection model
- Train
   ```
   python run/ddpg_train.py --cfg experiments/panoptic_hd/hd.yaml
   ```

- Test
   ```
   python run/ddpg_test.py --cfg experiments/panoptic_hd/hd.yaml 
   ```

### Visual Results of Our Method
![image](https://github.com/hyeon0819/nbv-pose/assets/153258272/49b1c4ed-8939-48ee-a828-b9e98def2ea4)
