# NBV-PoseRL

This is the official implementation of IEEE Access paper ["Real-Time Reinforcement Learning for Optimal Viewpoint Selection in Monocular 3D Human Pose Estimation"](https://ieeexplore.ieee.org/abstract/document/10787005) by Sanghyeon Lee, Yoonho Hwang, and Jong Taek Lee.

This repository contains code for producing results of our NBV selection model following our experimental setup on the [CMU Panoptic dataset](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox).

### Overview of Our NBV Selection Model
![image](https://github.com/user-attachments/assets/61fdaac1-0cab-4de1-9c5d-31fc6098b2b3)



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
![image](https://github.com/user-attachments/assets/c600182f-5acd-4d46-85c3-5c36c7ed70b4)

![github_video](https://github.com/user-attachments/assets/da6f6971-46c1-4e29-8532-ea9eb7370cdf)


## Citation
If you use our code or models in your research, please cite with:
```
@article{lee2024real,
  title={Real-Time Reinforcement Learning for Optimal Viewpoint Selection in Monocular 3D Human Pose Estimation},
  author={Lee, Sanghyeon and Hwang, Yoonho and Lee, Jong Taek},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
