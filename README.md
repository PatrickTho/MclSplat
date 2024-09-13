# MclSplat

Monte Carlo Localization using Gaussian Splatting 


# 1. Installation

## A. Prerequisities

- Install ROS2 by following [ROS website]([http://wiki.ros.org/ROS/Installation](https://docs.ros.org/en/humble/Installation.html)).
- Set up a new Conda Environment to run Nerfstudio. Python >=3.8 is required, I recommend 3.10. The python version of Conda has to match the version of your system distribution to be compatible with ROS2. Running Conda with Ros in finicky, but should work if the versions are aligning
- install nerfstudio by following (https://docs.nerf.studio/quickstart/installation.html). Version 1.1.3 was tested


# 2. Clone and install the Repo
```bash
#don't forget to source your workspace
source /opt/ros/humble/setup.bash
mkdir -p ~/MclSplat/src
cd ~/ros2_ws/src
# Clone Repo
git clone [https://github.com/ros/ros_tutorials.git](https://github.com/PatrickTho/MclSplat.git)

# Build 
colcon build . 

# source workspace
source install/setup.bash

#install dependencies:
#pip install -r requirements.txt

```

## Starting MclSplat

  1. ros2 run mclsplat navigate_launch.py
  2. if you are running a real time experiment, launch the corresponding Rviz and Teleop scripts
  ```

## Provided config files. 

```turtle.yaml``` set up for real time demo with odometry
```nerfstudio.yaml``` runs Benchmarks on Nerfstudio Datasets and performs global localization 
```llff_global.yaml``` legacy

## Using Nerfstudio data

# Download data with the nerfstudio loader:
ns-download-data nerfstudio --capture-name=poster
# Train model
ns-train splatfacto --data data/nerfstudio/poster

The project is currently in an experimental version, several parameters are still hardcoded for debugging purposes. You can adjust the paths for the models in the full_filter.py class. Furthermore, the focal length and scaling ratio currently needs to be adjusted as well in the render_helpers.py 

### Plotting results

Logging is for the most the same as before for logging only Locnerf results, so the original guide can be followed:
If you log results from Loc-NeRF, we provide code to plot the position and rotation error inside  ```tools/eval_logs.py```. 

  # Third-party code:
 Major parts of this project were based on https://github.com/MIT-SPARK/Loc-NeRF from which parts were based on [this pytorch implementation of iNeRF](https://github.com/salykovaa/inerf) and [NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch).
 The Code that stems from these project is Licenced under the MIT License, which can be found at the "Licence" file.

Furthermore, the Robot Motion Model python class, as well as the Methods it references in the utils.py File are adapted from https://github.com/debbynirwan/mcl
The Code  taht stems from this Project is Licensed under the Apache-2 Lincence, which can be found in the "Licence-Apache" file.

Also, the Models, Render Methods and Data Parsers for training that were used stem from the Nerfstudio project: https://docs.nerf.studio/
```
LocNeRF
@misc{maggio2022locnerfmontecarlolocalization,
      title={Loc-NeRF: Monte Carlo Localization using Neural Radiance Fields}, 
      author={Dominic Maggio and Marcus Abate and Jingnan Shi and Courtney Mario and Luca Carlone},
      year={2022},
      eprint={2209.09050},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2209.09050}, 
}


 ```
 NeRF-Pytorch:
 
 @misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
 ```
 ```
iNeRF

@article{yen2020inerf,
  title={{iNeRF}: Inverting Neural Radiance Fields for Pose Estimation},
  author={Lin Yen-Chen and Pete Florence and Jonathan T. Barron and Alberto Rodriguez and Phillip Isola and Tsung-Yi Lin},
  year={2020},
  journal={arxiv arXiv:2012.05877},
}
```
