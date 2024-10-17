#!/usr/bin/env python
import sys
sys.path.append('/home/student/anaconda3/envs/nerfstudio/lib/python3.10/site-packages')
from turtle import pos
#Diese Line muss im Terminal ausgeführt werden weil halt
#export PYTHONPATH=/home/student/anaconda3/envs/nerfstudio/lib/python3.10/site-packages:$PYTHONPATH


#export PYTHONPATH=$PYTHONPATH:/path/to/your/conda/env/lib/python3.10/site-packages
print(sys.version)
import rclpy
from rclpy.node import Node
import numpy as np
import gtsam
import cv2
import torch
import time
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose

from scipy.spatial.transform import Rotation as R
from copy import copy
#import sys

            
#from utils import 
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
from nerfstudio.cameras.cameras import Cameras, CameraType

from pathlib import Path


import locnerf
from locnerf.full_filter import NeRF
from locnerf.particle_filter import ParticleFilter
from locnerf.utils import get_pose, euler_from_quaternion, euler_to_quaternion, rot_psi, rot_phi, rot_theta
from locnerf.navigator_base import NavigatorBase

torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


# Global variables for publishers
pose_pub = None
particle_pub = None
gt_pub = None


qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
    history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
    depth=1
        )



class Navigator(NavigatorBase):
    def __init__(self, img_num=None, dataset_name=None):

        super().__init__( img_num=img_num, dataset_name=dataset_name)  # Call the base class initializer
        #initialize Node with correct name
        #Node.__init__(self,'nav_node',allow_undeclared_parameters= True, automatically_declare_parameters_from_overrides = True)
        # Base class handles loading most params.
       # NavigatorBase.__init__(self, img_num=img_num, dataset_name=dataset_name)
        print(self.run_inerf_compare)
        # Set initial distribution of particles.
       # self.get_logger().info(f"dataset name" + dataset_name)
        
        #TODO currently hardcoded for faster access, switch back to parameters
        self.full_particles_logged = False
        self.full_particles_count = 0
        self.min_world_bounds = {
            'px': -3.0,    # Minimum x position
            'py': 0.01,    # Minimum y position
            'pz': -2.0,    # Minimum z position
            'rz': -0.0000001,     # Fixed rotation around z (roll)
            'ry': -179.9,  # Minimum yaw
            'rx': -0.000001   # Fixed rotation around x (pitch)
        }

        self.max_world_bounds = {
            'px': -0.1,     # Maximum x position
            'py': 0.2,     # Maximum y position
            'pz': 1.0,     # Maximum z position
            'rz': 0.00001,     # Fixed rotation around z (roll)
            'ry': 179.9,   # Maximum yaw
            'rx': 0.0001    # Fixed rotation around x (pitch)
        }
        
        self.min_world_bounds_nerfstudio = {
            'px': -1.0,    # Minimum x position
            'py': -1.0,    # Minimum y position
            'pz': -1.0,    # Minimum z position
            'rz': 0.0001,     # Fixed rotation around z (roll)
            'ry': -180.0001, # Minimum yaw
            'rx':-20.0001   # Fixed rotation around x (pitch)
        }

        self.max_world_bounds_nerfstudio = {
            'px': 1.0,     # Maximum x position
            'py': 1.0,     # Maximum y position
            'pz': 1.0,     # Maximum z position
            'rz': 0.001,     # Fixed rotation around z (roll)
            'ry': 180.0,   # Maximum yaw
            'rx': 5.1  # Fixed rotation around x (pitch)
        }
        self.get_initial_distribution()
        self.odom_counts = 0 




        self.get_logger().info(f"PyTorch version: {torch.__version__}")

        # Check CUDA version
        self.get_logger().info(f"CUDA version: {torch.version.cuda}")

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        self.get_logger().info(f"CUDA available: {cuda_available}")

        if cuda_available:
            # Get current CUDA device index
            current_device = torch.cuda.current_device()
            self.get_logger().info(f"Current CUDA device index: {current_device}")

            # Get current CUDA device name
            device_name = torch.cuda.get_device_name(current_device)
            self.get_logger().info(f"Current CUDA device name: {device_name}")

            # Get current CUDA device properties
            device_properties = torch.cuda.get_device_properties(current_device)
            self.get_logger().info(f"Current CUDA device properties: {device_properties}")
        else:
            self.get_logger().warn("CUDA is not available.")



        self.br = CvBridge()
        
        #TODO check which formulation is correct
        self.pose_pub = self.create_publisher(Odometry, 'pose_est', 10)
        self.particle_pub = self.create_publisher(PoseArray, 'particle_poses', 10)
        self.gt_pub = self.create_publisher(PoseArray, 'ground_truth_pose', 10)


        # Set up publishers.
        #self.particle_pub = self.create_publisher(PoseArray, '/particles')
        #self.pose_pub = self.create_publisher(Odometry, '/estimated_pose')
        #self.gt_pub = self.create_publisher(PoseArray, '/gt_pose')
        

        # Set up subscribers.
        # We don't need callbacks to compare against inerf.
        if not self.run_inerf_compare:
        #check to subscribe the correct topics!

        #Important! This did have queue size 10 and buff size 2**24 in original code. It might cause problems on default settings
            self.image_sub = self.create_subscription(Image,self.rgb_topic,self.rgb_callback, 1)
            if self.run_predicts:
            # only needed when running live or i.e. otherwise important data. 
                self.vio_sub = self.create_subscription(Odometry, self.pose_topic, self.vio_callback,   qos_profile=qos_profile)

        # Show initial distribution of particles
        if self.plot_particles:
            self.visualize()

        if self.log_results:
            # If using a provided start we already have ground truth, so don't log redundant gt.
            if not self.use_logged_start:
                with open(self.log_directory + "/" + "gt_" + "fern" + "_" + str(self.obs_img_num) + "_" + "poses.npy", 'wb') as f:
                    np.save(f, self.gt_pose)

            # Add initial pose estimate before first update step is run.
            if self.use_weighted_avg:
                position_est = self.filter.compute_weighted_position_average()
            else:
                position_est = self.filter.compute_simple_position_average()
            rot_est = self.filter.compute_simple_rotation_average()
            pose_est = gtsam.Pose3(rot_est, position_est).matrix()
            self.all_pose_est.append(pose_est)

    def get_initial_distribution(self):
        # NOTE for now assuming everything stays in NeRF coordinates (x right, y up, z inward)
        if self.run_inerf_compare:
            # for non-global loc mode, get random pose based on iNeRF evaluation method from their paper
            # sample random axis from unit sphere and then rotate by a random amount between [-40, 40] degrees
            # translate along each axis by a random amount between [-10, 10] cm
            rot_rand = 40.0
            if self.global_loc_mode:
                trans_rand = 1.0
            else:
                trans_rand = 0.1
            
            # get random axis and angle for rotation
            x = np.random.rand()
            y = np.random.rand()
            z = np.random.rand()
            axis = np.array([x,y,z])
            axis = axis / np.linalg.norm(axis)
            angle = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
            euler = (gtsam.Rot3.AxisAngle(axis, angle)).ypr()

            # get random translation offset
            t_x = np.random.uniform(low=-trans_rand, high=trans_rand)
            t_y = np.random.uniform(low=-trans_rand, high=trans_rand)
            t_z = np.random.uniform(low=-trans_rand, high=trans_rand)


            # use initial random pose from previously saved log
            if self.use_logged_start:
                log_file = self.log_directory + "/" + "initial_pose_" + "_" + str(self.obs_img_num) + "_" + "poses.npy"
                start = np.load(log_file)
                print("this is important because it refers to our location")
                print(start)
                euler[0], euler[1], euler[2], t_x, t_y, t_z = start

            # log initial random pose TODO not really necessary
            elif self.log_results:
                with open(self.log_directory + "/" + "initial_pose_" + "fern" + "_" + str(self.obs_img_num) + "_" + "poses.npy", 'wb') as f:
                    np.save(f, np.array([euler[0], euler[1], euler[2], t_x, t_y, t_z]))

            if self.global_loc_mode:
                #print("IN GLOBAL LOC MODE!!", flush=True)
                # 360 degree rotation distribution about yaw
                self.initial_particles_noise = np.random.uniform(np.array(
                [self.min_world_bounds_nerfstudio['px'], self.min_world_bounds_nerfstudio['py'], self.min_world_bounds_nerfstudio['pz'], self.min_world_bounds_nerfstudio['rx'], self.min_world_bounds_nerfstudio['ry'], self.min_world_bounds_nerfstudio['rz']]),
                np.array([self.max_world_bounds_nerfstudio['px'], self.max_world_bounds_nerfstudio['py'], self.max_world_bounds_nerfstudio['pz'], self.max_world_bounds_nerfstudio['rx'], self.max_world_bounds_nerfstudio['ry'], self.max_world_bounds_nerfstudio['rz']]),
                size = (self.num_particles, 6))
                #self.initial_particles = self.set_initial_particles()
                #self.initial_particles_noise = np.random.uniform(np.array([-trans_rand, -trans_rand, -trans_rand, 0, -179, 0]), np.array([trans_rand, trans_rand, trans_rand, 0, 179, 0]), size = (self.num_particles, 6))
            else:
                self.initial_particles_noise = np.random.uniform(np.array([-trans_rand, -trans_rand, -trans_rand, 0, 0, 0]), np.array([trans_rand, trans_rand, trans_rand, 0, 0, 0]), size = (self.num_particles, 6))

            # center translation at randomly sampled position
            #self.initial_particles_noise[:, 0] += t_x
            #self.initial_particles_noise[:, 1] += t_y
            #self.initial_particles_noise[:, 2] += t_z
            
            if not self.global_loc_mode:
                """
                for i in range(self.initial_particles_noise.shape[0]):
                    # rotate random 3 DOF rotation about initial random rotation for each particle
                    n1 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                    n2 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                    n3 = np.pi * np.random.uniform(low=-rot_rand, high=rot_rand) / 180.0
                    euler_particle = gtsam.Rot3.AxisAngle(axis, angle).retract(np.array([n1, n2, n3])).ypr()

                    # add rotation noise for initial particle distribution
                    self.initial_particles_noise[i,3] = euler_particle[0] * 180.0 / np.pi
                    self.initial_particles_noise[i,4] = euler_particle[1] * 180.0 / np.pi 
                    self.initial_particles_noise[i,5] = euler_particle[2] * 180.0 / np.pi 
                 
                 
                """ 
                print("This code is not used anymore in the current Version ", flush = True)
         # get distribution of particles from user
        else: 
           
            self.initial_particles_noise = np.random.uniform(np.array(
                [self.min_world_bounds['px'], self.min_world_bounds['py'], self.min_world_bounds['pz'], self.min_world_bounds['rx'], self.min_world_bounds['ry'], self.min_world_bounds['rz']]),
                np.array([self.max_world_bounds['px'], self.max_world_bounds['py'], self.max_world_bounds['pz'], self.max_world_bounds['rx'], self.max_world_bounds['ry'], self.max_world_bounds['rz']]),
                size = (self.num_particles, 6))
        self.initial_particles = self.set_initial_particles()
        #here the particles first get initialized


        self.filter = ParticleFilter(self.initial_particles)


    #needs to be adjusted for the respective Robot
    def vio_callback(self, msg):
        if self.odom_counts ==0:
            print("VIO CALLBACK!!!", flush=True)
            # extract rotation and position from msg
            quat = msg.pose.pose.orientation
            position = msg.pose.pose.position
            #most likely not needed
            # rotate vins to be in nerf frame
            rx = self.R_bodyVins_camNerf['rx']
            ry = self.R_bodyVins_camNerf['ry']
            rz = self.R_bodyVins_camNerf['rz']
            #BIG PROBLEM WITH THIS
            #T_bodyVins_camNerf = gtsam.Pose3(gtsam.Rot3.Ypr(rz, ry, rx), gtsam.Point3(0,0,0))
            bodyvins_rot = rot_psi(rz)  @ rot_theta(ry)@ rot_phi(rx) 
            vins = np.eye(4)
            vins[:3, :3] = bodyvins_rot
            vins[:3, 3] = [0, 0, 0]
            T_bodyVins_camNerf = gtsam.Pose3(vins)
            #This is probably causing trouble. It seems like the roation and quats are in a different coordinate system. ?
            T_wVins_camVins = gtsam.Pose3(gtsam.Rot3(quat.w, quat.x, quat.y, quat.z), gtsam.Point3(position.x, position.y, position.z))
            #T_wVins_camNeRF = gtsam.Pose3(T_wVins_camVins.matrix() @ T_bodyVins_camNerf.matrix())


            #TODO new attempt at odometry

            if self.previous_vio_pose is not None:
                self.run_predict(self.previous_vio_pose, msg.pose.pose)

            #if self.previous_vio_pose is not None:
            #    T_camNerft_camNerftp1 = gtsam.Pose3(self.previous_vio_pose.inverse().matrix() @ T_wVins_camNeRF.matrix())
            #    self.run_predict(T_camNerft_camNerftp1)

            # log pose for next transform computation
            self.previous_vio_pose = T_wVins_camVins

            # publish particles for rviz
            if self.plot_particles:
                self.visualize()
            self.odom_counts+=1
    
    def rgb_callback(self, msg):
        self.img_msg = msg
        
    def rgb_run(self, msg, get_rays_fn=None, render_full_image=False):
        self.odom_counts = 0
        #print("processing image", flush=True)
        self.full_particles_count += 1
        self.total_start_time = time.time() if not hasattr(self, 'total_start_time') else self.total_start_time
        if self.full_particles_count >= 100:
            print("CONVERGENCE DONE!",flush=True)
            print("Runs until Convergence:" ,flush= True)
            print(self.full_particles_count, flush=True)
            average_hertz = self.full_particles_count / (time.time()-  self.total_start_time )
            print("AVERAGE HERTZ UNTIL CONVERSION" + str(average_hertz), flush=True)

            self.full_particles_logged = True    
            print()
        start_time = time.time()
        self.rgb_input_count += 1
        print(self.filter.particles['rotation'][0], flush= True)
        print(len(self.filter.particles['rotation']))
        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_rotation_before_update = [gtsam.Rot3(i.matrix()) for i in self.filter.particles['rotation']]

        if self.use_convergence_protection:
            for i in range(self.number_convergence_particles):
                t_x = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                t_y = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                t_z = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                # TODO this is not thread safe. have two lines because we need to both update
                # particles to check the loss and the actual locations of the particles
                self.filter.particles["position"][i] = self.filter.particles["position"][i] + np.array([t_x, 0, t_z])
                particles_position_before_update[i] = particles_position_before_update[i] + np.array([t_x, 0, t_z])
                #
        if self.use_received_image:
            #print("USING RECEIVED IMAGE, this is ther shape before and after cv2")
            #print(msg.shape, flush=True)
            #TODO check if necessary in my setup??
            img = self.br.imgmsg_to_cv2(msg)
            #print(img.shape, flush = True)
            # resize input image so it matches the scale that NeRF expects - downscale to low res.


            #img = cv2.resize(self.br.imgmsg_to_cv2(msg), (int(self.nerf.W), int(self.nerf.H)))
            self.nerf.obs_img = img

            self.nerf.obs_img_noised = img
            show_true = self.view_debug_image_iteration != 0 and self.num_updates == self.view_debug_image_iteration-1


        total_nerf_time = 0

        loss_poses = []
        for index, particle in enumerate(particles_position_before_update):
            loss_pose = np.zeros((4,4), dtype=np.float64)
            #print(particles_rotation_before_update[index])
            rot = particles_rotation_before_update[index]
            #quats = rot.toQuaternion()
            loss_pose[0:3, 0:3] = rot.matrix()
            loss_pose[0:3,3] = particle[0:3]
            loss_pose[3,3] = 1.0
            loss_poses.append(loss_pose)
            loss_poses_np = np.array(loss_poses)
            loss_poses_np = loss_poses_np.astype(np.float32)
            loss_poses_tensor = torch.from_numpy(loss_poses_np)
            loss_poses_tensor = loss_poses_tensor.to(device)
        # todo pass in pixel_amount 
        losses, nerf_time = self.nerf.get_loss(loss_poses_tensor, self.photometric_loss)
        

        for index, particle in enumerate(particles_position_before_update):
            self.filter.weights[index] = 1/losses[index]
        total_nerf_time += nerf_time

        self.filter.update()
        self.num_updates += 1
        print("UPDATE STEP NUMBER", self.num_updates, "RAN", flush=True)
        print("number particles:", self.num_particles, flush=True)

        if self.use_refining: # TODO make it where you can reduce number of particles without using refining
            self.check_refine_gate()

        if self.use_weighted_avg:
            avg_pose = self.filter.compute_weighted_position_average()
        else:
            avg_pose = self.filter.compute_simple_position_average()
            print("avg pose:")
            print(avg_pose)
        avg_rot = self.filter.compute_simple_rotation_average()
        self.nerf_pose = gtsam.Pose3(avg_rot, gtsam.Point3(avg_pose[0], avg_pose[1], avg_pose[2])).matrix()

        if self.plot_particles:
            self.visualize()
            
        # TODO add ability to render several frames
        if self.view_debug_image_iteration != 0 and (self.num_updates == self.view_debug_image_iteration):
            self.nerf.visualize_nerf_image(self.nerf_pose)

        if not self.use_received_image:
            if self.use_weighted_avg:
                print("average position of all particles: ", self.filter.compute_weighted_position_average())
                print("position error: ", np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average()))
            else:
                print("average position of all particles: ", self.filter.compute_simple_position_average())
                print("position error: ", np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average()))

        if self.use_weighted_avg:
            position_est = self.filter.compute_weighted_position_average()
        else:
            position_est = self.filter.compute_simple_position_average()
        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = gtsam.Pose3(rot_est, position_est).matrix()
        if self.num_updates % 5 == 0 and self.num_updates<50:
            print("saving image", flush=True)

            self.nerf.save_nerf_image(pose_est, self.num_updates)
        if self.log_results:
            self.all_pose_est.append(pose_est)
        
      
        if not self.run_inerf_compare:
            img_timestamp = msg.header.stamp
            self.publish_pose_est(pose_est, img_timestamp)
            
        else:
            self.publish_pose_est(pose_est)
       
        update_time = time.time() - start_time
        print("time it took:", total_nerf_time, "out of total", update_time, "for update step", flush=True)

        if not self.run_predicts:
            self.filter.predict_no_motion(self.px_noise, self.py_noise, self.pz_noise, self.rot_x_noise, self.rot_y_noise, self.rot_z_noise) #  used if you want to localize a static image
        
        # return is just for logging
        return pose_est
    
    def check_if_position_error_good(self, return_error = False):
        """
        check if position error is less than 5cm, or return the error if return_error is True
        """
        acceptable_error = 0.1
        if self.use_weighted_avg:
            error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average())
            if return_error:
                return error
            return error < acceptable_error
        else:
            error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average())
            if return_error:
                return error
            return error < acceptable_error

    def check_if_rotation_error_good(self, return_error = False):
        """
        check if rotation error is less than 5 degrees, or return the error if return_error is True
        """
        acceptable_error = 10.0
        average_rot_t = (self.filter.compute_simple_rotation_average()).transpose()
        # check rot in bounds by getting angle using https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices

        r_ab = average_rot_t @ (self.gt_pose[0:3,0:3])
        rot_error = np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2))
        print("rotation error: ", rot_error)
        if return_error:
            return rot_error
        return abs(rot_error) < acceptable_error
    
    def run_predict(self, previous_vio_pose, current_pose):
        self.filter.predict_particles(previous_vio_pose, current_pose, self.px_noise, self.py_noise, self.pz_noise, self.rot_x_noise, self.rot_y_noise, self.rot_z_noise)
    #def run_predict(self, delta_pose):particle
        #this needs to be severly changed
    #    self.filter.predict_with_delta_pose(delta_pose, self.px_noise, self.py_noise, self.pz_noise, self.rot_x_noise, self.rot_y_noise, self.rot_z_noise)

        if self.plot_particles:
            self.visualize()
    
    def set_initial_particles(self):
        initial_positions = np.zeros((self.num_particles, 3))
        rots = []
        print("intial Particle Rot", flush = True)
        print(self.initial_particles_noise[0])
        #correct at that piiunb
        for index, particle in enumerate(self.initial_particles_noise):
            x = particle[0]
            y = particle[1]
            z = particle[2]
            phi = particle[3]
            theta = particle[4]
            psi = particle[5]
            
            particle_pose = get_pose(phi, theta, psi, x, y, z, self.nerf.obs_img_pose, self.center_about_true_pose, self.use_nerfstudio_convention)
            #print(particle_pose)
            # set positions
            initial_positions[index,:] = [particle_pose[0,3], particle_pose[1,3], particle_pose[2,3]]
            # set orientations maybe i need to cahge how rot is handled idk
            rots.append(gtsam.Rot3(particle_pose[0:3,0:3]))
            #print(rots[index])
        print("initial Position for checkup", flush=True)
        print(rots[0], flush=True)
            # print(initial_particles)
        return {'position':initial_positions, 'rotation':np.array(rots)}

    def set_noise(self, scale):

        self.px_noise = self.get_parameter('px_noise').get_parameter_value().double_value / scale
        self.py_noise = self.get_parameter('py_noise').get_parameter_value().double_value / scale
        self.pz_noise = self.get_parameter('pz_noise').get_parameter_value().double_value / scale
        self.rot_x_noise = self.get_parameter('rot_x_noise').get_parameter_value().double_value / scale
        self.rot_y_noise = self.get_parameter('rot_y_noise').get_parameter_value().double_value / scale
        self.rot_z_noise = self.get_parameter('rot_z_noise').get_parameter_value().double_value / scale


    def check_refine_gate(self):
    
        # get standard deviation of particle position
        sd_xyz = np.std(self.filter.particles['position'], axis=0)
        norm_std = np.linalg.norm(sd_xyz)
        refining_used = False
        print("sd_xyz:", sd_xyz)
        print("norm sd_xyz:", np.linalg.norm(sd_xyz))

        if norm_std < self.alpha_super_refine:
            print("SUPER REFINE MODE ON")
            # reduce original noise by a factor of 4
            self.set_noise(scale = 4.0)
            refining_used = True
        elif norm_std < self.alpha_refine:
            print("REFINE MODE ON")
            # reduce original noise by a factor of 2
            self.set_noise(scale = 2.0)
            refining_used = True
        else:
            # reset noise to original value
            self.set_noise(scale = 1.0)
        
        if refining_used and self.use_particle_reduction:
            self.filter.reduce_num_particles(self.min_number_particles)
            self.num_particles = self.min_number_particles

            #TODO CHECKER
            
    def publish_pose_est(self, pose_est_gtsam, img_timestamp = None):
        pose_est = Odometry()
        pose_est.header.frame_id = "world"

        # if we don't run on rosbag data then we don't have timestamps
        if img_timestamp is not None:
            pose_est.header.stamp = img_timestamp

        pose_est_gtsam = gtsam.Pose3(pose_est_gtsam)
        position_est = pose_est_gtsam.translation()
        rot_est = pose_est_gtsam.rotation().toQuaternion()

        # populate msg with pose information
        pose_est.pose.pose.position.x = position_est[0]
        pose_est.pose.pose.position.z = position_est[1]
        pose_est.pose.pose.position.y = position_est[2]
        pose_est.pose.pose.orientation.w = rot_est.w()
        """
        pose_est.pose.pose.orientation.x = rot_est.x()
        pose_est.pose.pose.orientation.y = rot_est.z()
        pose_est.pose.pose.orientation.z = rot_est.y()
        
        """
        pose_est.pose.pose.orientation.x = rot_est.x()
        pose_est.pose.pose.orientation.y = rot_est.z()
        pose_est.pose.pose.orientation.z = rot_est.y()
        # print(pose_est_gtsam.rotation().ypr())

        # publish pose
        self.pose_pub.publish(pose_est)
            
    def visualize(self):
        # publish pose array of particles' poses

        poses = []
        R_nerf_body = gtsam.Rot3.Rx(-np.pi/2)
        for index, particle in enumerate(self.filter.particles['position']): 
            p = Pose()
            p.position.x = particle[0]
            p.position.y = -particle[2]
            p.position.z = particle[1]
            # print(particle[3],particle[4],particle[5])
            rot = self.filter.particles['rotation'][index]
            #gtsam.Quaternion.Exp(gtsam.Vector4(np.cos(-yaw_angle/2), 0, 0, np.sin(-yaw_angle/2)))
            orient = rot.toQuaternion()
            #print(orient.x())
            #print(orient.y())
            #print(orient.z())
            #p.orientation.w = orient.w()
            #p.orientation.x = orient.x()
            #p.orientation.y = orient.z()
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            #p.orientation.z = orient.y()
            
            
            _,_, yaw = euler_from_quaternion(orient.x(), orient.z(), orient.y(), orient.w())
            #print(yaw, flush=True)
            quat = euler_to_quaternion(yaw+np.pi/2, 0.0, 0.0)
            #print(quat.z)
            p.orientation.z = quat.z
            p.orientation.w = quat.w

            poses.append(p)
            
        pa = PoseArray()
        pa.poses = poses
        pa.header.frame_id = "map"
        #pa.header.stamp = self.get_clock().now().to_msg()
        self.particle_pub.publish(pa)

        # if we have a ground truth pose then publish it
        if not self.use_received_image or self.gt_pose is not None:
            gt_array = PoseArray()
            gt = Pose()
            gt_rot = gtsam.Rot3(self.gt_pose[0:3,0:3]).toQuaternion()
            gt.orientation.w = gt_rot.w()
            gt.orientation.x = gt_rot.x()
            gt.orientation.y = gt_rot.y()
            gt.orientation.z = gt_rot.z()
            gt.position.x = self.gt_pose[0,3]
            gt.position.y = self.gt_pose[2,3]
            gt.position.z = self.gt_pose[3,3]
            gt_array.poses = [gt]
            gt_array.header.frame_id = "world"
            #self.gt_pub.publish(gt_array)
 
def average_arrays(axis_list):
    """
    average arrays of different size
    adapted from https://stackoverflow.com/questions/49037902/how-to-interpolate-a-line-between-two-other-lines-in-python/49041142#49041142

    axis_list = [forward_passes_all, accuracy]
    """
    min_max_xs = [(min(axis), max(axis)) for axis in axis_list[0]]

    new_axis_xs = [np.linspace(min_x, max_x, 100) for min_x, max_x in min_max_xs]
    new_axis_ys = []
    for i in range(len(axis_list[0])):
        new_axis_ys.append(np.interp(new_axis_xs[i], axis_list[0][i], axis_list[1][i]))

    midx = [np.mean([new_axis_xs[axis_idx][i] for axis_idx in range(len(axis_list[0]))])for i in range(100)]
    midy = [np.mean([new_axis_ys[axis_idx][i] for axis_idx in range(len(axis_list[0]))]) for i in range(100)]

    plt.plot(midx, midy)
    plt.xlabel("number of forward passes (in thousands)")
    plt.grid()
    plt.savefig('average_plot.png')  # Save the plot as an image
    plt.show()


def main(args=None):
    rclpy.init(args=args)
    print(sys.executable)
    nav_node = rclpy.create_node('nav_node')
    nav_node.declare_parameter('run_inerf_compare', False)
    nav_node.declare_parameter('use_logged_start', False)
    nav_node.declare_parameter('log_directory', "default")

    print("testestest")

    run_inerf_compare = nav_node.get_parameter("run_inerf_compare").get_parameter_value().bool_value 
    # TODO replace!
    #run_inerf_compare = True
    use_logged_start = nav_node.get_parameter('use_logged_start').get_parameter_value().bool_value  # For a string parameter, maybe switch to BOOL
    log_directory = nav_node.get_parameter('log_directory').get_parameter_value().string_value  # For a string parameter
    nav_node.get_logger().info(f"value of inerf_compar: {run_inerf_compare}")
    nav_node.destroy_node()
    
    if run_inerf_compare:
        num_starts_per_dataset = 20 # 5 was the default here, needs to be changed back!!!# TODO make this a param
        #datasets = ['fern', 'horns', 'fortress', 'room'] # TODO make this a param
        datasets = ['no']

        total_position_error_good = []
        total_rotation_error_good = []
        total_num_forward_passes = []
        for dataset_index, dataset_name in enumerate(datasets):
            print("Starting iNeRF Style Test on Dataset: ", dataset_name)
            #node.get_logger(dataset_name)
            
            #Get Param since this be causing errors!!
            use_logged_start = False
            if use_logged_start:
                #what does this do????
                start_pose_files = glob.glob(log_directory + "/initial_pose_" + dataset_name + '_' +'*')

            # only use an image number once per dataset
            used_img_nums = set()
            for i in range(num_starts_per_dataset):
                if not use_logged_start:

                    

                    img_num = np.random.randint(low=0, high=225) # TODO can increase the range of images
                    while img_num in used_img_nums:
                        img_num = np.random.randint(low=0, high=225)
                    used_img_nums.add(img_num)
                
                else:
                    start_file = start_pose_files[i]
                    img_num = int(start_file.split('_')[5])
                # maybe we made a mistake here but it seems to only affect the image number and not at all the pose.. so what??
                mcl_local = Navigator(img_num, dataset_name)
                print()
                #print("Using Image Number:", mcl_local.obs_img_num)
                print("Test", i+1, "out of", num_starts_per_dataset,flush=True)

                num_forward_passes_per_iteration = [0]
                position_error_good = []
                rotation_error_good = []
                ii = 0
                while num_forward_passes_per_iteration[-1] < mcl_local.forward_passes_limit:
                    print()
                    #somewhere here something goes totally wrong.
                    print("total rendered images limit, current images:", mcl_local.forward_passes_limit, num_forward_passes_per_iteration[-1], flush=True)
                    print(mcl_local.gt_pose)
                    position_error_good.append(int(mcl_local.check_if_position_error_good()))
                    rotation_error_good.append(int(mcl_local.check_if_rotation_error_good()))
                    if ii != 0:
                        #thread crashes here
                        mcl_local.rgb_run('temp')
                        num_forward_passes_per_iteration.append(num_forward_passes_per_iteration[ii-1] + mcl_local.num_particles)
                    ii += 1
                print("still alive at 608 nav node", flush=True)
                if mcl_local.log_results:
                    with open(mcl_local.log_directory + "/" + "mocnerf_" + mcl_local.log_prefix + "_" + "fern" + "_" + str(mcl_local.obs_img_num) + "_" + "poses.npy", 'wb') as f:
                        np.save(f, np.array(mcl_local.all_pose_est))
                    with open(mcl_local.log_directory + "/" + "mocnerf_" + mcl_local.log_prefix + "_" + "fern" + "_" + str(mcl_local.obs_img_num) + "_" + "forward_passes.npy", 'wb') as f:
                        np.save(f, np.array(num_forward_passes_per_iteration))
                print("still alive at 614 nav node", flush=True)
                total_num_forward_passes.append(num_forward_passes_per_iteration)
                total_position_error_good.append(position_error_good)
                total_rotation_error_good.append(rotation_error_good)
                print("still alive at 622 nav node", flush=True)
        print("still alive at 623 outside for loop", flush=True)
        #TODO Isolated Bug!!
        average_arrays([total_num_forward_passes, total_position_error_good])
        average_arrays([total_num_forward_passes, total_rotation_error_good])
        print("still alive at 621 nav node", flush=True)
    #rclpy.spin(node)
    #rclpy.shutdown()
    else:
        node = Navigator()  # Initialisiere den Navigator-Node
        try:
            while rclpy.ok():
                rclpy.spin_once(node)
                if node.img_msg is not None:
                        node.rgb_run(node.img_msg)
                        node.img_msg = None # TODO not thread safe   
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()





if __name__ == "__main__":
    main()
    

    # run normal live ROS mode
    #why is t
   # else:
       # mcl = Navigator()      
       # while not rospy.is_shutdown():
          #  if mcl.img_msg is not None:
            #    mcl.rgb_run(mcl.img_msg)
            #    mcl.img_msg = None # TODO not thread safe
#
    def publish_pose_est(self, pose_est_gtsam, img_timestamp=None):
        pose_est = Odometry()
        pose_est.header.frame_id = "world"

        # if we don't run on rosbag data then we don't have timestamps
        if img_timestamp is not None:
            pose_est.header.stamp = img_timestamp

        pose_est_gtsam = gtsam.Pose3(pose_est_gtsam)
        position_est = pose_est_gtsam.translation()
        rot_est = pose_est_gtsam.rotation().quaternion()

        # populate msg with pose information
        pose_est.pose.pose.position.x = position_est[0]
        pose_est.pose.pose.position.y = position_est[1]
        pose_est.pose.pose.position.z = position_est[2]
        pose_est.pose.pose.orientation.w = rot_est[0]
        pose_est.pose.pose.orientation.x = rot_est[1]
        pose_est.pose.pose.orientation.y = rot_est[2]
        pose_est.pose.pose.orientation.z = rot_est[3]
