import rclpy
from rclpy.node import Node
import numpy as np
import warnings

#testing workaround
import sys
sys.path.append('/home/student/anaconda3/envs/locnerf_new/lib/python3.10/site-packages')

import torch

from locnerf.full_filter import NeRF


 # Base class to handle loading params from yaml.

class NavigatorBase(Node):
    def __init__(self, img_num=0, dataset_name=None, node=None):
        super().__init__('nav_node', allow_undeclared_parameters= True) 
         # Initialize the node here
        #self.get_logger().info(f"CUDA available:" + dataset_name)

        #try to log cuda and torch info
        # Check CUDA version

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


        
        
        
        
        # extract params


        #declare params
        self.declare_parameter('data_dir', '/default/path/to/data')
        self.declare_parameter('ckpt_dir', '/default/path/to/data')
        


        self.declare_parameter('factor', 0)
        self.declare_parameter('focal', 0)
        self.declare_parameter('H', 0)
        self.declare_parameter('W', 0)
        self.declare_parameter('dataset_type', 'default')
        self.declare_parameter('num_particles', 0)
        self.declare_parameter('visualize_particles', False)
        self.declare_parameter('rgb_topic', 'default')
        self.declare_parameter('vio_topic', 'default')
        self.declare_parameter('near', 0)
        self.declare_parameter('far', 0)
        self.declare_parameter('course_samples', 0)
        self.declare_parameter('fine_samples', 0)
        self.declare_parameter('batch_size', 0)
        self.declare_parameter('kernel_size', 0)
        self.declare_parameter('lrate', 0.0)
        self.declare_parameter('sampling_strategy', 'default')
        self.declare_parameter('no_ndc', False)
        self.declare_parameter('dil_iter', 0)
        self.declare_parameter('multires', 0)
        self.declare_parameter('multires_views', 0)
        self.declare_parameter('i_embed', 0)
        self.declare_parameter('netwidth', 0)
        self.declare_parameter('netdepth', 0)
        self.declare_parameter('netdepth_fine', 0)
        self.declare_parameter('netwidth_fine', 0)
        self.declare_parameter('use_viewdirs', False)
        self.declare_parameter('perturb', 0)
        self.declare_parameter('white_bkgd', False)
        self.declare_parameter('raw_noise_std', 0.00)
        self.declare_parameter('lindisp', False)
        self.declare_parameter('netchunk', 0)
        self.declare_parameter('chunk', 0)
        self.declare_parameter('bd_factor', 0.00)
        self.declare_parameter('use_nerfstudio_convention', False)
        self.declare_parameter('log_prefix', 'default')



        self.declare_parameter('photometric_loss', 'default')

        self.declare_parameter('view_debug_image_iteration', 0)

        self.declare_parameter('px_noise', 0.00)
        self.declare_parameter('py_noise', 0.00)
        self.declare_parameter('pz_noise', 0.00)
        self.declare_parameter('rot_x_noise', 0.00)
        self.declare_parameter('rot_y_noise', 0.00)
        self.declare_parameter('rot_z_noise', 0.00)

        self.declare_parameter('use_convergence_protection', False)
        self.declare_parameter('number_convergence_particles', 0)
        self.declare_parameter('convergence_noise', 0.00)

        self.declare_parameter('use_weighted_avg', False)

        self.declare_parameter('min_number_particles', 0)
        self.declare_parameter('use_particle_reduction', True)

        self.declare_parameter('alpha_refine', 0.00)
        self.declare_parameter('alpha_super_refine', 0.00)

        self.declare_parameter('run_predicts', False)
        self.declare_parameter('use_received_image', False)
        self.declare_parameter('run_inerf_compare')
        self.declare_parameter('global_loc_mode', False)
        self.declare_parameter('run_nerfnav_compare').get_parameter_value().bool_value
        self.declare_parameter('nerf_nav_directory').get_parameter_value().string_value
        self.declare_parameter('center_about_true_pose', False)
        self.declare_parameter('use_refining', False)
        self.declare_parameter('log_results', False)
        self.declare_parameter('log_directory', 'default')
        self.declare_parameter('use_logged_start').get_parameter_value().bool_value
        self.declare_parameter('forward_passes_limit', 0)

        self.factor = self.get_parameter('factor').get_parameter_value().integer_value
        self.focal = self.get_parameter('focal').get_parameter_value().integer_value
        self.H = self.get_parameter('H').get_parameter_value().integer_value
        self.W = self.get_parameter('W').get_parameter_value().integer_value
        self.dataset_type = self.get_parameter('dataset_type').get_parameter_value().string_value
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.plot_particles  = self.get_parameter('visualize_particles').get_parameter_value().bool_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.pose_topic = self.get_parameter('vio_topic').get_parameter_value().string_value
        self.near = self.get_parameter('near').get_parameter_value().integer_value
        self.far = self.get_parameter('far').get_parameter_value().integer_value
        self.course_samples = self.get_parameter('course_samples').get_parameter_value().integer_value
        self.fine_samples = self.get_parameter('fine_samples').get_parameter_value().integer_value
        self.batch_size = self.get_parameter('batch_size').get_parameter_value().integer_value
        self.kernel_size = self.get_parameter('kernel_size').get_parameter_value().integer_value
        self.lrate = self.get_parameter('lrate').get_parameter_value().double_value
        self.sampling_strategy = self.get_parameter('sampling_strategy').get_parameter_value().string_value
        self.no_ndc = self.get_parameter('no_ndc').get_parameter_value().bool_value
        self.dil_iter = self.get_parameter('dil_iter').get_parameter_value().integer_value
        self.multires = self.get_parameter('multires').get_parameter_value().integer_value
        self.multires_views = self.get_parameter('multires_views').get_parameter_value().integer_value
        self.i_embed = self.get_parameter('i_embed').get_parameter_value().integer_value
        self.netwidth = self.get_parameter('netwidth').get_parameter_value().integer_value
        self.netdepth = self.get_parameter('netdepth').get_parameter_value().integer_value
        self.netdepth_fine = self.get_parameter('netdepth_fine').get_parameter_value().integer_value
        self.netwidth_fine = self.get_parameter('netwidth_fine').get_parameter_value().integer_value
        self.use_viewdirs = self.get_parameter('use_viewdirs').get_parameter_value().bool_value
        self.perturb = self.get_parameter('perturb').get_parameter_value().integer_value
        self.white_bkgd = self.get_parameter('white_bkgd').get_parameter_value().bool_value
        self.raw_noise_std = self.get_parameter('raw_noise_std').get_parameter_value().double_value
        self.lindisp = self.get_parameter('lindisp').get_parameter_value().bool_value
        self.netchunk = self.get_parameter('netchunk').get_parameter_value().integer_value
        self.chunk = self.get_parameter('chunk').get_parameter_value().integer_value
        self.bd_factor = self.get_parameter('bd_factor').get_parameter_value().double_value
        self.use_nerfstudio_convention = self.get_parameter('use_nerfstudio_convention').get_parameter_value().bool_value
        self.log_prefix = self.get_parameter('log_prefix').get_parameter_value().string_value

        # just used for Nerf-Navigation comparison
        self.model_ngp = None
        self.ngp_opt = None
        #Dataset_name is None unfortunately
        #self.get_logger().info(f'this is the current dataset_name' + dataset_name)
        #if dataset_name is not None:
        #    self.model_name = dataset_name
        #else:
        self.declare_parameter('model_name', 'default')
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        
        self.data_dir = self.get_parameter('data_dir').get_parameter_value().string_value+ "/" + self.model_name

        self.ckpt_dir = self.get_parameter('ckpt_dir').get_parameter_value().string_value + "/" + self.model_name     

        self.obs_img_num = img_num

        # TODO these don't individually need to be part of the navigator class
        nerf_params = {'near':self.near, 'far':self.far, 'course_samples':self.course_samples, 'fine_samples':self.fine_samples,
                       'batch_size':self.batch_size, 'factor':self.factor, 'focal':self.focal, 'H':self.H, 'W':self.W, 'dataset_type':self.dataset_type,
                       'obs_img_num':self.obs_img_num, 'kernel_size':self.kernel_size, 'lrate':self.lrate, 'sampling_strategy':self.sampling_strategy,
                       'model_name':self.model_name, 'data_dir':self.data_dir, 'no_ndc':self.no_ndc, 'dil_iter':self.dil_iter,
                       'multires':self.multires, 'multires_views':self.multires_views, 'i_embed':self.i_embed, 'netwidth':self.netwidth, 'netdepth':self.netdepth,
                       'netdepth_fine':self.netdepth_fine, 'netwidth_fine':self.netwidth_fine, 'use_viewdirs':self.use_viewdirs, 'ckpt_dir':self.ckpt_dir,
                       'perturb':self.perturb, 'white_bkgd':self.white_bkgd, 'raw_noise_std':self.raw_noise_std, 'lindisp':self.lindisp,
                       'netchunk':self.netchunk, 'chunk':self.chunk, 'bd_factor':self.bd_factor, 'use_nerfstudio_convention': self.use_nerfstudio_convention}
        self.nerf = NeRF(nerf_params)
        
        self.image = None
        self.rgb_input_count = 0
        self.num_updates = 0
        self.photometric_loss = self.get_parameter('photometric_loss').get_parameter_value().string_value

        self.view_debug_image_iteration = self.get_parameter('view_debug_image_iteration').get_parameter_value().bool_value

        self.px_noise = self.get_parameter('px_noise').get_parameter_value().double_value
        self.py_noise = self.get_parameter('py_noise').get_parameter_value().double_value
        self.pz_noise = self.get_parameter('pz_noise').get_parameter_value().double_value
        self.rot_x_noise = self.get_parameter('rot_x_noise').get_parameter_value().double_value
        self.rot_y_noise = self.get_parameter('rot_y_noise').get_parameter_value().double_value
        self.rot_z_noise = self.get_parameter('rot_z_noise').get_parameter_value().double_value

        self.use_convergence_protection = self.get_parameter('use_convergence_protection').get_parameter_value().bool_value
        self.number_convergence_particles = self.get_parameter('number_convergence_particles').get_parameter_value().integer_value
        self.convergence_noise = self.get_parameter('convergence_noise').get_parameter_value().double_value

        self.use_weighted_avg = self.get_parameter('use_weighted_avg').get_parameter_value().bool_value

        self.min_number_particles = self.get_parameter('min_number_particles').get_parameter_value().integer_value
        self.use_particle_reduction = self.get_parameter('use_particle_reduction').get_parameter_value().bool_value

        self.alpha_refine = self.get_parameter('alpha_refine').get_parameter_value().double_value
        self.alpha_super_refine = self.get_parameter('alpha_super_refine').get_parameter_value().double_value

        self.run_predicts = self.get_parameter('run_predicts').get_parameter_value().bool_value
        self.use_received_image = self.get_parameter('use_received_image').get_parameter_value().bool_value
        self.run_inerf_compare = self.get_parameter('run_inerf_compare').get_parameter_value().bool_value
        self.global_loc_mode = self.get_parameter('global_loc_mode').get_parameter_value().bool_value
        self.run_nerfnav_compare = self.get_parameter('run_nerfnav_compare').get_parameter_value().bool_value
        self.nerf_nav_directory = self.get_parameter('nerf_nav_directory').get_parameter_value().string_value
        self.center_about_true_pose = self.get_parameter('center_about_true_pose').get_parameter_value().bool_value
        self.use_refining = self.get_parameter('use_refining').get_parameter_value().bool_value
        self.log_results = self.get_parameter('log_results').get_parameter_value().bool_value
        self.log_directory = self.get_parameter('log_directory').get_parameter_value().string_value
        self.use_logged_start = self.get_parameter('use_logged_start').get_parameter_value().bool_value
        self.forward_passes_limit = self.get_parameter('forward_passes_limit').get_parameter_value().integer_value

        self.declare_parameters(
            namespace='',
            parameters=[
                # min_bounds
                ('min_bounds.px', -0.5 ),
                ('min_bounds.py', -0.3 ),
                ('min_bounds.pz', 0.0 ),
                ('min_bounds.rz', -2.5 ),
                ('min_bounds.ry', -179.0 ),
                ('min_bounds.rx', -2.5),
                
                # max_bounds
                ('max_bounds.px', 0.5),
                ('max_bounds.py', 0.2),
                ('max_bounds.pz', 3.5 ),
                ('max_bounds.rz', 2.5 ),
                ('max_bounds.ry', 179.0 ),
                ('max_bounds.rx', 2.5 ),
                
                # R_bodyVins_camNerf
                ('R_bodyVins_camNerf.rz', 0.0 ),
                ('R_bodyVins_camNerf.ry', 0.0 ),
                ('R_bodyVins_camNerf.rx', 3.14159256 )
            ]
        )

        # Get parameters
        self.min_bounds = {
            'px': self.get_parameter('min_bounds.px').get_parameter_value().double_value,
            'py': self.get_parameter('min_bounds.py').get_parameter_value().double_value,
            'pz': self.get_parameter('min_bounds.pz').get_parameter_value().double_value,
            'rz': self.get_parameter('min_bounds.rz').get_parameter_value().double_value,
            'ry': self.get_parameter('min_bounds.ry').get_parameter_value().double_value,
            'rx': self.get_parameter('min_bounds.rx').get_parameter_value().double_value,
        }

        self.max_bounds = {
            'px': self.get_parameter('max_bounds.px').get_parameter_value().double_value,
            'py': self.get_parameter('max_bounds.py').get_parameter_value().double_value,
            'pz': self.get_parameter('max_bounds.pz').get_parameter_value().double_value,
            'rz': self.get_parameter('max_bounds.rz').get_parameter_value().double_value,
            'ry': self.get_parameter('max_bounds.ry').get_parameter_value().double_value,
            'rx': self.get_parameter('max_bounds.rx').get_parameter_value().double_value,
        }

        self.R_bodyVins_camNerf = {
            'rz': self.get_parameter('R_bodyVins_camNerf.rz').get_parameter_value().double_value,
            'ry': self.get_parameter('R_bodyVins_camNerf.ry').get_parameter_value().double_value,
            'rx': self.get_parameter('R_bodyVins_camNerf.rx').get_parameter_value().double_value,
        }


        def declare_and_get_params(self):
            # Function to declare and get a nested parameter
            def declare_and_get_nested_param(base, params):
                for param in params:
                    self.declare_parameter(f'{base}.{param}', rclpy.Parameter.Type.DOUBLE)
                return {param: self.get_parameter(f'{base}.{param}').value for param in params}

            # Declare and get min_bounds parameters
            self.min_bounds = declare_and_get_nested_param('min_bounds', ['px', 'py', 'pz', 'rz', 'ry', 'rx'])
            self.get_logger().info(f'min_bounds: {self.min_bounds}')

            # Declare and get max_bounds parameters
            self.max_bounds = declare_and_get_nested_param('max_bounds', ['px', 'py', 'pz', 'rz', 'ry', 'rx'])
            self.get_logger().info(f'max_bounds: {self.max_bounds}')

            # Declare and get R_bodyVins_camNerf parameters
            self.r_body_vins_cam_nerf = declare_and_get_nested_param('R_bodyVins_camNerf', ['rz', 'ry', 'rx'])
            self.get_logger().info(f'R_bodyVins_camNerf: {self.r_body_vins_cam_nerf}')



        def get_and_print_params(self):
        # Function to get a nested parameter
            def get_nested_param(base, params):
                return {param: self.get_parameter(f'{base}.{param}').value for param in params}

            # Get min_bounds parameters
            min_bounds = get_nested_param('min_bounds', ['px', 'py', 'pz', 'rz', 'ry', 'rx'])
            self.get_logger().info(f'min_bounds: {min_bounds}')

            # Get max_bounds parameters
            max_bounds = get_nested_param('max_bounds', ['px', 'py', 'pz', 'rz', 'ry', 'rx'])
            self.get_logger().info(f'max_bounds: {max_bounds}')

            # Get R_bodyVins_camNerf parameters
            r_body_vins_cam_nerf = get_nested_param('R_bodyVins_camNerf', ['rz', 'ry', 'rx'])
            self.get_logger().info(f'R_bodyVins_camNerf: {r_body_vins_cam_nerf}')

        # Get and print the parameters

        #TODO activate as these are currently hardcoded???? is this fixed??
        #declare_and_get_params(self)

        self.previous_vio_pose = None
        self.nerf_pose = None
        self.all_pose_est = [] # plus 1 since we put in the initial pose before the first update
        self.img_msg = None
        
        # for now only have gt pose for llff dataset for inerf comparison and nerf-nav comparison
        self.gt_pose = None
        if not self.use_received_image:
            self.gt_pose = np.copy(self.nerf.obs_img_pose)
        
        self.check_params()

    def check_params(self):
        """
        Useful helper function to check if suspicious or invalid params are being used.
        TODO: Not all bad combinations of params are currently checked here.
        """

        if self.alpha_super_refine > self.alpha_refine:
            warnings.warn("alpha_super_refine is larger than alpha_refine, code will run but they are probably flipped by the user")
        
        if self.sampling_strategy != "random":
            warnings.warn("did not enter a valid sampling strategy. Currently the following are supported: random")

        if self.photometric_loss != "rgb":
            warnings.warn("did not enter a valid photometric loss. Currently the following are supported: rgb")
