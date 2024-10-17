from matplotlib.markers import MarkerStyle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms

from locnerf.utils import show_img, img2mse, load_llff_data, get_pose, ssim_loss, ssim_combined, sam_score, fsim_score, ncc, load_nerfstudio_data, sam_rgb, ncc_rgb
from locnerf.full_nerf_helpers import load_nerf
from locnerf.render_helpers import render, to8b, get_rays, render_new
from locnerf.particle_filter import ParticleFilter

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer as ViewerState
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
from nerfstudio.cameras.cameras import Cameras, CameraType




from pathlib import Path


from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# part of this script is adapted from iNeRF https://github.com/salykovaa/inerf
# and NeRF-Pytorch https://github.com/yenchenlin/nerf-pytorch/blob/master/load_llff.py

class NeRF:
    #TODO refactor this for splatfacto!
    
    def __init__(self, nerf_params):
        # Parameters
        self.output_dir = './output/'
        self.data_dir = nerf_params['data_dir']
        self.model_name = nerf_params['model_name']
        self.obs_img_num = nerf_params['obs_img_num']
        self.batch_size = nerf_params['batch_size']
        self.factor = nerf_params['factor']
        #self.near = nerf_params['near']
        #self.far = nerf_params['far']
        self.spherify = False
        #self.kernel_size = nerf_params['kernel_size']
        #self.lrate = nerf_params['lrate']
        self.dataset_type = nerf_params['dataset_type']
        self.sampling_strategy = nerf_params['sampling_strategy']
        self.delta_phi, self.delta_theta, self.delta_psi, self.delta_x, self.delta_y, self.delta_z = [0,0,0,0,0,0]
        #self.no_ndc = nerf_params['no_ndc']
        #self.dil_iter = nerf_params['dil_iter']
        #self.chunk = nerf_params['chunk'] # number of rays processed in parallel, decrease if running out of memory
        self.bd_factor = nerf_params['bd_factor']

        print("dataset type:", self.dataset_type)
        #print("no ndc:", self.no_ndc)
        #TODO WORK IN PROGRESS!! MODEL LOAD ONLY ONCE!!
        #TODO make sure to always load correct model confi. This needs to be a PARAM. In the LLFF dataset it must be supported to load every specific config per dataset.
        #load_config="/home/student/catkin_ws/outputs/unnamed/splatfacto/2024-08-22_193711/config.yml"
        #print("jetz wirds model geladen", flush=True)
        #load_config=Path(load_config)
        #config, self.pipeline, _, step = eval_setup(
        #    load_config,
        #    eval_num_rays_per_chunk=None,
        #    test_mode="test",
            
        #)    
        if self.dataset_type == 'custom':
            print("self.factor", self.factor)
            self.focal = nerf_params['focal'] / self.factor
            self.H =  nerf_params['H'] / self.factor
            self.W =  nerf_params['W'] / self.factor
        
            # we don't actually use obs_img_pose when we run live images. this prevents attribute errors later in the code
            self.obs_img_pose = None

            self.H, self.W = int(self.H), int(self.W)

        else:
            #bug: this line of code gets a wrong directory. Why is it crashing

            #TODO TURN INTO PARAMS
            data_dir = "/home/student/data/nerfstudio/poster/images"
            dataparser_json = "/home/student/catkin_ws/outputs/unnamed/splatfacto/2024-09-06_184116/dataparser_transforms.json"
            json_path = "/home/student/data/nerfstudio/poster/transforms.json"
            #dataparser_json = "/home/student/outputs/unnamed/splatfacto/2024-09-02_201612/dataparser_transforms.json"
                                

            self.obs_img, self.obs_img_pose, self.image_name = load_nerfstudio_data(json_path, self.obs_img_num, data_dir,  dataparser_json)
        
            self.focal = 900
            print(self.obs_img_pose, flush=True)

            
            self.H, self.W = 224, 224
            self.obs_img = (np.array(self.obs_img)).astype(np.float32)
            self.obs_img_noised = self.obs_img

        load_config = self.model_name
        #load_config = "/home/student/catkin_ws/outputs/unnamed/splatfacto/2024-09-06_184116/config.yml"
        #TODO make sure to always load correct model confi. This needs to be a PARAM. In the LLFF dataset it must be supported to load every specific config per dataset.
        #load_config="/home/student/21.08_Splat/21.08/21.08.rescaled/outputs/unnamed/splatfacto/2024-08-26_181155/config.yml"
        print("jetz wirds model geladen", flush=True)
        load_config=Path(load_config)
        #Testen das model global zu machen!
        self.config, self.pipeline, _, step = eval_setup(
                load_config,
                eval_num_rays_per_chunk=None,
                test_mode="test",
                
        )

    

    def get_loss(self, particles, photometric_loss='rgb', save_camera = True):
        #target_s = self.obs_img_noised[batch[:, 1], batch[:, 0]] # TODO check ordering here
        target_s = self.obs_img_noised
        #print(target_s.shape)

        start_time = time.time()
        
        batch = 224*224
        num_pixels = len(particles) * batch
        #print(num_pixels)

        
        
        rendered_pixels = render_new(
            self.pipeline, 
            self.H, self.W, self.focal, c2w=particles,
                                        ndc=True)

  
        losses = []
        #print(target_s.shape)
        #resize_transform = transforms.Resize((224, 224))
        #target_s = target_s.permute(2,0 , 1)  # Change to (H, W, C)
        #resized_image = resize_transform(target_s)

        #resized_image = resized_image.permute(1, 2, 0)
        resized_image = target_s
        resized_image = cv2.resize(target_s,( 224, 224),interpolation= cv2.INTER_AREA)
        #cv2.imwrite('camera.jpg', resized_image)
        resized_image = (np.array(resized_image) / 255.0).astype(np.float32)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        #print(resized_image.shape, flush=True)

        #print(torch.tensor(rendered_pixels[0]).shape)
        #plt.imshow(self.obs_img_noised)
        #plt.show()
        #if save_camera == True:
        #real_save= cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('actualImageLossy.jpg', real_save * 255)
        #save_img = (rendered_pixels[0] * 255).astype(np.uint8)
        
        #cv2.imwrite('rendered_particle.jpg', save_img)
        #print("saved image matrix", flush=True)
        #print(particles[0], flush=True)
        if photometric_loss != 'features':
            print(photometric_loss, flush=True)

            # Use this method if you need to use CLAHE for different Lightning conditions
            #lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
            # lab_planes = cv2.split(lab)
            #clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))


            #lab[:,:,0] = clahe.apply(lab[:,:,0])

            #img = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

            #lab_planes[0] = clahe.apply(lab_planes[0])
            #lab = cv2.merge(lab_planes)
            #resized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            #resized_image = torch.Tensor(img).to(device)
            for i in range(len(particles)):
                rgb = rendered_pixels[i]
                #print(len(rgb))
                rgb = (rgb).astype(np.float32)
                #print(len(rgb))
                #print(rgb.shape)

                if photometric_loss == 'rgb':


                    resized_image = torch.Tensor(resized_image).to(device)

                    #lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
                    #lab_planes = cv2.split(lab)
                    #clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                    #lab_planes[0] = clahe.apply(lab_planes[0])
                    #lab = cv2.merge(lab_planes)
                    #lab[:,:,0] = clahe.apply(lab[:,:,0])
                    #rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    rgb = torch.Tensor(rgb).to(device)
                    loss = img2mse(rgb, resized_image)

                if photometric_loss == 'ssim':
                    #print("correct_metric ", flush = True)
                    loss = ssim_loss(rgb * 255, resized_image * 255)

                if photometric_loss == 'ssim+rgb':
                    loss = ssim_combined(rgb * 255, resized_image * 255)
                if photometric_loss == 'sam':
                    loss = sam_score(resized_image , rgb)
                if photometric_loss == 'fsim':
                    loss = fsim_score(rgb * 255, resized_image * 255)
                if photometric_loss == 'ncc':
                    loss = ncc(rgb * 255, resized_image * 255)
                if photometric_loss == 'sam+rgb':
                    loss = sam_rgb(rgb * 255, resized_image * 255)
                if photometric_loss == 'ncc+rgb':
                    loss = ncc_rgb(rgb * 255, resized_image * 255)


                #else:
                    # TODO throw an error            
                #    print("DID NOT ENTER A VALID LOSS METRIC")
                losses.append(loss.item())
                #if loss.item() != 0.0:
                #   print("Success")
        else:

            image_array = []
            for i in range(len(particles)):
                rgb = rendered_pixels[i]
                rgb = (rgb * 255).astype(np.uint8)
                image_array.append(rgb)
            image_array = torch.stack(image_array)    
            reference_image = target_s.unsqueeze(0)
            losses = calculate_perceptual_loss(reference_image, image_array)



       
        #losses.append(loss.item())
        return losses, nerf_time
    
    
    def save_nerf_image(self, nerf_pose, update_num):
        bar = np.array(nerf_pose).astype(np.float32)
        print(bar.shape, flush=True)
        bar = np.expand_dims(bar, axis=0)
        #oss_poses_np = np.array(loss_poses_np.astype(np.float32))
        print(bar.shape, flush=True)
        bar_tensor = torch.Tensor(bar).to(device)
        print(bar_tensor.shape, flush=True)
        #is this causing issues????
        #bar_tensor = bar_tensor.to(device)
        foo = render_new(
            self.pipeline, 
            self.H, self.W, self.focal, c2w=bar_tensor,
                                        ndc=True)
        save_img = (foo[0] * 255).astype(np.uint8)
        save_img= cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite('test' + str(update_num) +'.jpg', save_img)

        target_s = self.obs_img_noised
        resized_image = cv2.resize(target_s,( 224, 224),interpolation= cv2.INTER_AREA)
        resized_image = (np.array(resized_image) / 255.0).astype(np.float32)
        cv2.imwrite('actualImageLossy' + str(update_num) +'.jpg', resized_image * 255)
  


"""

                    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
                    #lab_planes = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                    #lab_planes[0] = clahe.apply(lab_planes[0])
                    #lab = cv2.merge(lab_planes)
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                    rgb = torch.Tensor(rgb).to(device)
                    loss = img2mse(rgb, resized_image)"""
