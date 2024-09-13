import torch
import numpy as np
import imageio
import cv2
import json
import os
from math import atan2, asin, pi
from numpy.linalg import inv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from geometry_msgs.msg import Quaternion
import random
import gtsam

from image_similarity_measures.quality_metrics import ssim, fsim, sam
# most of this script is adapted from iNeRF https://github.com/salykovaa/inerf
# and NeRF-Pytorch https://github.com/yenchenlin/nerf-pytorch/blob/master/load_llff.py




rot_psi_old = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta_old = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi_old = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])



rot_phi = lambda phi: np.array([
        [1, 0, 0 ],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
        ])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)]
        ])

rot_psi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
        ])

trans_t = lambda x,y,z: np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]])

def load_blender(data_dir, model_name, obs_img_num, half_res, white_bkgd, *kwargs):

    with open(os.path.join(data_dir + str(model_name) + "/obs_imgs/", 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    img_path =  os.path.join(data_dir + str(model_name) + "/obs_imgs/", frames[obs_img_num]['file_path'] + '.png')
    img_rgba = imageio.imread(img_path)
    img_rgba = (np.array(img_rgba) / 255.).astype(np.float32) # rgba image of type float32
    H, W = img_rgba.shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    if white_bkgd:
        img_rgb = img_rgba[..., :3] * img_rgba[..., -1:] + (1. - img_rgba[..., -1:])
    else:
        img_rgb = img_rgba[..., :3]

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    img_rgb = np.asarray(img_rgb*255, dtype=np.uint8)
    obs_img_pose = np.array(frames[obs_img_num]['transform_matrix']).astype(np.float32)
    phi, theta, psi, x, y, z = kwargs
    start_pose =  trans_t(x, y, z) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
    #start_pose =  trans_t(x, y, z) @ rot_psi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_phi(psi/180.*np.pi)  @ obs_img_pose

    return img_rgb, [H, W, focal], start_pose, obs_img_pose # image of type uint8

def get_pose(phi, theta, psi, x, y, z, obs_img_pose, center_about_true_pose, use_nerfstudio_convention):

    #use_nerfstudio_convention = True
    if use_nerfstudio_convention:
        #print("CORRECT!!!!!!!!!!!!!!!", flush=True)
        adjust = np.deg2rad(90)
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
        psi = np.deg2rad(psi)
        rotation = rot_phi(adjust)@rot_theta(theta) @ rot_phi(phi)@ rot_psi(psi)
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = [x, y , z]
        
    elif center_about_true_pose:
        #print("IIIIIIIIIIINCORRECT!!!!!!!!!!!!!!!", flush=True)

        # print("recentering")
        print(obs_img_pose)
        pose = trans_t(x, y, z) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
    else:
        print("correct branch", flush=True)
        #pose = trans_t(x, y, z) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi)
        #pose = trans_t(x, y, z) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
        #print("correct transform function", flush=True)
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
        psi = np.deg2rad(psi)
        rotation =  rot_psi(psi) @ rot_theta(theta) @ rot_phi(phi) 
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = [x, y , z]
        
    return pose

def rgb2bgr(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def show_img(title, img_rgb):  # img - rgb image
    img_bgr = rgb2bgr(img_rgb)
    cv2.imshow(title, img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy # pixel coordinates


# Misc
#
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Load llff data

########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    print(bds.shape)
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            #change to apply_gamma=false from ignoregamam=True
            return imageio.imread(f, apply_gamma=False)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    # print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs
def ssim_loss(render, camera):
    score = ssim(render, camera)
    #normalize value to 0 to 1
    score = 1 - (1 + score ) / 2  
    score = torch.tensor(score)

    return score
def ncc_rgb(camera, render):
    "implement"
def ncc(camera, render):
    camera_centered = camera - np.mean(camera)
    render_centered = render - np.mean(render)

    # Compute the numerator of the NCC
    
    ncc_numerator = np.sum(camera_centered * render_centered)

    # Compute the denominators
    ncc_denominator = np.sqrt(np.sum(camera_centered ** 2)) * np.sqrt(np.sum(render_centered ** 2))

    # Calculate the NCC
    ncc = ncc_numerator / ncc_denominator
    ncc_score = 1- (1+ncc)/2
    score= torch.tensor(ncc_score)
    return score
def sam_rgb(camera, render):
    score = sam(camera, render)
    print(score)
    score = score/180
    camera = torch.Tensor(camera/255).to('cpu')
    render = torch.Tensor(render/255).to('cpu')
    mse_loss = img2mse(camera, render)
    value = mse_loss.item()
    combined = 0.7 * score + 0.3 * value
    return combined

def sam_score(camera, render):
    score = sam(camera, render)
    print(score)
    score = score/180
    if score == 0.0:
        score = 2.0
    score = torch.tensor(score).to('cpu')
    return score

def fsim_score(camera, render):
    score = fsim(camera, render)
    score = 1 - score
    score =torch.tensor(score)
    return score

def ssim_combined(camera, render):
    ssim_score = ssim(render, camera)
    #normalize value to 0 to 1
    ssim_score = 1 - (1 + ssim_score ) / 2  

    camera = torch.Tensor(camera/255).to('cpu')
    render = torch.Tensor(render/255).to('cpu')
    mse = img2mse(render, camera)
    value = mse.item()
    #print(value, flush=True)
    combined = 0.7 * ssim_score + 0.3 * value
  
    return combined

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]

    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, bds


def load_llff_data(data_dir, model_name, obs_img_num, *kwargs, factor=8, recenter=True, bd_factor=.75, spherify=False):
    poses, bds, imgs = _load_data(data_dir + "/", factor=factor)  # factor=8 downsamples original imgs by 8x
    print('Loaded', data_dir + str(model_name) + "/", bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    print("BDS MIN", flush=True)
    print(bds.min(), flush = True)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    #poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, bds = spherify_poses(poses, bds)
    print("loading image number: ", obs_img_num)
    #images = images.astype(np.float32)
    images = np.asarray(images * 255, dtype=np.uint8)
    poses = poses.astype(np.float32)
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    obs_img = images[obs_img_num]
    print(len(obs_img))
    print(obs_img.shape)
    obs_img_pose = np.concatenate((poses[obs_img_num], np.array([[0,0,0,1.]])), axis=0)
    print(obs_img_pose.shape, flush=True)
    phi, theta, psi, x, y, z = kwargs
    # is there something going wrong here??
    start_pose = rot_phi_old(phi/180.*np.pi) @ rot_theta_old(theta/180.*np.pi) @ rot_psi_old(psi/180.*np.pi) @ trans_t(x, y, z) @ obs_img_pose
        # Save only the obs_img_pose in JSON format
    obs_img_pose_json = {
        "matrix": obs_img_pose.tolist()
    }
    
    # Change the filename to include obs_img_num
    filename = f"obs_img_pose_{obs_img_num}.json"
    with open(filename, "w") as f:
        json.dump(obs_img_pose_json, f)
    return obs_img, hwf, start_pose, obs_img_pose, bds

#This is AI Generated Code, using Github Copilot

def calculate_perceptual_loss(reference_image, image_array, batch_size=150):
    """
    Berechnet den Perceptual Loss fuer jedes Bild im image_array im Vergleich zum reference_image.
    
    :param reference_image: Das Referenzbild (Tensor)
    :param image_array: Ein Array von Bildern (Tensor) der Form (batch_size, C, H, W)
    :param batch_size: Die egereese der Batches fur die Verarbeitung
    :return: Ein Tensor mit den Verlustwerten fur jedes Bild im image_array
    """
    vgg = models.vgg16(pretrained=True).features.eval()
    layers = nn.Sequential(*list(vgg)[:16])
    for param in layers.parameters():
        param.requires_grad = False

    reference_features = layers(reference_image.unsqueeze(0)).detach()
    losses = []

    # Verarbeite die Bilder in Batches
    for i in range(0, len(image_array), batch_size):
        batch = image_array[i:i+batch_size]
        batch_features = layers(batch).detach()
        batch_losses = torch.mean((reference_features - batch_features) ** 2, dim=[1, 2, 3])
        losses.extend(batch_losses)

    return torch.tensor(losses)





def load_nerfstudio_data(jsonpath=None, obs_img_num = 0,  filepath = None, dataparserpath=None, factor=None, width=None, height=None, load_imgs=True):
    
    test_transform= np.array([[
    0.09486033,  0.38717473, -0.91711354,  0.11318887],
 [ 0.9853164,   0.09486033,  0.14196165, -0.09984465],
 [ 0.14196165, -0.91711354, -0.37249112,  0.013491  ],
   [ 0. ,         0.  ,        0.    ,      1.        ]
 ])
    test_matrix =  np.array([[ 0.3130143,  -0.79339266,
                               -0.52206335, -0.799124  ],
 [-0.47521135,  0.34509486, -0.80937241, -2.57343738],
  [ 0.82231151,  0.50143557, -0.26900957, -2.23334674],
 [ 0. ,         0.  ,        0.    ,      1.        ]])
    
    """
 [[-0.90845021 -0.40152283 -0.11617932  1.08924634]
 [ 0.38007611 -0.67782236 -0.62936396 -1.44840134]
 [ 0.17395506 -0.6159028   0.76837711  3.09208214]
  [ 0.          0.          0.          1.        ]]



"""



    """
    [0.8302655609888863, 0.18542754472397374, -0.5256193716877726, -2.4762210134715823, 
    -0.5552775777329589, 0.19358150269322377, -0.8088219912205971, -2.3757947784543973, 
    -0.04822768813480355, 0.9634016958012231, 0.263687812809959, -0.4508942305015159, 
    0.0, 0.0, 0.0, 1.0]
    """
    weird_transform = [
        [
            0,
            1,
            0,
            0
        ],
        [
            1,
            0,
            0,
            0
        ],
        [
            0,
            0,
            -1,
            0
        ],
        [0,0,0,1]
    ]
    print("HUGE TEST!!::", flush=True)
    reapply = weird_transform @ test_matrix
    out_test =    test_transform @ reapply
    #print(out_test, flush=True)

    print("checker", flush=True)
    print(test_transform @ weird_transform @ test_matrix )
        # Load the JSON data
    with open(jsonpath, 'r') as file:
        data = json.load(file)

    with open(dataparserpath, 'r') as file:
        transforms = json.load(file)

    transform = transforms['transform']
    scale = transforms["scale"]
    print(transform, flush=True)
    transform = np.array(transform)
    transform = np.vstack((transform, [0, 0, 0, 1]))
    print("transform", flush=True)   
    print(transform, flush=True)
    #gtsam_matrix = gtsam.Pose3(transform)
    random_frame  = data['frames'][obs_img_num]
    matrix = random_frame['transform_matrix']
    matrix = np.array(matrix)
    #matrix = np.vstack((matrix,[0,0,0,1]))
    print("matrix", flush=True)
    print(matrix, flush= True)
    
    #matrix = gtsam.Pose3(matrix)
    image_frame = transform @ weird_transform @ matrix 
    # Extract the transformatio_matn matrix and the image file path
    image_frame[:3, 3] *= scale
    image_path = random_frame['file_path']
    # Extract the transformation matrices
    matrices = []
    image_path = image_path.split('/')[-1]
    image = cv2.imread(filepath + "/"+image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    center = image.shape
    x = center[1]/2 - 1080/2
    y = center[0]/2 - 1080/2

    crop_img = image[int(y):int(y+1080), int(x):int(x+1080)]

    cv2.imwrite("loadedImg.jpg", crop_img)
    image = crop_img
    image_array = np.array(crop_img)
    #for frame in data['frames']:
    #    matrix = frame['transform_matrix']
    #    matrices.append(matrix)

    # Save the matrices to a file
    #with open('matrices.txt', 'w') as file:
    #    for matrix in matrices:
    #        file.write(str(matrix) + '\n')
    print("this is image matrix from json", flush=True)
    print(image_frame, flush=True)
    return image_array, image_frame, image_path

    print("Transformation matrices extracted and saved to matrices.txt")

#The following functions are taking from https://github.com/debbynirwan/mcl/blob/master/mcl/util.py
#licensing:
"""Utility Module
Description:
    A group of useful common functions used by other nodes
License:
    Copyright 2021 Debby Nirwan
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specissim_lossfic language governing permissions and
    limitations under the License.
"""


def yaw_from_quaternion(orientation: Quaternion):
    _, yaw, _ = euler_from_quaternion(orientation.x,
                                      orientation.y,
                                      orientation.z,
                                      orientation.w)
    return yaw






def euler_from_quaternion(x, y, z, w):
    """
    https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def euler_to_quaternion(yaw, pitch, roll) -> Quaternion:
    # https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return Quaternion(x=qx, y=qy, z=qz, w=qw)


def angle_diff(angle1: float, angle2: float) -> float:
    d1 = angle1 - angle2
    d2 = 2 * pi - abs(d1)
    if d1 > 0.0:
        d2 *= -1.0

    if abs(d1) < abs(d2):
        return d1
    else:
        return d2