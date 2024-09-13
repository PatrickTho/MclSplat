"""Motion Model
Description:
    Odometry Motion Model
License:
    Copyright 2021 Debby Nirwan
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import numpy as np

from geometry_msgs.msg import Pose, Point
from math import atan2, sqrt, cos, sin
from locnerf import utils
from scipy.stats import norm
import gtsam
from locnerf.utils import rot_theta, euler_from_quaternion


MOVED_TOO_CLOSE = 0.01


def sample_odom_motion_model(prev_true_xt: Pose,
                             latest_odom: Pose,
                             prev_odom: Pose,
                             ) -> float:
    if prev_odom == latest_odom:
        #print("not odoming anything", flush = True)
        return prev_true_xt

    d1, dt, d2 = _calculate_pose_delta(latest_odom, prev_odom)
    #d1 = 0
    #dt= 0
    #d2 = 0
    #perhaps d1 d2 in wrong direction?

    #alpha1 = cfg['alpha1']
    #alpha2 = cfg['alpha2']
    #alpha3 = cfg['alpha3']
    #alpha4 = cfg['alpha4']

    #std_dev_d1 = sqrt((alpha1 * (d1**2)) + (alpha2 * (dt**2)))
    #std_dev_dt = sqrt((alpha3 * (dt**2)) + (alpha4 * (d1**2)) + (alpha4 * (d2**2)))
    #std_dev_d2 = sqrt((alpha1 * (d2**2)) + (alpha2 * (dt**2)))

    noised1 = 0.0
    noisedt = 0.0
    noised2 = 0.0
    #if std_dev_d1 > 0:
    #    noised1 = np.random.normal(scale=std_dev_d1)
    #if std_dev_dt > 0:
    #    noisedt = np.random.normal(scale=std_dev_dt)
    #if std_dev_d2 > 0:
    #    noised2 = np.random.normal(scale=std_dev_d2)

    #t_d1 = utils.angle_diff(d1, noised1)
    t_dt = dt + noisedt
    #t_d2 = utils.angle_diff(d2, noised2)

    curr_x = prev_true_xt.position.x
    curr_y = prev_true_xt.position.y
    curr_z = prev_true_xt.position.z
    

    _, _, curr_yaw = utils.euler_from_quaternion(prev_true_xt.orientation.x, prev_true_xt.orientation.z, prev_true_xt.orientation.y, prev_true_xt.orientation.w)

    #todo hier vl -d1 # noch weiter gesagt sind vl cos und sin vertauscht..
    x = curr_x + t_dt * cos(curr_yaw + d1)
    z = curr_z + t_dt * sin(curr_yaw + d1)
    yaw = curr_yaw + d1 + d2

    position = Point(x=x, y=curr_y, z=z)

    position = gtsam.Point3(x, curr_y, z)  # y is set to 0.0 for a 2D plane

    # Create a Rot3 object for the orientation from the yaw angle
    #?????
    mat=np.eye(4,4)




    orientation = gtsam.Rot3(rot_theta(yaw))

    
    #orientation = gtsam.Rot3.matrix(orientation)  # Create a rotation around the Z-axis
    #print(orientation)

    # Return a gtsam Pose3 object
    return gtsam.Pose3(orientation, position)
    #orientation = utils.euler_to_quaternion(yaw, 0, 0)

    #return Pose(position=position, orientation=orientation)


def odom_motion_model(true_xt: Pose, prev_true_xt: Pose,
                      latest_odom: Pose, prev_odom: Pose,px_noise, py_noise, pz_noise, rot_x_noise, rot_y_noise, rot_z_noise) -> float:
    d1, dt, d2 = _calculate_pose_delta(latest_odom, prev_odom)
    t_d1, t_dt, t_d2 = _calculate_pose_delta(true_xt, prev_true_xt)
    print("EXECUTING ODOM MODEL!!", flush = True)
    #alpha1 = cfg['alpha1']
    #alpha2 = cfg['alpha2']
    #alpha3 = cfg['alpha3']
    #alpha4 = cfg['alpha4']
    #p1 = norm(loc=d1 - t_d1, scale=sqrt((alpha1 * (t_d1**2)) + (alpha2 * (t_dt**2)))).pdf(d1 - t_d1)
    #p2 = norm(loc=dt - t_dt, scale=sqrt((alpha3 * (t_dt**2)) + (alpha4 * (t_d1**2)) + (alpha4 * (t_d2**2)))).pdf(dt - t_dt)
    #p3 = norm(loc=d2 - t_d2, scale=sqrt((alpha1 * (t_d2**2)) + (alpha2 * (t_dt**2)))).pdf(d2 - t_d2)

    return p1 * p2 * p3


def _calculate_pose_delta(xt: Pose, prev_xt: Pose):
    x = prev_xt.position.x
    y = prev_xt.position.y
    _, _, theta = utils.euler_from_quaternion(prev_xt.orientation.x,prev_xt.orientation.y, prev_xt.orientation.z, prev_xt.orientation.w )
    x_prime = xt.position.x
    y_prime = xt.position.y
    _,_,theta_prime = utils.euler_from_quaternion(xt.orientation.x,xt.orientation.y, xt.orientation.z, xt.orientation.w )

    delta_translation = sqrt(((x - x_prime) ** 2) + ((y - y_prime) ** 2))
    delta_rotation1 = 0.0
    if delta_translation > MOVED_TOO_CLOSE:
        print("ACTUALLY GETTING ROT1", flush=True)
        delta_rotation1 = utils.angle_diff(atan2(y_prime - y, x_prime - x), theta)
    delta = utils.angle_diff(theta_prime, theta)
    delta_rotation2 = utils.angle_diff(delta, delta_rotation1)

    return delta_rotation1, delta_translation, delta_rotation2