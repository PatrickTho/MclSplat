import numpy as np
import gtsam

from multiprocessing import Lock
from locnerf import robot_motion_model

from geometry_msgs.msg import Pose

class ParticleFilter:

    def __init__(self, initial_particles):
        self.num_particles=len(initial_particles['position'])
        self.particles = initial_particles
        self.weights=np.ones(self.num_particles)
        self.particle_lock = Lock()

    def reduce_num_particles(self, num_particles):
        self.particle_lock.acquire()
        self.num_particles = num_particles
        self.weights = self.weights[0:num_particles]
        self.particles['position'] = self.particles['position'][0:num_particles]
        self.particles['rotation'] = self.particles['rotation'][0:num_particles]
        self.particle_lock.release()

    def predict_no_motion(self, p_x, p_y, p_z, r_x, r_y, r_z):
        self.particle_lock.acquire()
        self.particles['position'][:,0] += p_x *  np.random.normal(size = (self.particles['position'].shape[0]))
        self.particles['position'][:,1] += p_y * np.random.normal(size = (self.particles['position'].shape[0]))
        self.particles['position'][:,2] += p_z * np.random.normal(size = (self.particles['position'].shape[0]))

        # TODO see if this can be made faster
        print("ROT FORMAT")
        print(self.particles['rotation'][0])
        for i in range(len(self.particles['rotation'])):
            n1 = r_x * np.random.normal()
            #* np.random.normal()
            n2 = r_y * np.random.normal()
            n3 = r_z * np.random.normal()
            #* np.random.normal()
            #self.particles['rotation'][i] = self.particles['rotation'][i].retract(np.array([n1, n2, n3]))
            self.particles['rotation'][i]= gtsam.Rot3(self.particles['rotation'][i].retract(np.array([n1, n2, n3])).matrix())
        self.particle_lock.release()


    def predict_particles(self, previous_vio_pose, current_pose, p_x, p_y, p_z, r_x, r_y, r_z):
        self.particle_lock.acquire()
        previous_vio_pose = gtsam_to_pose(previous_vio_pose)
        current_vio_pose = current_pose
        for i in range(len(self.particles['rotation'])):
            #HIER WAHRSCHEINLICH
            pose = gtsam.Pose3(self.particles['rotation'][i], self.particles['position'][i])
            pose = gtsam_to_pose(pose)

            new_pose = robot_motion_model.sample_odom_motion_model(pose,
                current_vio_pose,
                previous_vio_pose,
               )

            #now i have the pose in gtsam format
            new_position = new_pose.translation()
            self.particles['position'][i][0] = new_position[0]
            self.particles['position'][i][1] = new_position[1]
            self.particles['position'][i][2] = new_position[2]
            self.particles['rotation'][i] = new_pose.rotation()
            print(new_pose, gtsam.Pose3(self.particles['rotation'][i], self.particles['position'][i]), flush=True)
            n1 = r_x * 0
            n2 = r_y * np.random.normal()
            n3 = r_z * 0
            self.particles['rotation'][i] = gtsam.Rot3(self.particles['rotation'][i].retract(np.array([n1, n2, n3])).matrix())

        self.particles['position'][:,0] += (p_x * np.random.normal(size = (self.particles['position'].shape[0])))
        self.particles['position'][:,1] += (p_y * np.random.normal(size = (self.particles['position'].shape[0])))
        self.particles['position'][:,2] += (p_z * np.random.normal(size = (self.particles['position'].shape[0])))
        
        self.particle_lock.release()


    def predict_with_delta_pose(self, delta_pose, p_x, p_y, p_z, r_x, r_y, r_z):
        self.particle_lock.acquire()

        # TODO see if this can be made faster
        delta_rot_t_tp1= delta_pose.rotation()
        for i in range(len(self.particles['rotation'])):
            # TODO do rotation in gtsam without casting to matrix
            pose = gtsam.Pose3(self.particles['rotation'][i], self.particles['position'][i])
            new_pose = gtsam.Pose3(pose.matrix() @ delta_pose.matrix())
            new_position = new_pose.translation()
            self.particles['position'][i][0] = new_position[0]
            self.particles['position'][i][1] = new_position[1]
            self.particles['position'][i][2] = new_position[2]
            self.particles['rotation'][i] = new_pose.rotation()

            n1 = r_x * np.random.normal()
            n2 = r_y * np.random.normal()
            n3 = r_z * np.random.normal()
            self.particles['rotation'][i] = gtsam.Rot3(self.particles['rotation'][i].retract(np.array([n1, n2, n3])).matrix())

        self.particles['position'][:,0] += (p_x * np.random.normal(size = (self.particles['position'].shape[0])))
        self.particles['position'][:,1] += (p_y * np.random.normal(size = (self.particles['position'].shape[0])))
        self.particles['position'][:,2] += (p_z * np.random.normal(size = (self.particles['position'].shape[0])))
        
        self.particle_lock.release()

    def update(self):
        # use fourth power
        self.weights[np.isnan(self.weights)] = 0.001
        self.weights = np.square(self.weights)
        self.weights = np.square(self.weights)

        # normalize weights
        sum_weights=np.sum(self.weights)
        # print("pre-normalized weight sum", sum_weights)
        self.weights=self.weights / sum_weights
            
        #resample
        self.particle_lock.acquire()
        choice = np.random.choice(self.num_particles, self.num_particles, p = self.weights, replace=True)
        temp = {'position':np.copy(self.particles['position'])[choice, :], 'rotation':np.copy(self.particles['rotation'])[choice]}
        self.particles = temp
        self.particle_lock.release()

    def compute_simple_position_average(self):
        # Simple averaging does not use weighted average or k means.
        avg_pose = np.average(self.particles['position'], axis=0)
        return avg_pose

    def compute_weighted_position_average(self):
        avg_pose = np.average(self.particles['position'], weights=self.weights, axis=0)
        print(avg_pose, flush = True)
        return avg_pose
    
    def compute_simple_rotation_average(self):
        # Simple averaging does not use weighted average or k means.
        # https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf section 5.3 Algorithm 1
        
        epsilon = 0.000001
        max_iters = 300
        rotations = self.particles['rotation']
        R = rotations[0]
        for i in range(max_iters):
            rot_sum = np.zeros((3))
            for rot in rotations:
                #?? is something going
                rot_sum = rot_sum  + gtsam.Rot3.Logmap(gtsam.Rot3(R.transpose() @ rot.matrix()))

            r = rot_sum / len(rotations)
            if np.linalg.norm(r) < epsilon:
                # print("rotation averaging converged at iteration: ", i)
                # print("average rotation: ", R)
                return R
            else:
                # TODO do the matrix math in gtsam to avoid all the type casting
                R = gtsam.Rot3(R.matrix() @ gtsam.Rot3.Expmap(r).matrix())
def gtsam_to_pose(gtsam_pose):
    geometry_pose = Pose()
    #print(gtsam_pose.translation()[0])
    #print(gtsam_pose.x)
    geometry_pose.position.x = gtsam_pose.translation()[0]
    geometry_pose.position.y = gtsam_pose.translation()[1]
    geometry_pose.position.z = gtsam_pose.translation()[2]
    
    # Get the quaternion components
    quat = gtsam_pose.rotation()
    quat = quat.toQuaternion()
    #print(quat)
    #print(quat.w())
    
    geometry_pose.orientation.x = quat.x()
    geometry_pose.orientation.y = quat.y()
    geometry_pose.orientation.z = quat.z()
    geometry_pose.orientation.w = quat.w()
    
    return geometry_pose

def gtsam_nerf_to_pose(gtsam_pose):
    geometry_pose = Pose()
    #print(gtsam_pose.translation()[0])
    #print(gtsam_pose.x)
    geometry_pose.position.x = gtsam_pose.translation()[0]
    geometry_pose.position.y = gtsam_pose.translation()[1]
    geometry_pose.position.z = gtsam_pose.translation()[2]
    
    # Get the quaternion components
    quat = gtsam_pose.rotation()
    quat = quat.toQuaternion()
    #print(quat)
    #print(quat.w())
    
    geometry_pose.orientation.x = quat.x()
    geometry_pose.orientation.y = quat.y()
    geometry_pose.orientation.z = quat.z()
    geometry_pose.orientation.w = quat.w()
    
    return geometry_pose