import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import math as m
from numpy.linalg import inv
from func_dh_table import Tmat

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def Rx(theta):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])


def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def Ry(theta):
    return np.array([[m.cos(theta), 0, m.sin(theta), 0],
                     [0, 1, 0, 0],
                     [-m.sin(theta), 0, m.cos(theta), 0],
                     [0,    0,  0,  1]])



def H_rot_mat(roll, pitch, yaw):
    mat1 = Rx(roll)
    mat2 = Ry(pitch)
    mat3 = Rz(yaw)

    # euler angle matrix
    rot_mat = mat3 @ mat2 @ mat1
    # rot_mat = mat1 @ mat2 @ mat3
    return rot_mat


def roted_frame(roll, pitch, yaw):
    mat1 = Rx(roll)
    mat2 = Ry(pitch)
    mat3 = Rz(yaw)

    rot_mat = mat3 @ mat2 @ mat1

    roted_x = rot_mat @ x
    roted_y = rot_mat @ y
    roted_z = rot_mat @ z

    return roted_x, roted_y, roted_z


if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=10, arrowstyle='->', shrinkA=0, shrinkB=0)

    roll = -2.69
    pitch = 0.03
    yaw = 0.0
    x_offset = 0.07
    y_offset = 0.39
    z_offset = 1.91

    x_w = np.array([1, 0, 0])
    y_w = np.array([0, 1, 0])
    z_w = np.array([0, 0, 1])

# dh -> theta, d ,a, alpha

    dh_table = np.array([[-np.pi/2, 0, 0, -np.pi/2],
                         [0,        z_offset,   x_offset, 0],
                         [np.pi/2,   0,  0,  0],
                         [0,                0,  y_offset,  0],
                         [-np.pi/2,  0,      0,  0]])

    dh_base_to_fid = np.array([[0, 0.12, 0.105, 0],
                                [0, 0, 1.305, 0],
                                [-np.pi/2, 0, 0, np.pi/2]])

    # T_0tocam = Tmat(dh_table[0,:])
    T_0tofidp = Tmat(dh_table)
    T_0tofid = T_0tofidp @ H_rot_mat(roll, pitch, yaw)

    T_camtofid = Tmat(dh_table[1:,:]) @ H_rot_mat(roll, pitch, yaw)


    T_base_to_laser = Tmat(dh_base_to_fid[0, :])
    T_base_to_fid = Tmat(dh_base_to_fid)
    T_fidtocam = np.linalg.inv(T_camtofid)

    T_base_to_cam = T_base_to_fid @ T_fidtocam

    p_zero = np.array([0,0,0,1])
    unit_x = np.array([0.3,0,0,1])
    unit_y = np.array([0,0.3,0,1])
    unit_z = np.array([0,0,0.3,1])

    base_x = unit_x
    base_y = unit_y
    base_z = unit_z
    
    print('T_base_to_cam = ')
    print(T_base_to_cam)

    cam_zero = T_base_to_cam @ p_zero
    cam_x = T_base_to_cam @ unit_x
    cam_y = T_base_to_cam @ unit_y
    cam_z = T_base_to_cam @ unit_z

    laser_zero = T_base_to_laser @ p_zero
    laser_x = T_base_to_laser @ unit_x
    laser_y = T_base_to_laser @ unit_y
    laser_z = T_base_to_laser @ unit_z

    fid_zero = T_base_to_fid @ p_zero
    fid_x = T_base_to_fid @ unit_x
    fid_y = T_base_to_fid @ unit_y
    fid_z = T_base_to_fid @ unit_z

    a = Arrow3D([0, base_x[0]], [0, base_x[1]], [0, base_x[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([0, base_y[0]], [0, base_y[1]], [0, base_y[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([0, base_z[0]], [0, base_z[1]], [0, base_z[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)

    a = Arrow3D([laser_zero[0], laser_x[0]], [laser_zero[1], laser_x[1]], [laser_zero[2], laser_x[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([laser_zero[0], laser_y[0]], [laser_zero[1], laser_y[1]], [laser_zero[2], laser_y[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([laser_zero[0], laser_z[0]], [laser_zero[1], laser_z[1]], [laser_zero[2], laser_z[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)

    a = Arrow3D([fid_zero[0], fid_x[0]], [fid_zero[1], fid_x[1]], [fid_zero[2], fid_x[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([fid_zero[0], fid_y[0]], [fid_zero[1], fid_y[1]], [fid_zero[2], fid_y[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([fid_zero[0], fid_z[0]], [fid_zero[1], fid_z[1]], [fid_zero[2], fid_z[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)

    a = Arrow3D([cam_zero[0], cam_x[0]], [cam_zero[1], cam_x[1]], [cam_zero[2], cam_x[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([cam_zero[0], cam_y[0]], [cam_zero[1], cam_y[1]], [cam_zero[2], cam_y[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([cam_zero[0], cam_z[0]], [cam_zero[1], cam_z[1]], [cam_zero[2], cam_z[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)

    max_range = 2
    ax1.set_xlim3d(-1, max_range)
    ax1.set_ylim3d(-1, max_range)
    ax1.set_zlim3d(-1, max_range)
    # Give them a name:

    ax1.text(0.0, 0.0, -0.1, r'$base$', fontsize=5)
    ax1.text(base_x[0], base_x[1], base_x[2], r'$x_{base}$', fontsize=5)
    ax1.text(base_y[0], base_y[1], base_y[2], r'$y_{base}$', fontsize=5)
    ax1.text(base_z[0], base_z[1], base_z[2], r'$z_{base}$', fontsize=5)

    ax1.text(laser_zero[0], laser_zero[1], laser_zero[2]-0.1, r'$laser$', fontsize=5)
    ax1.text(laser_x[0], laser_x[1], laser_x[2], r'$x_{laser}$', fontsize=5)
    ax1.text(laser_y[0], laser_y[1], laser_y[2], r'$y_{laser}$', fontsize=5)
    ax1.text(laser_z[0], laser_z[1], laser_z[2], r'$z_{laser}$', fontsize=5)

    ax1.text(fid_zero[0], fid_zero[1], fid_zero[2]-0.1, r'$fid$', fontsize=5)
    ax1.text(fid_x[0], fid_x[1], fid_x[2], r'$x_{fid}$', fontsize=5)
    ax1.text(fid_y[0], fid_y[1], fid_y[2], r'$y_{fid}$', fontsize=5)
    ax1.text(fid_z[0], fid_z[1], fid_z[2], r'$z_{fid}$', fontsize=5)

    ax1.text(cam_zero[0], cam_zero[1], cam_zero[2]-0.1, r'$cam$', fontsize=5)
    ax1.text(cam_x[0], cam_x[1], cam_x[2], r'$x_{cam}$', fontsize=5)
    ax1.text(cam_y[0], cam_y[1], cam_y[2], r'$y_{cam}$', fontsize=5)
    ax1.text(cam_z[0], cam_z[1], cam_z[2], r'$z_{cam}$', fontsize=5)

    plt.show()