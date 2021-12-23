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

    roll = -2.74
    pitch = 0.04
    yaw = 0.0

    x_w = np.array([1, 0, 0])
    y_w = np.array([0, 1, 0])
    z_w = np.array([0, 0, 1])

    dh_table = np.array([[-np.pi/2, 0, 0, -np.pi/2],
                         [0,        1.77,   0,  0.08],
                         [np.pi/2,   0,  0,  0],
                         [0,                0,  0.52,  0],
                         [-np.pi/2,  0,      0,  0]])

    T_0tocam = Tmat(dh_table[0,:])
    T_0tofidp = Tmat(dh_table)
    T_0tofid = T_0tofidp @ H_rot_mat(roll, pitch, yaw)

    p_zero = np.array([0,0,0,1])
    unit_x = np.array([0.5,0,0,1])
    unit_y = np.array([0,0.5,0,1])
    unit_z = np.array([0,0,0.5,1])

    cam_0 = T_0tocam @ p_zero
    cam_x_vec = T_0tocam @ unit_x
    cam_y_vec = T_0tocam @ unit_y
    cam_z_vec = T_0tocam @ unit_z

    fidp_0 = T_0tofidp @ p_zero
    fidp_x_vec = T_0tofidp @ unit_x
    fidp_y_vec = T_0tofidp @ unit_y
    fidp_z_vec = T_0tofidp @ unit_z

    fid_0 = T_0tofid @ p_zero
    fid_x_vec = T_0tofid @ unit_x
    fid_y_vec = T_0tofid @ unit_y
    fid_z_vec = T_0tofid @ unit_z


    a = Arrow3D([0, cam_x_vec[0]], [0, cam_x_vec[1]], [0, cam_x_vec[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([0, cam_y_vec[0]], [0, cam_y_vec[1]], [0, cam_y_vec[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([0, cam_z_vec[0]], [0, cam_z_vec[1]], [0, cam_z_vec[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)

    a = Arrow3D([fidp_0[0], fid_x_vec[0]], [fidp_0[1], fid_x_vec[1]], [fidp_0[2], fid_x_vec[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([fidp_0[0], fid_y_vec[0]], [fidp_0[1], fid_y_vec[1]], [fidp_0[2], fid_y_vec[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([fidp_0[0], fid_z_vec[0]], [fidp_0[1], fid_z_vec[1]], [fidp_0[2], fid_z_vec[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)

    max_range = 2
    ax1.set_xlim3d(-1, max_range)
    ax1.set_ylim3d(-1, max_range)
    ax1.set_zlim3d(-1, max_range)
    # Give them a name:

    ax1.text(0.0, 0.0, -0.1, r'$0$', fontsize=5)
    ax1.text(cam_x_vec[0], cam_x_vec[1], cam_x_vec[2], r'$x_{cam}$', fontsize=5)
    ax1.text(cam_y_vec[0], cam_y_vec[1], cam_y_vec[2], r'$y_{cam}$', fontsize=5)
    ax1.text(cam_z_vec[0], cam_z_vec[1], cam_z_vec[2], r'$z_{cam}$', fontsize=5)

    ax1.text(fidp_0[0], fidp_0[1], fidp_0[2 ]-0.1, r'$fid$', fontsize=5)
    ax1.text(fid_x_vec[0], fid_x_vec[1], fid_x_vec[2], r'$x_{fid}$', fontsize=5)
    ax1.text(fid_y_vec[0], fid_y_vec[1], fid_y_vec[2], r'$y_{fid}$', fontsize=5)
    ax1.text(fid_z_vec[0], fid_z_vec[1], fid_z_vec[2], r'$z_{fid}$', fontsize=5)

    plt.show()