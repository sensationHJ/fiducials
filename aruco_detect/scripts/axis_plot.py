import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import math as m
from numpy.linalg import inv

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])  # x coordinate of points in inside surface
    y = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])    # y coordinate of points in inside surface
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]])                # z coordinate of points in inside surface
    return x, y, z

import math as m
  
def Rx(theta):
  return np.array([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.array([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def func_rot_mat(roll, pitch, yaw):
    mat1 = Rx(roll)
    mat2 = Ry(pitch)
    mat3 = Rz(yaw)
    
    # euler angle matrix
    rot_mat = mat3 @ mat2 @ mat1
    # rot_mat = mat1 @ mat2 @ mat3
    return rot_mat

def roted_frame(roll, pitch, yaw, x, y, z):

    mat1 = Rx(roll)
    mat2 = Ry(pitch)
    mat3 = Rz(yaw)
    
    rot_mat = mat3 @ mat2 @ mat1

    roted_x = rot_mat @  x
    roted_y = rot_mat @  y
    roted_z = rot_mat @  z

    return roted_x, roted_y, roted_z

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=10, arrowstyle='->', shrinkA=0, shrinkB=0)

    roll = -2.74
    pitch = 0.04
    yaw = 0.0

    x_w = np.array([1,0,0])
    y_w = np.array([0,1,0])
    z_w = np.array([0,0,1])

    T0_1 = np.array([0.08, 0.52, 1.77])

    rot_w_to_cam = Rz(np.deg2rad(-90)) @ Rx(np.deg2rad(-90))
    rot_cam_to_w = np.linalg.inv(rot_w_to_cam)

    rot_cam_to_w2 = Rx(np.deg2rad(90)) @ Rz(np.deg2rad(90))
    print(np.round(rot_w_to_cam,2))
    print(np.round(rot_cam_to_w,2))
    print(np.round(rot_cam_to_w2,2))

    rot_cam_to_fid = func_rot_mat(roll, pitch, yaw)
    rot_fid_to_cam = np.linalg.inv(rot_cam_to_fid)
    x_cam_0 = rot_w_to_cam @ x_w
    y_cam_0 = rot_w_to_cam @ y_w
    z_cam_0 = rot_w_to_cam @ z_w


    # x_fid_0 = rot_cam_to_w @ np.array([T0_1[0], 0, 0]) 
    # y_fid_0 = rot_cam_to_w @ np.array([0, T0_1[1], 0])
    # z_fid_0 = rot_cam_to_w @ np.array([0, 0, T0_1[2]])
    x_fid_0 = rot_w_to_cam @ np.array([T0_1[0], 0, 0]) 
    y_fid_0 = rot_w_to_cam @ np.array([0, T0_1[1], 0])
    z_fid_0 = rot_w_to_cam @ np.array([0, 0, T0_1[2]])


    print(np.round(x_fid_0,2))
    print(np.round(y_fid_0,2))
    print(np.round(z_fid_0,2))

    # x_0_1 = x_fid_0 + rot_w_to_cam @ rot_fid_to_cam @ x_cam_0
    # y_0_1 = y_fid_0 + rot_w_to_cam @ rot_fid_to_cam @ y_cam_0
    # z_0_1 = z_fid_0 + rot_w_to_cam @ rot_fid_to_cam @ z_cam_0

    x_0_1 = x_fid_0 + rot_w_to_cam @ x_cam_0
    y_0_1 = y_fid_0 + rot_w_to_cam @ y_cam_0
    z_0_1 = z_fid_0 + rot_w_to_cam @ z_cam_0


    x, y, z = roted_frame(roll, pitch, yaw, x_w, y_w, z_w)
    

    a = Arrow3D([0, x_cam_0[0]], [0, x_cam_0[1]], [0, x_cam_0[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([0, y_cam_0[0]], [0, y_cam_0[1]], [0, y_cam_0[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([0, z_cam_0[0]], [0, z_cam_0[1]], [0, z_cam_0[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)

    a = Arrow3D([x_fid_0[0], x_0_1[0]], [x_fid_0[1],x_0_1[1]], [x_fid_0[2], x_0_1[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([y_fid_0[0], y_0_1[0]], [y_fid_0[1], y_0_1[1]], [y_fid_0[2], y_0_1[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([z_fid_0[0], z_0_1[0]], [z_fid_0[1], z_0_1[1]], [z_fid_0[2], z_0_1[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)
    
    max_range = np.max(T0_1)
    ax1.set_xlim3d(-max_range, max_range)
    ax1.set_ylim3d(-max_range, max_range)
    ax1.set_zlim3d(-max_range, max_range)
    # Give them a name:

    ax1.text(0.0, 0.0, -0.1, r'$0$')
    ax1.text(x_cam_0[0], x_cam_0[1], x_cam_0[2], r'$x$')
    ax1.text(y_cam_0[0], y_cam_0[1], y_cam_0[2] , r'$y$')
    ax1.text(z_cam_0[0], z_cam_0[1], z_cam_0[2], r'$z$')

    plt.show()