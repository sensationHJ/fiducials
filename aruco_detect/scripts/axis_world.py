import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
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


if __name__ == '__main__':
    center = [0, 0, 0]
    length = 1
    width = 1
    height = 1
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    # X, Y, Z = cuboid_data(center, (length, width, height))
    # ax1.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=0.1)
    ax1.set_xlabel('X')
    ax1.set_xlim(-1, 1)
    ax1.set_ylabel('Y')
    ax1.set_ylim(-1, 1)
    ax1.set_zlabel('Z')
    ax1.set_zlim(-1, 1)

    x_w = np.array([1,0,0])
    y_w = np.array([0,1,0])
    z_w = np.array([0,0,1])

    rot_cam_frame = Rz(np.deg2rad(-90)) @ Rx(np.deg2rad(-90))
    print(np.linalg.inv(rot_cam_frame))
    x_cam = rot_cam_frame @ x_w
    y_cam = rot_cam_frame @ y_w
    z_cam = rot_cam_frame @ z_w


    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)

    a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, 1], [0, 0], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, 1], **arrow_prop_dict, color='b')
    ax1.add_artist(a)

    offset = 2
    # camera frame
    a = Arrow3D([0+ offset, x_cam[0]+ offset], [0, x_cam[1]], [0, x_cam[2]], **arrow_prop_dict, color='r')
    ax1.add_artist(a)
    a = Arrow3D([0+ offset, y_cam[0]+ offset], [0, y_cam[1]], [0, y_cam[2]], **arrow_prop_dict, color='g')
    ax1.add_artist(a)
    a = Arrow3D([0+ offset, z_cam[0]+ offset], [0, z_cam[1]], [0, z_cam[2]], **arrow_prop_dict, color='b')
    ax1.add_artist(a)
    
    # a = Arrow3D([0, x_cam[0]], [0, x_cam[1]], [0, x_cam[2]], **arrow_prop_dict, color='r')
    # ax1.add_artist(a)
    # a = Arrow3D([0, y_cam[0]], [0, y_cam[1]], [0, y_cam[2]], **arrow_prop_dict, color='g')
    # ax1.add_artist(a)
    # a = Arrow3D([0, z_cam[0]], [0, z_cam[1]], [0, z_cam[2]], **arrow_prop_dict, color='b')
    # ax1.add_artist(a)

    # Give them a name:
    ax1.text(0.0, 0.0, -0.1, r'$0$')
    ax1.text(1.1, 0, 0, r'$x$')
    ax1.text(0, 1.1, 0, r'$y$')
    ax1.text(0, 0, 1.1, r'$z$')

    max_range = 3
    ax1.set_xlim3d(-max_range, max_range)
    ax1.set_ylim3d(-max_range, max_range)
    ax1.set_zlim3d(-max_range, max_range)
    plt.show()