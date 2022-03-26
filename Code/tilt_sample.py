from skimage import io
import numpy as np
import math
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from PIL import Image
import os
from roi_search import RoiSearch


class WalnutSample(object):
    """Rotational model of the sample.

      Attributes:
        axis: Vector coordinates of the rotation axis
        savepath: Storage location for files after sample rotation.
        filepath: Storage location of sample raw files.
        point: Coordinates of the center of gold dust.
        num: Number of images acquired after rotating the sample by 360 angles.
        distance: Sample horizontal drift distance.
        detector: the size of detector(pixel).
    """

    def __init__(self, axis, savepath, filepath, point=None, num=360, distance=0, detector=501):
        if point is None:
            point = [400, 81, 138]
        self.point = point
        self.axis = axis  # [x, y, z]
        self.num = num
        self.distance = distance
        self.filepath = filepath
        self.ref_left_tilt_theta = -math.atan(self.axis[0] / self.axis[2]) / math.pi * 180
        self.ref_front_tilt_theta = math.atan(self.axis[1] / self.axis[2]) / math.pi * 18
        self.left_tilt_theta = -math.atan(self.axis[0] / self.axis[2]) / math.pi * 180
        self.front_tilt_theta = math.atan(self.axis[1] / self.axis[2]) / math.pi * 18
        self.savepath = savepath
        self.detector = detector
        self.roi = RoiSearch()
        self.img3D = self.init_sample()

    def init_sample(self):
        img3D = np.zeros(shape=(self.detector, self.detector, self.detector), dtype='float')
        files = os.listdir(self.filepath)
        i = 0
        for file_ in files:
            image = Image.open(os.path.join(self.filepath, file_))
            img3D[i][:, self.distance:] = np.array(image)[:, self.distance:]
            i += 1

        # Simulated gold dust
        # for i in range(3):
        #     z = point[2] + i
        #     img3D[z][point[1]-1][point[0]-1] = 1000
        #     img3D[z][point[1]][point[0]-1] = 1000
        #     img3D[z][point[1]+1][point[0]-1] = 1000
        #     img3D[z][point[1]-1][point[0]] = 1000
        #     img3D[z][point[1]][point[0]] = 1000
        #     img3D[z][point[1]+1][point[0]] = 1000
        #     img3D[z][point[1]-1][point[0]+1] = 1000
        #     img3D[z][point[1]][point[0]+1] = 1000
        #     img3D[z][point[1]+1][point[0]+1] = 1000
        return img3D

    def tile_sample(self, rotate_img, left_tilt_theta, front_tilt_theta):
        """Sample rotation and preservation

          arg:
            rotate_img: Sample slice.
            left_tilt_theta: Tilt angle of rotation axis to the left.
            front_tilt_theta: Tilt angle of rotation axis to the front.

        """
        if left_tilt_theta != 0:
            for x_ in range(len(rotate_img)):
                rotate_img[:, x_, :] = np.array(Image.fromarray(rotate_img[:, x_, :]).rotate(left_tilt_theta))
        if front_tilt_theta != 0:
            for y_ in range(len(rotate_img)):
                rotate_img[:, :, y_] = np.array(Image.fromarray(rotate_img[:, :, y_]).rotate(front_tilt_theta))
        return rotate_img

    def adjustment_axis(self, x, y):
        """Adjust the rotation axis tilt angle by vector

          arg:
            x: Tilt vector of rotation axis to the left.
            y: Tilt vector of rotation axis to the front.

         """
        self.axis[0] += x
        self.axis[1] -= y
        self.left_tilt_theta = -math.atan(self.axis[0] / self.axis[2]) / math.pi * 180
        self.front_tilt_theta = math.atan(self.axis[1] / self.axis[2]) / math.pi * 18

    def rotate_360(self):
        theta = int(360 / self.num)
        rotate_img = self.img3D
        for i in range(self.num):
            rotate_theta = i * theta
            for z_ in range(len(rotate_img)):
                rotate_img[z_, :, :] = np.array(Image.fromarray(rotate_img[z_, :, :]).rotate(rotate_theta))
            tile_img = self.tile_sample(rotate_img, self.left_tilt_theta, self.front_tilt_theta)
            img = np.zeros(shape=(self.detector, self.detector), dtype='float')
            for x_ in range(len(tile_img)):
                img += tile_img[:, x_, :]
            io.imsave('%s/walnut_%s_%s_%s.png' %
                      (self.savepath, self.left_tilt_theta, self.front_tilt_theta, rotate_theta), img)
            self.roi.workflow_manual(image='%s/walnut_%s_%s_%s.png' %
                                           (self.savepath, self.left_tilt_theta, self.front_tilt_theta, rotate_theta),
                                     num=1,
                                     grey=10,
                                     margin=0)
            roi_info = self.roi.get_roi_list()
            print(roi_info)

    def rotate_mat(self, axis, radian):
        """Rotation matrix"""
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix

    def rotate_theta(self, theta):
        """Rotate to an angle"""
        print(1)
        rand_axis = self.axis
        theta = theta
        img3D_tilt = np.zeros(shape=(self.detector, self.detector, self.detector), dtype='float')
        yaw = -theta * math.pi / 180
        rot_matrix = self.rotate_mat(rand_axis, yaw)
        for z in range(self.detector):
            for y in range(self.detector):
                for x in range(self.detector):
                    point = [x - 250, y - 250, z - 250]
                    point_value = self.img3D[z][y][x]
                    point_rotate = np.dot(rot_matrix, point)
                    point_rotate_x = int(point_rotate[0] + self.detector/2)
                    point_rotate_y = int(point_rotate[1] + self.detector/2)
                    point_rotate_z = int(point_rotate[2] + self.detector/2)
                    # point_rotate = [point_rotate_x, point_rotate_z]
                    # rotate_points.append(point_rotate)
                    # rotate_points_x.append(point_rotate[0])
                    # rotate_points_z.append(point_rotate[1])
                    # theta_list.append(theta * i)
                    if 0 <= point_rotate_x < self.detector and 0 <= point_rotate_y < self.detector and 0 <= point_rotate_z < self.detector:
                        img3D_tilt[point_rotate_z, point_rotate_y, point_rotate_x] = point_value
        img = np.zeros(shape=(self.detector, self.detector), dtype='float')
        gold= np.dot(rot_matrix, self.point)
        gold_point_rotate_x = int(gold[0] + self.detector / 2)
        gold_point_rotate_y = int(gold[1] + self.detector / 2)
        gold_point_rotate_z = int(gold[2] + self.detector / 2)
        for x_ in range(len(img3D_tilt)):
            img += img3D_tilt[:, x_, :]
        print(img)
        img = img / 10
        img[gold_point_rotate_z, gold_point_rotate_y] = 1.0
        img[gold_point_rotate_z - 1, gold_point_rotate_y] = 1.0
        img[gold_point_rotate_z + 1, gold_point_rotate_y] = 1.0
        img[gold_point_rotate_z, gold_point_rotate_y - 1] = 1.0
        img[gold_point_rotate_z, gold_point_rotate_y + 1] = 1.0
        img[gold_point_rotate_z - 1, gold_point_rotate_y - 1] = 1.0
        img[gold_point_rotate_z - 1, gold_point_rotate_y + 1] = 1.0
        img[gold_point_rotate_z + 1, gold_point_rotate_y + 1] = 1.0
        img[gold_point_rotate_z + 1, gold_point_rotate_y - 1] = 1.0
        io.imsave('%s/walnut_%s_%s_%s.png' %
                  (self.savepath, self.left_tilt_theta, self.front_tilt_theta, theta), img)
        self.roi.workflow_manual(image='%s/walnut_%s_%s_%s.png' %
                                       (self.savepath, self.left_tilt_theta, self.front_tilt_theta, theta),
                                 num=1,
                                 grey=10,
                                 margin=0)
        roi_info = self.roi.get_roi_list()
        print(roi_info)
        # plt.figure()
        # plt.axis('equal')
        # plt.scatter(rotate_points_x, rotate_points_z)
        # f = plt.gcf()
        # f.savefig('%s/tilt_%s_%s_%s.png' % (self.savepath, self.axis[0], self.axis[1], self.axis[2]))
        # f.clear()
        # plt.show()
        # return rotate_points


class TiltSample(object):
    """Simulate the rotation of gold dust

      Attributes:
        axis: Vector coordinates of the rotation axis
        savepath: Storage location for files after sample rotation.
        point: Coordinates of the center of gold dust.
        num: Number of images acquired after rotating the sample by 360 angles.
    """
    def __init__(self, axis, savepath, point=None, num=360):
        if point is None:
            # point = [150, 138, 81]
            point = [0, 0, 3]
        self.point = point
        self.axis = axis  # [x, y, z]
        self.num = num
        self.ref_left_tilt_theta = -math.atan(self.axis[0] / self.axis[2]) / math.pi * 180
        self.ref_front_tilt_theta = math.atan(self.axis[1] / self.axis[2]) / math.pi * 18
        self.savepath = savepath

    def adjustment_axis(self, x, y):
        """Adjust the rotation axis tilt angle by vector

          arg:
            x: Tilt vector of rotation axis to the left.
            x: Tilt vector of rotation axis to the front.

        """
        self.axis[0] += x
        self.axis[1] -= y

    def adjustment_axis_theta(self, left_theta, front_theta):
        """Adjust the rotation axis tilt angle by angle

          arg:
            left_theta: Tilt angle of rotation axis to the left.
            front_theta: Tilt vectoangleion axis to the front.

        """
        left_tan = -math.tan(left_theta / 180 * math.pi)
        front_tan = math.tan(front_theta / 180 * math.pi)
        self.axis[0] += self.axis[0] * left_tan
        self.axis[1] -= self.axis[0] * front_tan

    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix

    def rotate_360(self):
        rand_axis = self.axis
        point = self.point
        n = self.num
        theta = int(360 / n)
        rotate_points = []
        rotate_points_x = []
        rotate_points_z = []
        theta_list = []
        for i in range(n):
            yaw = -theta * i * math.pi / 180
            rot_matrix = self.rotate_mat(rand_axis, yaw)
            point_rotate = np.dot(rot_matrix, point)
            point_rotate_x = point_rotate[0] + np.random.rand() * 0.1
            point_rotate_z = point_rotate[2] + np.random.rand() * 0.1
            point_rotate = [point_rotate_x, point_rotate_z]
            rotate_points.append(point_rotate)
            rotate_points_x.append(point_rotate[0])
            rotate_points_z.append(point_rotate[1])
            theta_list.append(theta * i)
        plt.figure()
        plt.grid()
        plt.axis('equal')
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.yticks(fontproperties='Times New Roman', size=28)
        plt.xticks(fontproperties='Times New Roman', size=28)
        plt.scatter(rotate_points_x, rotate_points_z)
        # plt.xlim((-6, 3))
        # plt.ylim((7, 12))
        # my_x_ticks = np.arange(-6, 3, 1)
        # my_y_ticks = np.arange(7, 12, 1)
        # plt.xticks(my_x_ticks)
        # plt.yticks(my_y_ticks)
        # plt.scatter(rotate_points_x, rotate_points_z)
        f = plt.gcf()
        f.savefig('%s/tilt_%s_%s_%s.png' % (self.savepath, self.axis[0], self.axis[1], self.axis[2]),
                  bbox_inches='tight')
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        f.savefig('%s/tilt_%s_%s_%s_axis.png' % (self.savepath, self.axis[0], self.axis[1], self.axis[2]),
                  bbox_inches='tight')
        f.clear()
        plt.show()
        return rotate_points


if __name__ == '__main__':
    import time

    walnut_sample = WalnutSample(axis=[-10, 10, 40], savepath='tiltGUI3', detector=501,
                                 filepath=r'C:\Users\zhaigj\Desktop\sampleAlignment\data', num=360)
    # walnut_sample.rotate_theta(0)
    for i in range(36):
        walnut_sample.rotate_theta(i * 10)