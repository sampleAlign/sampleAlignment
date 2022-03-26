
import math
import PIL
from PIL import Image
import numpy as np
import os
from skimage import io
import cv2
import time

class RotationSample(object):
    """Rotational model of the sample.

      Attributes:
        filepath: Storage location of sample raw files.
        distance: Sample horizontal drift distance.
        savepath: Storage location for files after sample rotation.
        detector: the size of detector(pixel).
    """

    def __init__(self, filepath, distance_x, distance_y=0, savepath='roughTest', detector=501):
        self.filepath = filepath
        self.savepath = savepath
        self.distance_init = distance_x
        self.distance = distance_x
        self.distance_y_init = distance_y
        self.distance_y = distance_y
        self.detector = detector
        self.time_consume = 0
        # self.img3D = self.init_sample()

    def init_sample(self):
        img3D = np.zeros(shape=(self.detector, self.detector, self.detector), dtype='float32')
        files = os.listdir(self.filepath)
        i = 0
        for file_ in files:
            image = Image.open(os.path.join(self.filepath, file_))
            img3D[i][:, self.distance:] = np.array(image)[:, self.distance:]
            i += 1
        return img3D

    def init_distance(self):
        """Sample position reset"""
        self.distance = self.distance_init
        self.distance_y = self.distance_y_init

    def move(self, length, length_y):
        """Move the position of the sample

          arg:
            length: Move to the right as a positive number.
        """
        self.distance += length
        self.distance_y += length_y
        flag = math.fabs(self.distance * 100) % 1
        flag_y = math.fabs(self.distance_y * 100) % 1
        if flag > 0.5:
            self.distance = (int(self.distance * 100) + 1) / 100
        else:
            self.distance = (int(self.distance * 100)) / 100
        if flag_y > 0.5:
            self.distance_y = (int(self.distance_y * 100) + 1) / 100
        else:
            self.distance_y = (int(self.distance_y * 100)) / 100
        print('Sample: ', self.distance, length, self.distance_y, length_y)

    def move_manual(self, old=[], new=[], conversion=None):
        """Manually move the position of the sample

          arg:
            old: Coordinates of the position of a point in the image before moving.
            new: Coordinates of the position of a point in the image after moving
            conversion: Conversion factor of image coordinates to real coordinates.
        """
        if conversion is None:
            conversion = []
        length_h = new[0] - old[0]

        # to do: conversion
        # length_h = conversion_fun(length_h)
        # length_v = conversion_fun(length_v)

        self.distance += length_h
        flag = math.fabs(self.distance * 100) % 1
        if flag > 0.5:
            self.distance = (int(self.distance * 100) + 1) / 100
        else:
            self.distance = int(self.distance * 100) / 100
        print('Sample (manual): ', self.distance, length_h)

    def rotate(self, theta):
        """Rotate the sample to an angle and save the image"""
        time_start = time.time()
        theta = int(theta % 360)
        flag = 1
        shift = self.distance * math.sin(theta / 180 * math.pi)
        shift_y = self.distance_y * math.cos(theta / 180 * math.pi)
        # ref_theta = 180 - math.atan(self.distance_y / self.distance) * 180 / math.pi
        # Handles sub-pixel level offsets
        shift = shift + shift_y
        if shift > 0:
            flag = 1
        else:
            flag = -1
        shift = abs(shift)
        shift_radix = shift % 1
        # shift_y_radix = shift_y % 1
        if shift_radix >= 0.5:
            shift = int(shift) + 1
        else:
            shift = int(shift)
        # if shift_y_radix >= 0.5:
        #     shift_y = int(shift_y) + 1
        # else:
        #     shift_y = int(shift_y)
        files = os.listdir(self.filepath)
        files.sort(key=lambda x: int(x[-8:-4]))
        count = 0
        img = np.zeros(shape=(self.detector, self.detector), dtype='float32')
        layer = np.zeros(shape=(self.detector,), dtype='float32')
        # shift += shift_y
        # if theta > 180:
        #     shift = -int(math.sqrt(shift*shift + shift_y*shift_y))
        # else:
        #     shift = int(math.sqrt(shift * shift + shift_y * shift_y))
        if not os.path.exists('%s/sample_%s_%s_%s.png' % (self.savepath, self.distance, self.distance_y, theta)):
            for file_ in files:
                image = cv2.imread(os.path.join(self.filepath, file_), -1)
                image = Image.fromarray(image)
                # image = Image.open(os.path.join(self.filepath, file_))
                rot = image.rotate(theta)
                im2d = np.array(rot)
                if flag > 0:
                    for i in range(0, self.detector - shift):
                        num = 0.0
                        for j in range(self.detector):
                            num += im2d[i][j]
                        layer[i + shift] = num
                else:
                    for i in range(shift, self.detector):
                        num = 0.0
                        for j in range(self.detector):
                            num += im2d[i][j]
                        layer[i - shift] = num
                img[count] = layer
                count += 1

            io.imsave('%s/sample_%s_%s_%s.png' % (self.savepath, self.distance, self.distance_y, theta), img)
        time_end = time.time()
        self.time_consume += (time_end - time_start)
        return img
        # Show the image
        # plt.imshow(img, plt.cm.gray)
        # plt.axis('off')
        # plt.show()


class RotationSampleMasked(object):
    """Rotational model of the sample with mask.

          Attributes:
            filepath: Storage location of sample raw files.
            distance: Sample horizontal drift distance.
            savepath: Storage location for files after sample rotation.
            detector: the size of detector(pixel).
        """

    def __init__(self, filepath, distance, savepath='roughTest', detector=501):
        self.filepath = filepath
        self.savepath = savepath
        self.distance_init = distance
        self.distance = distance
        self.detector = detector
        self.time_consume = 0
        # self.img3D = self.init_sample()

    def init_sample(self):
        img3D = np.zeros(shape=(self.detector, self.detector, self.detector), dtype='float16')
        files = os.listdir(self.filepath)
        i = 0
        for file_ in files:
            image = Image.open(os.path.join(self.filepath, file_))
            img3D[i][:, self.distance:] = np.array(image)[:, self.distance:]
            i += 1
        return img3D

    def init_distance(self):
        """Sample position reset"""
        self.distance = self.distance_init

    def move(self, length):
        """Move the position of the sample

          arg:
            length: Move to the right as a positive number.
        """
        self.distance += length
        flag = math.fabs(self.distance * 100) % 1
        if flag > 0.5:
            self.distance = (int(self.distance * 100) + 1) / 100
        else:
            self.distance = (int(self.distance * 100)) / 100
        print('Sample: ', self.distance, length)

    def move_manual(self, old=[], new=[], conversion=None):
        """Manually move the position of the sample

          arg:
            old: Coordinates of the position of a point in the image before moving.
            new: Coordinates of the position of a point in the image after moving
            conversion: Conversion factor of image coordinates to real coordinates.
        """
        if conversion is None:
            conversion = []
        length_h = new[0] - old[0]

        # to do: conversion
        # length_h = conversion_fun(length_h)
        # length_v = conversion_fun(length_v)

        self.distance += length_h
        flag = math.fabs(self.distance * 100) % 1
        if flag > 0.5:
            self.distance = (int(self.distance * 100) + 1) / 100
        else:
            self.distance = int(self.distance * 100) / 100
        print('Sample (manual): ', self.distance, length_h)

    def rotate(self, theta):
        """Rotate the sample to an angle and save the image"""
        time_start = time.time()
        theta = int(theta % 360)
        shift = abs(self.distance * math.sin(theta / 180 * math.pi))
        # Handles sub-pixel level offsets
        shift_radix = shift % 1
        if shift_radix >= 0.5:
            shift = int(shift) + 1
        else:
            shift = int(shift)

        files = os.listdir(self.filepath)
        files.sort(key=lambda x: int(x[-9:-5]))
        count = 0
        img = np.zeros(shape=(self.detector, self.detector), dtype='float')
        layer = np.zeros(shape=(self.detector,), dtype='float')
        if not os.path.exists('%s/sample_%s_%s.png' % (self.savepath, self.distance, theta)):
            for file_ in files:
                image = Image.open(os.path.join(self.filepath, file_))
                rot = image.rotate(theta)
                im2d = np.array(rot)
                if theta <= 180:
                    for i in range(0, self.detector - shift):
                        num = 0.0
                        for j in range(self.detector):
                            num += im2d[i][j]
                        layer[i + shift] = num
                else:
                    for i in range(shift, self.detector):
                        num = 0.0
                        for j in range(self.detector):
                            num += im2d[i][j]
                        layer[i - shift] = num
                img[count] = layer
                count += 1
            io.imsave('%s/sample_%s_%s.png' % (self.savepath, self.distance, theta), img)
        time_end = time.time()
        self.time_consume += (time_end - time_start)
        return img
        # Show the image
        # plt.imshow(img, plt.cm.gray)
        # plt.axis('off')
        # plt.show()


if __name__ == '__main__':
    import time
    distance_x = 33.0
    distance_y = -76.0
    time_start = time.time()
    print('Start: ', time_start)
    # sample = RotationSample(distance=distance,  detector=513, savepath='MaskSample',
                            # filepath=r'D:\XRF_rec\8348_filter')
    sample = RotationSample(distance_x=distance_x, distance_y=distance_y, detector=500, savepath='align3',
                            filepath=r'D:\XRF_rec\8348')
    sample.rotate(30)
    # sample.rotate(270)
    # sample.rotate(180)
    # sample.rotate(0)
    # sample.rotate(270)
    # sample.rotate(96)
    # sample.rotate(186)
    # sample.rotate(276)
    # for i in range(0, 180):
        # sample.move(-1)
        # sample.rotate(i*2)
    time_end = time.time()
    print('All time: ', time_end - time_start)