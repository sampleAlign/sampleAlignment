import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class RoiSearch(object):
    """Find the ROIs of the light spots.

      Attributes:
        roi_num: The number of ROIs to find.
        roi_flag: ROI confirmation flag.
        roi_list: List of unmatched ROIs.
    """

    def __init__(self):
        self.roi_num = 15
        self.roi_flag = 0
        self.roi_list = []
        self.contours_list = []
        self.center_list = []

    def clear(self):
        self.roi_flag = 0
        self.roi_list = []
        self.contours_list = []
        self.center_list = []

    def set_roi_flag(self, roi_flag):
        self.roi_flag = roi_flag

    def get_roi_flag(self):
        return self.roi_flag

    def set_roi_list(self, roi_list):
        """The API of the ROIs determination/modification command."""
        self.roi_list = roi_list
        self.roi_flag = 1

    def get_roi_list(self):
        return self.roi_list

    def get_contours_list(self):
        return self.contours_list

    def get_center_list(self):
        return self.center_list

    def filter_rois(self, roi_list, num=15):
        """Filter out eligible ROIs"""
        dtype = [('area', int),
                 ('left_top_x', int), ('left_top_y', int),
                 ('right_top_x', int), ('right_top_y', int),
                 ('left_bottom_x', int), ('left_bottom_y', int),
                 ('right_bottom_x', float), ('right_bottom_y', float),
                 ('center_x', float), ('center_y', float)]
        roi_list_sort = np.sort(np.array(roi_list, dtype=dtype), order='area')[::-1]
        if len(roi_list_sort) < num:
            return roi_list_sort
        # filter
        # if condition:
        #    delete some ROIs
        if len(roi_list_sort) > num:
            roi_list_sort = roi_list_sort[0:num]
        return roi_list_sort

    def grey_center(self, img):
        grey_list = img
        grey_sum = grey_list.sum()
        vector = np.arange(len(grey_list))
        y = np.dot(vector, grey_list).sum()
        x = np.dot(vector, grey_list.T).sum()
        return x/grey_sum, y/grey_sum

    def noise(self, img):
        """To do: remove image noise"""
        h = 10
        # templateWindowSize用于计算权重的模板补丁的像素大小，为奇数，默认7
        templateWindowSize = 3
        # searchWindowSize窗口的像素大小，用于计算给定像素的加权平均值，为奇数，默认21
        searchWindowSize = 21
        # dst = cv2.medianBlur(img, 5)
        return img

    def workflow(self, image, mask=[], num=15, margin=10):
        """Automatic workflow for finding spot ROIs

         Args:
            image: The URL of spots image.
            num: The number of ROIs to find.
            margin: The information variation range of light spot.
            mask: mask area:[x_lefttop, y_lefttop, width, height]
        """
        a = time.time()
        print(image)
        raw = cv2.imread(image, 0)
        # image_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        image_gray = raw
        image_gray = self.noise(image_gray)
        b = time.time()
        print(b - a)
        if mask:
            for area in mask:
                maks_area = np.zeros(shape=(area[2], area[3]), dtype="uint8")
                image_gray[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = maks_area
                # maks_area = np.zeros(shape=(image_gray.shape[0], image_gray.shape[1]), dtype="uint8")
                # maks_area[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = 255
                # image_gray = cv2.add(image_gray, np.zeros(np.shape(image_gray), dtype=np.uint8), mask=maks_area)
        ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilate = cv2.dilate(thresh, kernel_dil)
        kernel_ero = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        erode = cv2.erode(dilate, kernel_ero)
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_info = []
        for i in range(len(contours)):
            cnt = contours[i]
            # M = cv2.moments(cnt)
            # if M["m00"] == 0:
            #     center_x = 0.0
            #     center_y = 0.0
            # else:
            #     center_x = M["m10"] / M["m00"]
            #     center_y = M["m01"] / M["m00"]
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w / 2
            center_y = y + h / 2
            contours_info.append((w * h, x - margin, y - margin, x + w + margin, y - margin,
                                  x - margin, y + h + margin, x + w + margin, y + h + margin,
                                  center_x, center_y))
        if contours_info:
            self.roi_list = self.filter_rois(contours_info, num)
            self.contours_list = np.array(self.roi_list.tolist())[:, 1:9]
            self.center_list = np.array(self.roi_list.tolist())[:, 9:11]
            x0, y0 = self.roi_list[0][9], self.roi_list[0][10]
            cv2.drawContours(raw, contours, -1, (0, 255, 0), 3)
            cv2.rectangle(raw, (self.roi_list[0][1], self.roi_list[0][2]), (self.roi_list[0][3], self.roi_list[0][6]),
                          (255, 0, 0), 2)
            cv2.circle(raw, (int(x0), int(y0)), 4, (0, 255, 0), 4)
            print(x0, y0)
            plt.imshow(raw, plt.cm.gray)
            plt.axis('off')
            plt.show()
        else:
            self.roi_list = []
            self.contours_list = []
            self.center_list = []
        c = time.time()
        print(c - b)

    def workflow_th(self, image, mask=[], th=0.1, num=15, margin=10):
        """Automatic workflow for finding spot ROIs by manually setting the gray threshold

         Args:
            image: The URL of spots image.
            th: grey threshold.
            num: The number of ROIs to find.
            margin: BGR information variation range of light spot.
            mask: mask area list.
        """
        a = time.time()
        print(image)
        raw = cv2.imread(image, 0)
        # image_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        image_gray = raw
        image_gray = self.noise(image_gray)
        b = time.time()
        print(b - a)
        if mask:
            for area in mask:
                maks_area = np.zeros(shape=(area[2], area[3]))
                image_gray[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = maks_area
                # maks_area = np.zeros(shape=(image_gray.shape[0], image_gray.shape[1]), dtype="uint8")
                # maks_area[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = 255
                # image_gray = cv2.add(image_gray, np.zeros(np.shape(image_gray), dtype=np.uint8), mask=maks_area)
        length = len(image_gray)
        grey = (np.max(image_gray) - np.nanmean(image_gray)) * th + np.nanmean(image_gray)
        ret, thresh = cv2.threshold(image_gray, grey, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilate = cv2.dilate(thresh, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        erode = cv2.erode(dilate, kernel)
        # erode = dilate
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_info = []
        cnt_index = 0
        for i in range(len(contours)):
            cnt = contours[i]
            # M = cv2.moments(cnt)
            # if M["m00"] == 0:
            #     center_x = 0
            #     center_y = 0
            # else:
            #     center_x = M["m10"] / M["m00"]
            #     center_y = M["m01"] / M["m00"]
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w / 2
            center_y = y + h / 2
            contours_info.append((w * h, x - margin, margin, x + w + margin, margin,
                                  x - margin, length - margin, x + w + margin, length - margin,
                                  center_x, center_y))
        if contours_info:
            self.roi_list = self.filter_rois(contours_info, num)
            self.contours_list = np.array(self.roi_list.tolist())[:, 1:9]
            self.center_list = np.array(self.roi_list.tolist())[:, 9:11]
            x0, y0 = self.roi_list[0][9], self.roi_list[0][10]
            # cv2.drawContours(raw, contours, -1, (0, 255, 0), 3)
            # cv2.rectangle(raw, (self.roi_list[0][1], self.roi_list[0][2]), (self.roi_list[0][3], self.roi_list[0][6]),
            #               (255, 0, 0), 2)
            # cv2.circle(raw, (int(x0), int(y0)), 4, (0, 255, 0), 4)
            print(x0, y0)
            # plt.imshow(raw, plt.cm.gray)
            # plt.axis('off')
            # plt.show()
        else:
            self.roi_list = []
            self.contours_list = []
            self.center_list = []

    def workflow_manual(self, image, mask=[], grey=None, num=15, margin=10):
        """Automatic workflow for finding spot ROIs by manually setting the gray value

         Args:
            image: The URL of spots image.
            grey: grey information of light spot.
            num: The number of ROIs to find.
            margin: BGR information variation range of light spot.
            mask: mask area.
        """
        if grey is None:
            grey = 10
        raw = cv2.imread(image)
        image_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if mask:
            for area in mask:
                maks_area = np.zeros(shape=(area[2], area[3]))
                image_gray[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = maks_area
                # maks_area = np.zeros(shape=(image_gray.shape[0], image_gray.shape[1]), dtype="uint8")
                # maks_area[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = 255
                # image_gray = cv2.add(image_gray, np.zeros(np.shape(image_gray), dtype=np.uint8), mask=maks_area)
        ret, thresh = cv2.threshold(image_gray, grey, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_info = []
        for i in range(len(contours)):
            cnt = contours[i]
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                center_x = 0
                center_y = 0
            else:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)
            contours_info.append((w * h, x - margin, y - margin, x + w + margin, y - margin,
                                  x - margin, y + h + margin, x + w + margin, y + h + margin,
                                  center_x, center_y))
            cv2.rectangle(raw, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 0, 255), 2)
        if contours_info:
            self.roi_list = self.filter_rois(contours_info, num)
            self.contours_list = np.array(self.roi_list.tolist())[:, 1:9]
            self.center_list = np.array(self.roi_list.tolist())[:, 9:11]
            cv2.drawContours(raw, contours, -1, (0, 255, 0), 3)
            cv2.rectangle(raw, (self.roi_list[0][1], self.roi_list[0][2]), (self.roi_list[0][3], self.roi_list[0][6]),
                          (255, 0, 0), 2)
            plt.imshow(raw, plt.cm.gray)
            plt.axis('off')
            plt.show()
        else:
            self.roi_list = []
            self.contours_list = []
            self.center_list = []


    def workflow_CoM(self, image, mask=[], num=15, margin=10):
        """Automatic workflow for finding spot ROIs

         Args:
            image: The URL of spots image.
            num: The number of ROIs to find.
            margin: BGR information variation range of light spot.
            mask: mask area.
        """
        a = time.time()
        raw = cv2.imread(image)
        image_gray = raw
        # image_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        image_gray = self.noise(image_gray)
        b = time.time()
        print(b - a)
        if mask:
            for area in mask:
                maks_area = np.zeros(shape=(area[2], area[3]))
                image_gray[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = maks_area
                # maks_area = np.zeros(shape=(image_gray.shape[0], image_gray.shape[1]), dtype="uint8")
                # maks_area[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = 255
                # image_gray = cv2.add(image_gray, np.zeros(np.shape(image_gray), dtype=np.uint8), mask=maks_area)
        ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        dilate = cv2.dilate(thresh, kernel_dil)
        kernel_ero = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        erode = cv2.erode(dilate, kernel_ero)
        contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_info = []
        for i in range(len(contours)):
            cnt = contours[i]
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                center_x = 0.0
                center_y = 0.0
            else:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
            x, y, w, h = cv2.boundingRect(cnt)
            contours_info.append((w * h, x - margin, y - margin, x + w + margin, y - margin,
                                  x - margin, y + h + margin, x + w + margin, y + h + margin,
                                  center_x, center_y))
        if contours_info:
            self.roi_list = self.filter_rois(contours_info, num)
            cnt = np.array([[[self.roi_list[0][1], self.roi_list[0][2]],
                             [self.roi_list[0][3], self.roi_list[0][4]],
                             [self.roi_list[0][7], self.roi_list[0][8]],
                             [self.roi_list[0][5], self.roi_list[0][6]]]], dtype=np.int32)
            cnt_mask = np.zeros(image_gray.shape[:2], dtype="uint8")
            # cv2.polylines(cnt_mask, cnt, 1, 255)
            cnt_mask = cv2.fillPoly(cnt_mask, cnt, 255)
            self.contours_list = np.array(self.roi_list.tolist())[:, 1:9]
            self.center_list = np.array(self.roi_list.tolist())[:, 9:11]
            res = cv2.cvtColor(cv2.bitwise_and(raw, raw, mask=erode), cv2.COLOR_BGR2GRAY)
            res = cv2.bitwise_and(res, res, mask=cnt_mask)
            img_cal = res
            x0, y0 = self.grey_center(img_cal)
            print(x0, y0)
            self.roi_list[0][-2] = x0
            self.roi_list[0][-1] = y0
            self.center_list[0][-2] = x0
            self.center_list[0][-1] = y0
            cv2.drawContours(raw, contours, -1, (0, 255, 0), 3)
            cv2.rectangle(raw, (self.roi_list[0][1], self.roi_list[0][2]), (self.roi_list[0][3], self.roi_list[0][6]),
                          (255, 0, 0), 2)
            cv2.circle(raw, (int(x0), int(y0)), 4, (0, 255, 0), 4)
            plt.imshow(raw, plt.cm.gray)
            plt.axis('off')
            plt.show()
        else:
            self.roi_list = []
            self.contours_list = []
            self.center_list = []
        c = time.time()
        print(c - b)

    def workflow_th_CoM(self, image, mask=[], th=0.2, num=15, margin=10):
        """Automatic workflow for finding spot ROIs by manually setting the gray threshold

         Args:
            image: The URL of spots image.
            th: grey threshold.
            num: The number of ROIs to find.
            margin: BGR information variation range of light spot.
            mask: mask area list.
        """
        raw = cv2.imread(image)
        image_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        image_gray = self.noise(image_gray)
        if mask:
            for area in mask:
                maks_area = np.zeros(shape=(area[2], area[3]))
                image_gray[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = maks_area
                # maks_area = np.zeros(shape=(image_gray.shape[0], image_gray.shape[1]), dtype="uint8")
                # maks_area[area[0]:area[0] + area[2], area[1]:area[1] + area[3]] = 255
                # image_gray = cv2.add(image_gray, np.zeros(np.shape(image_gray), dtype=np.uint8), mask=maks_area)
        length = len(image_gray)
        grey = (np.max(image_gray) - np.nanmean(image_gray)) * th + np.nanmean(image_gray)
        ret, thresh = cv2.threshold(image_gray, grey, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilate = cv2.dilate(thresh, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        erode = cv2.erode(dilate, kernel)
        # erode = dilate
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_info = []
        cnt_index = 0
        for i in range(len(contours)):
            cnt = contours[i]
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                center_x = 0
                center_y = 0
            else:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
            x, y, w, h = cv2.boundingRect(cnt)
            contours_info.append((w * h, x - margin, margin, x + w + margin, margin,
                                  x - margin, length - margin, x + w + margin, length - margin,
                                  center_x, center_y))
            if w * h > 1000:
                cnt_index = i
        if contours_info:
            self.roi_list = self.filter_rois(contours_info, num)
            self.contours_list = np.array(self.roi_list.tolist())[:, 1:9]
            self.center_list = np.array(self.roi_list.tolist())[:, 9:11]
            cv2.drawContours(image=raw, contours=contours, contourIdx=cnt_index, color=(0, 255, 0), thickness=1)
            res = cv2.cvtColor(cv2.bitwise_and(raw, raw, mask=erode), cv2.COLOR_BGR2GRAY)
            img_cal = res
            x0, y0 = self.grey_center(img_cal)
            print(x0 - 250, y0 - 250)
            self.roi_list[0][-2] = x0
            self.roi_list[0][-1] = y0
            self.center_list[0][-2] = x0
            self.center_list[0][-1] = y0
            print(self.roi_list)
            print(self.center_list)
            # cv2.rectangle(raw, (self.roi_list[0][1], self.roi_list[0][2]), (self.roi_list[0][3], self.roi_list[0][6]),
            #               (255, 0, 0), 2)
            # cv2.circle(raw, (int(x0), int(y0)), 4, (0, 255, 0), 4)
            plt.imshow(res, plt.cm.gray)
            plt.axis('off')
            plt.show()
        else:
            self.roi_list = []
            self.contours_list = []
            self.center_list = []


if __name__ == "__main__":
    roi = RoiSearch()
    # roi.workflow(image=r'C:\Users\zhaigj\Desktop\mamba\mamba\mamba\plot\plots.png', num=5)
    # for th in range(3):
    #    roi.workflow_th(image=r'rotate\walnut_0_90_0.png', th=th/100, num=1, margin=0)
    #    c_90 = roi.get_roi_list()[0][9]
    #    roi.workflow_th(image=r'rotate\walnut_0_270_0.png', th=th/100, num=1, margin=0)
    #    c_270 = roi.get_roi_list()[0][9]
    #    print('Number: ', th)
    #    print('C_90: ', c_90)
    #    print('C_270: ', c_270)
    #    print('Cal: ', c_90 - c_270)
    # roi.workflow(image=r'rotate_58V2/sample_0_0.png', mask=[], num=1, margin=0)
    # plt.imshow(image)
    # plt.show()
    roi.workflow_th(image=r'C:\Users\zhaigj\Desktop\sampleAlignment\script\align3\sample_0_0_0.png',
                     mask=[], num=1, margin=0)
    list = roi.get_roi_list()
    print((list[0][9]))
    # roi.workflow(image=r'XRF_test/sample_0_90.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'XRF_test/sample_0_180.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'XRF_test/sample_0_270.png', mask=[], num=1, margin=0)

    # roi.workflow(image=r'rotate_xrf/sample_-0.01_90.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'rotate_xrf/sample_-0.01_270.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'rotate_xrf/sample_1.97_90.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'rotate_xrf/sample_1.97_270.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'rotate_58V2/sample_0_180.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'rotate_58V2/sample_0_270.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'rotate_58V2/sample_0_270.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'XRF/8348/crop_Corrected_rep01_04321_NMC_622_Charged_10C_RT_-089.00_Degree_08348.00_eV_001of001.xrm.bim.bim.bim.jpg', mask=[], num=1, margin=0)
    # roi.workflow(image=r'300/walnut_52_0.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'300/walnut_52_90.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'300/walnut_52_180.png', mask=[], num=1, margin=0)
    # roi.workflow(image=r'300/walnut_52_270.png', mask=[], num=1, margin=0)
    # print(np.array(roi.get_roi_list().tolist())[:, 1:9])
    # print(np.array(roi.get_roi_list().tolist())[:, 9:11])

    # roi.workflow(image=r'rotate_58/sample_0_96.png', mask=[], num=1, margin=0)
    # # print(np.array(roi.get_roi_list().tolist())[:, 1:9])
    # # print(np.array(roi.get_roi_list().tolist())[:, 9:11])
    #
    # roi.workflow(image=r'rotate_58/sample_0_186.png', mask=[], num=1, margin=0)
    # # print(np.array(roi.get_roi_list().tolist())[:, 1:9])
    # # print(np.array(roi.get_roi_list().tolist())[:, 9:11])
    #
    # roi.workflow(image=r'rotate_58/sample_0_276.png', mask=[], num=1, margin=0)
    # print(np.array(roi.get_roi_list().tolist())[:, 1:9])
    # print(np.array(roi.get_roi_list().tolist())[:, 9:11])
    # roi.workflow(image=r'tomo_00058_rec_prej/sample_0_96.png', mask=[], num=1, margin=0)
    # print(np.array(roi.get_roi_list().tolist())[:, 1:9])
    # print(np.array(roi.get_roi_list().tolist())[:, 9:11])
    # roi.workflow(image=r'tomo_00058_rec_prej/sample_0_276.png', mask=[], num=1, margin=0)
    # print(np.array(roi.get_roi_list().tolist())[:, 1:9])
    # print(np.array(roi.get_roi_list().tolist())[:, 9:11])
