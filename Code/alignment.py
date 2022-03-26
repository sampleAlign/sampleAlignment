import math
from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
from roi_search import RoiSearch
import time
from roate_sample import RotationSample


def init_sample(distance_x=200, distance_y=200, detector=501, savepath='roughTest',
                filepath=r'C:\Users\zhaigj\Desktop\sampleAlignment\data'):
    """Rotating sample initialization

      Args:
        distance: Sample horizontal drift distance.
        detector: the size of detector.
        savepath: Storage location for files after sample rotation.
        filepath: Storage location of sample raw files.
    """

    sample = RotationSample(filepath=filepath, distance_x=distance_x, distance_y=distance_y, savepath=savepath,
                            detector=detector)
    return sample


def rotate_sample(sample, theta):
    """Rotating sample initialization

      Args:
        sample: Rotating sample.
        theta: Rotation angle of the sample.
    """
    rotate_start_time = time.time()
    sample.rotate(theta)
    rotate_end_time = time.time()
    print('Rotate time: ', rotate_end_time - rotate_start_time)


def rotate_sample_360(sample, num):
    """Rotating the sample by 360 angles

      Args:
        sample: Rotating sample.
        num: Number of images acquired after rotating the sample by 360 angles.
    """
    rotate_start_time = time.time()
    theta = int(360 / num)
    for i in range(num):
        sample.rotate(i * theta)
    sample.rotate(0)
    sample.rotate(90)
    sample.rotate(180)
    sample.rotate(270)
    rotate_end_time = time.time()
    print('Rotate 360 time: ', rotate_end_time - rotate_start_time)


class RoughAlignment(object):
    """Automated coarse alignment algorithm

      Attributes:
        sample: Rotating sample.
        size:  the size of detector(pixel).
        distance: Sample horizontal drift distance.
        savepath: Storage location for files after sample rotation.
        move_distance: Optimal movement distance.
        roi: roi filter module.
    """

    def __init__(self, sample, mask=[]):
        self.sample = sample
        self.size = self.sample.detector
        self.savepath = self.sample.savepath
        self.distance = self.sample.distance
        self.distance_y = self.sample.distance_y
        self.mask = mask
        self.move_distance = 0
        self.move_distance_y = 0
        self.roi = RoiSearch()

    def cal_distance(self, num):
        """Calculate the offset distance of the sample"""
        time_rot = time.time()
        rotate_sample_360(self.sample, num)
        time_rot_end = time.time()
        time_con = time_rot_end - time_rot
        with open('log.txt', 'a') as f:
            f.write("%s: %s\n" % ("Coarse Rotate Consume: ", time_con))
        mis_flag = 0
        mis_x = 0
        mis_y = 0
        for i in range(num):
            self.roi.workflow_th(
                image=r'%s\sample_%s_%s_%s.png' % (self.savepath, self.distance, self.distance_y, i * int(360 / num)),
                mask=self.mask, num=1, margin=0)
            roi_info = self.roi.get_roi_list()
            if not roi_info:
                mis_flag = 1
            else:
                roi_left_top = [roi_info[0][1], roi_info[0][2]]
                roi_right_top = [roi_info[0][3], roi_info[0][4]]
                if roi_right_top[0] == self.size or roi_left_top[0] == 0:
                    mis_flag = 1
                    # if i * 360 / num == 0 or i * 360 / num == 180:
                    #     mis_y = 1
                    # if i * 360 / num == 90 or i * 360 / num == 270:
                    #     mis_x = 1

        # roi_info_y = self.roi.workflow_th(image=r'%s/sample_%s_%s_%s.png' % (self.savepath, self.distance, self.distance_y, 0),
        #                   mask=self.mask, num=1, margin=0)
        if True:
            # roi_left_top = [roi_info[0][1], roi_info[0][2]]
            # roi_right_top = [roi_info[0][3], roi_info[0][4]]
            # ref_width = roi_right_top[0] - roi_left_top[0]
            #
            # # Other useful parameters
            # roi_area = roi_info[0][0]
            # roi_right_bottom = [roi_info[0][5], roi_info[0][6]]
            # roi_left_bottom = [roi_info[0][7], roi_info[0][8]]

            effect_theta = []

            if mis_flag == 1:
                search_theta = []
                key_theta = 0
                search_mis_flag = 0
                print(num)
                for i in range(num):
                    if 0 < i * int(360 / num) < 180:
                        search_theta.append(i * int(360 / num))
                for cur_theta in search_theta:
                    self.roi.workflow_th(image=r'%s\sample_%s_%s_%s.png' %
                                            (self.savepath, self.distance, self.distance_y, cur_theta),
                                      mask=self.mask, num=1, margin=0)
                    cur_roi_info = self.roi.get_roi_list()
                    if abs(cur_roi_info[0][3] / 2 + cur_roi_info[0][1] / 2 - self.size / 2) < 10:
                        key_theta = cur_theta
                    if cur_roi_info[0][3] == self.size:
                        search_mis_flag = 1
                    if search_mis_flag == 1 and cur_roi_info[0][3] < self.size:
                        effect_theta.append({
                            'cur_theta': cur_theta,
                            'cur_roi_info': cur_roi_info
                        })
                #
                # if mis_y == 1 and mis_x == 0:
                #     search_theta = []
                #     search_mis_flag = 0
                #     for i in range(num):
                #         if 0 < i * 360 / num < 90:
                #             search_theta.append(int(i * 360 / num))
                #     for cur_theta in search_theta:
                #         self.roi.workflow_th(image=r'%s/sample_%s_%s_%s.png' %
                #                                 (self.savepath, self.distance, self.distance_y, cur_theta),
                #                           mask=self.mask, num=1, margin=0)
                #         cur_roi_info = self.roi.get_roi_list()
                #         if cur_roi_info[0][3] == self.size:
                #             search_mis_flag = 1
                #         if search_mis_flag == 1 and cur_roi_info[0][3] < self.size:
                #             effect_theta.append({
                #                 'cur_theta': cur_theta,
                #                 'cur_roi_info': cur_roi_info
                #             })
                # if mis_y == 0 and mis_x == 1:
                #     search_theta = []
                #     search_mis_flag = 0
                #     for i in range(num):
                #         if 90 < i * 360 / num < 180:
                #             search_theta.append(int(i * 360 / num))
                #     for cur_theta in search_theta:
                #         self.roi.workflow_th(image=r'%s/sample_%s_%s_%s.png' %
                #                                 (self.savepath, self.distance, self.distance_y, cur_theta),
                #                           mask=self.mask, num=1, margin=0)
                #         cur_roi_info = self.roi.get_roi_list()
                #         if cur_roi_info[0][3] == self.size:
                #             search_mis_flag = 1
                #         if search_mis_flag == 1 and cur_roi_info[0][3] < self.size:
                #             effect_theta.append({
                #                 'cur_theta': cur_theta,
                #                 'cur_roi_info': cur_roi_info
                #             })
                # if mis_y == 1 and mis_x == 1:
                #     pass
                # else:
                #     pass
                # search_theta = []
                # search_mis_flag = 0
                # for i in range(num):
                #     if 90 < i * 360 / num < 180:
                #         search_theta.append(int(i * 360 / num))
                # for cur_theta in search_theta:
                #     self.roi.workflow_th(image=r'%s/sample_%s_%s_%s.png' %
                #                             (self.savepath, self.distance, self.distance_y, cur_theta),
                #                       mask=self.mask, num=1, margin=0)
                #     cur_roi_info = self.roi.get_roi_list()
                #     if cur_roi_info[0][3] == self.size:
                #         search_mis_flag = 1
                #     if search_mis_flag == 1 and cur_roi_info[0][3] < self.size:
                #         effect_theta.append({
                #             'cur_theta': cur_theta,
                #             'cur_roi_info': cur_roi_info
                #         })
                if effect_theta == [] or key_theta == 0:
                    return -2000, -2000
            else:
                print('ALL IN DETECTOR!')
            print('AFFECT THETA: ', effect_theta)
            if effect_theta:
                print(key_theta)
                cal_theta = effect_theta[0]['cur_theta']
                cal_roi_info = effect_theta[0]['cur_roi_info']
                cal_dis = int((cal_roi_info[0][3] + cal_roi_info[0][1]) / 2) - self.size / 2
                print(cal_dis)
                move_distance_x = int(cal_dis / (
                        math.sin(cal_theta / 180 * math.pi) - math.cos(cal_theta / 180 * math.pi) * math.tan(
                    key_theta / 180 * math.pi)))
                move_distance_y = -int(move_distance_x * math.tan(key_theta / 180 * math.pi))
                # roi_width = cal_roi_info[0][3] - cal_roi_info[0][1]
                # move_distance_min = roi_width/2 -
                # cal_theta = cal_theta - 90
                # cal_theta = cal_theta * math.pi / 180
                # move_distance_max = int(roi_width / math.cos(cal_theta)) - ref_width
                return move_distance_y, move_distance_x
            else:
                print('NO AFFECT THETA')
                return -3000, -3000
        else:
            return int(self.size / 2), int(self.size / 2)

    def move_sample(self, move_distance_x, move_distance_y):
        self.sample.move(move_distance_x, move_distance_y)

    def get_move_distance(self):
        return self.move_distance

    def rough_alignment(self, number=12):
        """Automated coarse alignment workflow for samples"""
        num = number
        while True:
            move_distance_y, move_distance_x = self.cal_distance(num=num)
            print('Y distance：', move_distance_y)
            print('X distance：', move_distance_x)
            if abs(move_distance_y) < 2000 and abs(move_distance_x) < 2000:
                # if move_distance_max <= int(self.size / 2):
                #     self.move_distance += -move_distance_max
                #     self.move_sample(-move_distance_max)
                #     self.distance = self.sample.distance
                #     # distance = distance - move_distance_max
                # else:
                self.move_sample(-move_distance_x, -move_distance_y)
                self.distance -= float(move_distance_x)
                self.distance_y -= float(move_distance_y)
                # print('CC', self.distance_y)
                # distance = distance - move_distance_min
            elif move_distance_y == -2000 and move_distance_x == -2000:
                num = num * 2
            else:
                # t = time.time()
                # with open('log.txt', 'a') as f:
                #     f.write("%s: %s\n" % ("Coarse Rotate Consume: ", t))
                # Coarse alignment final step: move the sample to the center of rotation
                print('T')
                self.roi.workflow_th(image=r'%s\sample_%s_%s_%s.png' % (self.savepath, self.distance, self.distance_y, 90),
                                  mask=self.mask, num=1, margin=0)
                # print(roi.get_roi_list())
                roi_info = self.roi.get_roi_list()
                # roi_left_top = [roi_info[0][1], roi_info[0][2]]
                # roi_right_top = [roi_info[0][3], roi_info[0][4]]
                center_x_90 = roi_info[0][9]
                # self.roi.workflow_th(image=r'%s\sample_%s_%s_%s.png' % (self.savepath, self.distance, self.distance_y, 270),
                #                   mask=self.mask, num=1, margin=0)
                # # print(roi.get_roi_list())
                # roi_info = self.roi.get_roi_list()
                # center_x_270 = roi_info[0][9]
                distance = int(center_x_90 - self.size / 2) + np.random.randint(-3, 3)
                # print('zz', distance)
                self.move_distance -= distance
                self.roi.workflow_th(image=r'%s\sample_%s_%s_%s.png' % (self.savepath, self.distance, self.distance_y, 0),
                                  mask=self.mask, num=1, margin=0)
                # print(roi.get_roi_list())
                roi_info = self.roi.get_roi_list()
                roi_left_top = [roi_info[0][1], roi_info[0][2]]
                roi_right_top = [roi_info[0][3], roi_info[0][4]]
                center_x_0 = roi_info[0][9]
                # self.roi.workflow_th(image=r'%s\sample_%s_%s_%s.png' % (self.savepath, self.distance, self.distance_y, 180),
                #                   mask=self.mask, num=1, margin=0)
                # # print(roi.get_roi_list())
                # roi_info = self.roi.get_roi_list()
                # center_x_180 = roi_info[0][9]
                distance_y = int(center_x_0 - self.size / 2) + np.random.randint(-3, 3)
                self.move_distance_y -= distance_y
                print('ZZ', distance, distance_y)
                self.move_sample(-distance, -distance_y)
                # print(roi_info)
                # print(roi_right_top[0] - roi_left_top[0])
                time_rot_2 = time.time()
                rotate_sample_360(sample=self.sample, num=12)
                time_rot_end_2 = time.time()
                time_con_2 = time_rot_end_2 - time_rot_2
                with open('log.txt', 'a') as f:
                    f.write("%s: %s\n" % ("Coarse Rotate Consume: ", time_con_2))
                # print(roi_info)
                # print(roi_right_top[0] - roi_left_top[0])
                print('Rough Alignment Down!')
                break
        return self.sample


class PreciseAlignment(object):
    """Automated precise alignment algorithm

      Attributes:
        sample: Rotating sample.
        roi: roi filter module.
        orignal_distance: Sample horizontal drift distance.
        DNA_SIZE: Number of DNA in an individual.
        POP_SIZE: Number of individuals in the population.
        CROSSOVER_RATE: Probability of gene crossover.
        MUTATION_RATE: Probability of genetic mutation.
        N_GENERATIONS: Number of iterations of the population.
        DISTANCE_BOUND: Adjustment range of distance moved.
        move_distance: Optimal movement distance.
    """

    def __init__(self, sample, mask=[]):
        self.sample = sample
        self.roi = RoiSearch()
        self.orignal_distance = self.sample.distance
        self.orignal_distance_y = self.sample.distance_y
        self.DNA_SIZE = 24
        self.POP_SIZE = 20
        self.CROSSOVER_RATE = 0.8
        self.MUTATION_RATE = 0.005
        self.N_GENERATIONS = 15
        self.DISTANCE_BOUND = [-4, 4]
        self.mask = mask
        self.move_distance = 0
        self.move_distance_y = 0

    def init_sample(self):
        self.sample.init_distance()

    def get_move_distance(self):
        return self.move_distance

    def move_sample(self, distance_x, distance_y):
        # print('1')
        # print(distance_y)
        self.sample.move(-distance_x, -distance_y)

    def translateDNA(self, pop):
        """Dissecting the DNA of a population"""
        print('Translate DNA...')
        left_pop = pop[:, 1::2]
        front_pop = pop[:, ::2]
        distances = left_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * \
               (self.DISTANCE_BOUND[1] - self.DISTANCE_BOUND[0]) + self.DISTANCE_BOUND[0]
        distances_y = front_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * \
                (self.DISTANCE_BOUND[1] - self.DISTANCE_BOUND[0]) + self.DISTANCE_BOUND[0]
        print('Translate DNA Down')
        for i in range(len(distances)):
            flag = distances[i] % 0.01
            if flag > 0.005:
                distances[i] = (int(distances[i] * 100) + 1) / 100
            else:
                distances[i] = int(distances[i] * 100) / 100
        for i in range(len(distances_y)):
            flag = distances_y[i] % 0.01
            if flag > 0.005:
                distances_y[i] = (int(distances_y[i] * 100) + 1) / 100
            else:
                distances_y[i] = int(distances_y[i] * 100) / 100
        return distances, distances_y

    def cal_img(self, distances, distances_y):
        """Calculate the fitness"""
        print('Calculate Image...')
        pred = []
        for i in range(len(distances)):
            # mis = 0
            flag = math.fabs(distances[i]) % 1
            if flag > 0.5:
                mis = 1 - flag
            else:
                mis = flag
            flag_y = math.fabs(distances_y[i]) % 1
            if flag_y > 0.5:
                mis_y = 1 - flag_y
            else:
                mis_y = flag_y
            self.move_sample(distances[i], distances_y[i])
            rotate_sample(self.sample, 90)
            self.roi.workflow_th(image=r'%s/sample_%s_%s_%s.png' % (self.sample.savepath, self.sample.distance, self.sample.distance_y, 90),
                              mask=self.mask, num=1, margin=0)
            roi_info_90 = self.roi.get_roi_list()
            center_90 = roi_info_90[0][9]

            rotate_sample(self.sample, 270)
            self.roi.workflow_th(image=r'%s/sample_%s_%s_%s.png' % (self.sample.savepath, self.sample.distance, self.sample.distance_y, 270),
                              mask=self.mask, num=1, margin=0)
            roi_info_270 = self.roi.get_roi_list()
            center_270 = roi_info_270[0][9]

            pred_value = (math.fabs(center_90 - center_270) + 2 * mis) * 1000

            rotate_sample(self.sample, 0)
            self.roi.workflow_th(image=r'%s/sample_%s_%s_%s.png' % (self.sample.savepath, self.sample.distance, self.sample.distance_y, 0),
                              mask=self.mask, num=1, margin=0)
            roi_info_0 = self.roi.get_roi_list()
            center_0 = roi_info_0[0][9]

            rotate_sample(self.sample, 180)
            self.roi.workflow_th(image=r'%s/sample_%s_%s_%s.png' % (self.sample.savepath, self.sample.distance, self.sample.distance_y, 180),
                              mask=self.mask, num=1, margin=0)
            roi_info_180 = self.roi.get_roi_list()
            center_180 = roi_info_180[0][9]

            pred_value += (math.fabs(center_180 - center_0) + 2 * mis_y) * 1000


            print('D: ', distances[i])
            print('D_Y: ', distances_y[i])
            print('Center 90 X: ', center_90)
            print('Center 270 X: ', center_270)
            print('Center 0 X: ', center_0)
            print('Center 180 X: ', center_180)
            print('Mis: ', mis)
            print('Pred: ', pred_value)
            pred.append(pred_value)
            self.init_sample()

        print('Calculate Image Down')
        return pred

    def get_fitness(self, pop):
        print('Get Fitness...')
        distances, distance_y = self.translateDNA(pop)
        pred = self.cal_img(distances, distance_y)
        print('Pred ', pred)
        print('Get Down')
        # The smaller the fitness, the better
        return -(pred - np.max(pred)) + 1e-3

    def crossover_and_mutation(self, pop):
        print('Crossover...')
        new_pop = []
        for father in pop:
            child = father
            if np.random.rand() < self.CROSSOVER_RATE:
                mother = pop[np.random.randint(self.POP_SIZE)]
                cross_points = np.random.randint(low=0, high=self.DNA_SIZE * 2)
                child[cross_points:] = mother[cross_points:]
            self.mutation(child)
            new_pop.append(child)
        print('Crossover Down')
        return new_pop

    def mutation(self, child):
        print('Mutation...')
        if np.random.rand() < self.MUTATION_RATE:
            mutate_point = np.random.randint(0, self.DNA_SIZE * 2)
            child[mutate_point] = child[mutate_point] ^ 1
        print('Mutation Down')

    def select(self, pop, fitness):
        print('Select...')
        idx = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE,
                               replace=True, p=fitness / (fitness.sum()))
        print('Select Down')
        return pop[idx]

    def print_info(self, pop, fitness):
        fitness = fitness
        print('Fitness: ', fitness)
        max_fitness_index = np.argmax(fitness)
        print("Max fitness:", fitness[max_fitness_index])
        distances, distances_y = self.translateDNA(pop)

        print("最优的基因型：", pop[max_fitness_index])
        print("Distance: ", distances)
        print("Distance_Y: ", distances_y)
        print("Distance Fine: ", (distances[max_fitness_index]))
        print("Distance_Y Fine: ", (distances_y[max_fitness_index]))
        with open('log.txt', 'a') as f:
            f.write("%s: %s\n" % ("Fitness", fitness))
            f.write("%s: %s\n" % ("Max Fitness", fitness[max_fitness_index]))
            f.write("%s: %s\n" % ("Fine Gene", pop[max_fitness_index]))
            f.write("%s: %s\n" % ("Distance", distances))
            f.write("%s: %s\n" % ("Distance_Y", distances_y))
            f.write("%s: %s\n" % ("Distance Fine", distances[max_fitness_index]))
            f.write("%s: %s\n" % ("Distance_Y Fine", distances_y[max_fitness_index]))
        # return distances

    def print_final_info(self, pop):
        fitness = self.get_fitness(pop)
        print('Fitness: ', fitness)
        max_fitness_index = np.argmax(fitness)
        print("Max fitness:", fitness[max_fitness_index])
        distances, distances_y = self.translateDNA(pop)
        print("最优的基因型：", pop[max_fitness_index])
        print("Distance: ", distances)
        print("Distance_Y: ", distances_y)
        self.move_distance = distances[max_fitness_index]
        self.move_distance_y = distances_y[max_fitness_index]
        print("Distance Fine: ", (distances_y[max_fitness_index]))
        with open('log.txt', 'a') as f:
            f.write("%s: %s\n" % ("Fitness", fitness))
            f.write("%s: %s\n" % ("Max Fitness", fitness[max_fitness_index]))
            f.write("%s: %s\n" % ("Fine Gene", pop[max_fitness_index]))
            f.write("%s: %s\n" % ("Distance", distances))
            f.write("%s: %s\n" % ("Distance_Y", distances_y))
            f.write("%s: %s\n" % ("Distance Fine", distances[max_fitness_index]))
            f.write("%s: %s\n" % ("Distance_Y Fine", distances_y[max_fitness_index]))

    def precise_alignment(self):
        """Automated precise alignment workflow for samples"""
        pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE * 2))
        y = []
        y2 = []
        x = []
        for gen in range(self.N_GENERATIONS):
            print('Generation ', gen)
            distances, distances_y = self.translateDNA(pop)
            print(len(distances_y))
            for i in range(len(distances)):
                y.append(self.sample.distance - distances[i])
                y2.append(self.sample.distance_y - distances_y[i])
                x.append(gen)
            pop = np.array(self.crossover_and_mutation(pop))
            fitness = self.get_fitness(pop)
            print('Generation Results ', gen)
            self.print_info(pop, fitness)
            pop = self.select(pop, fitness)

        # The process of population convergence with coordinates
        plt.figure()
        plt.grid()
        plt.axis('equal')
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.yticks(fontproperties='Times New Roman', size=28)
        plt.xticks(fontproperties='Times New Roman', size=28)
        plt.scatter(x, y)
        f = plt.gcf()
        f.savefig('%s/gen_%s_%s_axis.png' % (self.sample.savepath, self.POP_SIZE, self.N_GENERATIONS),
                  bbox_inches='tight')
        f.clear()
        plt.close()

        plt.figure()
        plt.grid()
        plt.axis('equal')
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.yticks(fontproperties='Times New Roman', size=28)
        plt.xticks(fontproperties='Times New Roman', size=28)
        plt.scatter(x, y2)
        f = plt.gcf()
        f.savefig('%s/gen_%s_%s_axis_2.png' % (self.sample.savepath, self.POP_SIZE, self.N_GENERATIONS),
                  bbox_inches='tight')
        f.clear()
        plt.close()

        # The process of population convergence without coordinates
        plt.figure()
        plt.grid()
        plt.axis('equal')
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.yticks(fontproperties='Times New Roman', size=28)
        plt.xticks(fontproperties='Times New Roman', size=28)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.scatter(x, y)
        f = plt.gcf()
        f.savefig('%s/gen_%s_%s.png' % (self.sample.savepath, self.POP_SIZE, self.N_GENERATIONS),
                  bbox_inches='tight')
        f.clear()
        plt.close()

        plt.figure()
        plt.grid()
        plt.axis('equal')
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.yticks(fontproperties='Times New Roman', size=28)
        plt.xticks(fontproperties='Times New Roman', size=28)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.scatter(x, y2)
        f = plt.gcf()
        f.savefig('%s/gen_%s_%s_2.png' % (self.sample.savepath, self.POP_SIZE, self.N_GENERATIONS),
                  bbox_inches='tight')
        f.clear()
        plt.close()
        self.print_final_info(pop)


if __name__ == '__main__':
    distance_x = 35.0
    distance_y = -75.0
    time_start = time.time()
    print('Start: ', time_start)
    # sample = init_sample(distance_x=distance_x, distance_y=distance_y, detector=500, savepath='rotate_xrf', filepath=r'D:\XRF_rec\8348_filter_mod')
    sample = RotationSample(distance_x=distance_x, distance_y=distance_y, detector=500, savepath='align10',
                            filepath=r'D:\XRF_rec\8348')
    time_init = time.time()
    with open('log.txt', 'a') as f:
        f.write("%s: %s\n" % ("Coarse Init: ", time_init))
    print('Init time: ', time_init - time_start)
    # rough_alignment = RoughAlignment(sample=sample)
    # rough_sample = rough_alignment.rough_alignment()
    # time_rough = time.time()
    # print('Rough time: ', time_rough - time_init)
    precise_alignment = PreciseAlignment(sample=sample)
    precise_alignment.precise_alignment()
    #     # return distances
    print(precise_alignment.sample.time_consume)
    time_precise = time.time()
    print('Precise time: ', time_precise - time_init)
    # sample = init_sample(distance=distance, detector=500, savepath='rotate_58', filepath=r'D:\tomo_00058_rec_full_re')
    # time_init = time.time()
    # print('Init time: ', time_init - time_start)
    # rotate_sample_360(sample, 360)
    # time_rotate = time.time()
    # print('Rotate time: ', time_rotate - time_init)
    # for i in range(300):
    #     sample.move(-1)
    #     sample.rotate(90)
    #     sample.rotate(270)
    time_end = time.time()
    with open('log.txt', 'a') as f:
        f.write("%s: %s\n" % ("Time Start: ", time_start))
        f.write("%s: %s\n" % ("Time Init: ", time_init))
        # f.write("%s: %s\n" % ("Time Rough: ", time_rough))
        f.write("%s: %s\n" % ("Fine Consume: ", precise_alignment.sample.time_consume))
        f.write("%s: %s\n" % ("Time End: ", time_end))
    print('All time: ', time_end - time_start)
