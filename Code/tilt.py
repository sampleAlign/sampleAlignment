import numpy as np
import math
from tilt_sample import WalnutSample, TiltSample


class RoughAdjustment(object):
    """Rotation axis coarse alignment algorithm.

      Attributes:
        tilt_sample: Sample model with tilted rotation axis.
        rough_left_tilt_theta: Adjustment angle of left and right direction.
        rough_front_tilt_theta: Adjustment angle of front and back direction.

    """
    def __init__(self, tilt_sample):
        self.tilt_sample = tilt_sample
        self.rough_left_tilt_theta = 0
        self.rough_front_tilt_theta = 0

    def get_left_theta(self):
        return self.rough_left_tilt_theta

    def get_front_theta(self):
        return self.rough_front_tilt_theta

    def cal_tilt(self, precision=1):
        """Calculate the tilt angle of the current sample rotation axis"""
        rotate_points = self.tilt_sample.rotate_360()
        x = []
        z = []
        for point in rotate_points:
            x.append(point[0])
            z.append(point[1])
        distances = []
        center_index_list = []
        for i in range(int(self.tilt_sample.num / 2)):
            if math.fabs(x[i] - x[i + 180]) < precision:
                center_index_list.append(i)
            distances.append((x[i] - x[i + 180]) * (x[i] - x[i + 180]) + (z[i] - z[i + 180]) * (z[i] - z[i + 180]))
        while True:
            if len(center_index_list) > 0:
                break
            for i in range(int(self.tilt_sample.num / 2)):
                if math.fabs(x[i] - x[i + 180]) < precision:
                    center_index_list.append(i)
                    break
            precision = precision * 2
        center_index = center_index_list[int(len(center_index_list) / 2)]
        d_max = np.max(distances)
        d_max_index = distances.index(d_max)
        d_min_index = center_index
        print('d_max_index: ', d_max_index)
        print('d_min_index: ', d_min_index)
        x_90 = x[d_max_index]
        z_90 = z[d_max_index]
        x_270 = x[d_max_index + 180]
        z_270 = z[d_max_index + 180]
        x_0 = x[d_min_index]
        z_0 = z[d_min_index]
        x_180 = x[d_min_index + 180]
        z_180 = z[d_min_index + 180]
        left_tilt_theta = math.atan((z_270 - z_90) / (x_270 - x_90)) / math.pi * 180
        self.rough_left_tilt_theta = left_tilt_theta
        print('rough_left_tilt_theta: ', left_tilt_theta)
        front_tilt_theta = math.asin(math.sqrt((x_180 - x_0) * (x_180 - x_0) + (z_180 - z_0) * (z_180 - z_0)) /
                                     math.sqrt((x_270 - x_90) * (x_270 - x_90) + (z_270 - z_90) * (
                                             z_270 - z_90))) / math.pi * 180
        self.rough_front_tilt_theta = front_tilt_theta
        print('rough_front_tilt_theta: ', front_tilt_theta)

    def rough_adjustment(self, precision=1):
        """Automated coarse alignment workflow for rotation axis"""
        self.cal_tilt(precision=precision)
        left = math.tan(self.rough_left_tilt_theta * math.pi / 180)
        front = math.tan(self.rough_front_tilt_theta * math.pi / 180)
        init_z = self.tilt_sample.axis[2]
        adj_x = round(init_z * left)
        adj_y = round(init_z * front)
        print(left, front)
        print(adj_x, adj_y)
        self.tilt_sample.adjustment_axis(adj_x, adj_y)
        self.tilt_sample.rotate_360()
        return self.tilt_sample


class PreciseAdjustment(object):
    """Rotation axis precise alignment algorithm.

      Attributes:
        tilt_sample: Rotating sample.
        init_axis: Rotation axis of the sample.
        precise_left_tilt_theta: Adjustment angle of left and right direction.
        precise_front_tilt_theta: Adjustment angle of front and back direction.
        DNA_SIZE: Number of DNA in an individual.
        POP_SIZE: Number of individuals in the population.
        CROSSOVER_RATE: Probability of gene crossover.
        MUTATION_RATE: Probability of genetic mutation.
        N_GENERATIONS: Number of iterations of the population.
        LEFT_BOUND: Left and right adjustment range of rotation axis.
        FRONT_BOUND: Front and back adjustment range of rotation axis.
    """
    def __init__(self, tilt_sample):
        self.init_axis = tilt_sample.axis
        self.tilt_sample = tilt_sample
        self.precise_left_tilt_theta = 0
        self.precise_front_tilt_theta = 0
        self.DNA_SIZE = 24
        self.POP_SIZE = 20
        self.CROSSOVER_RATE = 0.8
        self.MUTATION_RATE = 0.005
        self.N_GENERATIONS = 10
        self.LEFT_BOUND = [-3, 3]
        self.FRONT_BOUND = [-3, 3]

    def get_left(self):
        return self.precise_left_tilt_theta

    def get_front(self):
        return self.precise_front_tilt_theta

    def move_sample(self, rough_left_tilt_theta, rough_front_tilt_theta):
        """Adjust the tilt angle of the rotation axis"""
        left = math.tan(rough_left_tilt_theta * math.pi / 180)
        front = math.tan(rough_front_tilt_theta * math.pi / 180)
        init_z = self.tilt_sample.axis[2]
        adj_x = round(init_z * left)
        adj_y = round(init_z * front)
        print(left, front)
        print(adj_x, adj_y)
        self.tilt_sample.adjustment_axis(adj_x, adj_y)

    def translateDNA(self, pop):
        print('Translate DNA...')
        left_pop = pop[:, 1::2]
        front_pop = pop[:, ::2]
        left = left_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * \
               (self.LEFT_BOUND[1] - self.LEFT_BOUND[0]) + self.LEFT_BOUND[0]
        front = front_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * \
                (self.FRONT_BOUND[1] - self.FRONT_BOUND[0]) + self.FRONT_BOUND[0]
        print('Translate DNA Down')
        return left, front

    def cal_tilt(self, left, front):
        """Calculate the fitness"""
        pred = []
        for i in range(len(left)):
            self.move_sample(left[i], front[i])
            rotate_points = self.tilt_sample.rotate_360()
            z = []
            for point in rotate_points:
                z.append(point[1])
            z_max = np.max(z)
            z_min = np.min(z)
            self.move_sample(-left[i], -front[i])
            pred.append(math.fabs(z_max - z_min))
        return pred

    def get_fitness(self, pop):
        print('Get Fitness...')
        left, front = self.translateDNA(pop)
        pred = self.cal_tilt(left, front)
        print('Get Down')
        return -(pred - np.max(pred)) + 1e-3

    def crossover_and_mutation(self, pop):
        new_pop = []
        for father in pop:
            child = father
            if np.random.rand() < self.CROSSOVER_RATE:
                mother = pop[np.random.randint(self.POP_SIZE)]
                cross_points = np.random.randint(low=0, high=self.DNA_SIZE * 2)
                child[cross_points:] = mother[cross_points:]
            self.mutation(child)
            new_pop.append(child)
        return new_pop

    def mutation(self, child):
        if np.random.rand() < self.MUTATION_RATE:
            mutate_point = np.random.randint(0, self.DNA_SIZE * 2)
            child[mutate_point] = child[mutate_point] ^ 1

    def select(self, pop, fitness):
        idx = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True,
                               p=(fitness) / (fitness.sum()))
        return pop[idx]

    def print_info(self, pop):
        fitness = self.get_fitness(pop)
        max_fitness_index = np.argmax(fitness)
        print("max_fitness:", fitness[max_fitness_index])
        left, front = self.translateDNA(pop)
        print("最优的基因型：", pop[max_fitness_index])
        print("(left_theta, front_theta):", (left[max_fitness_index], front[max_fitness_index]))
        left = math.tan(left[max_fitness_index] * math.pi / 180)
        front = math.tan(front[max_fitness_index] * math.pi / 180)
        print("(left, front):", (left, front))
        init_z = self.tilt_sample.axis[2]
        adj_x = round(init_z * left)
        adj_y = round(init_z * front)
        print(left, front)
        print(adj_x, adj_y)
        self.tilt_sample.adjustment_axis(adj_x, adj_y)
        self.tilt_sample.rotate_360()

    def precise_ajustment(self):
        """Automated precise alignment workflow for rotation axis"""
        pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE * 2))
        for gen in range(self.N_GENERATIONS):
            print('Generation ', gen)
            # x = self.translateDNA(pop)
            pop = np.array(self.crossover_and_mutation(pop))
            fitness = self.get_fitness(pop)
            pop = self.select(pop, fitness)
        self.print_info(pop)


if __name__ == '__main__':
    import time

    walnut_sample = WalnutSample(axis=[-10, 10, 40], savepath='film_2', detector=500,
                                 filepath=r'D:\XRF_rec\8348_filter_mod', num=360)
    # walnut_sample.rotate_theta(0)
    walnut_sample.rotate_theta(90)
    # walnut_sample.rotate_theta(180)
    # walnut_sample.rotate_theta(270)
    # walnut_sample.rotate_theta(215)
    # walnut_sample.rotate_theta(119)
    # walnut_sample.rotate_theta(299)
    #
    # walnut_sample = WalnutSample(axis=[1, 1, 40], savepath='axisTest9',
    #                              filepath=r'C:\Users\zhaigj\Desktop\sampleAlignment\data', num=360)
    # walnut_sample.rotate_theta(35)
    # walnut_sample.rotate_theta(215)
    # walnut_sample.rotate_theta(119)
    # walnut_sample.rotate_theta(299)
    #
    # walnut_sample = WalnutSample(axis=[0, 0, 40], savepath='axisTest9',
    #                              filepath=r'C:\Users\zhaigj\Desktop\sampleAlignment\data', num=360)
    # walnut_sample.rotate_theta(35)
    # walnut_sample.rotate_theta(215)
    # walnut_sample.rotate_theta(119)
    # walnut_sample.rotate_theta(299)

    # tilt_sample = TiltSample([-10, 10, 40], savepath='axisTest7', point=[0, 0, 5])
    #
    tilt_sample = TiltSample([-10, 10, 40], point=[0, 0, 2], savepath='axisTest14')
    print('ref_left_tilt_theta', tilt_sample.ref_left_tilt_theta)
    print('ref_front_tilt_theta', tilt_sample.ref_front_tilt_theta)
    time_start = time.time()
    rough_adj = RoughAdjustment(tilt_sample)
    tilt_sample_rough = rough_adj.rough_adjustment()
    time_rough = time.time()
    time.sleep(10)
    print('Rough: ', time_rough - time_start)
    precise_adj = PreciseAdjustment(tilt_sample_rough)
    precise_adj.precise_ajustment()
    time_precise = time.time()
    print('Precise: ', time_precise - time_rough)

    # tilt_sample.cal_tilt()
