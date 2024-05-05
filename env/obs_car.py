import numpy as np


class Obstacle_car():
    def __init__(self):
        # you can set the obstacle cars here

        # three cars
        # self.x = [19.25, 27.75, 27.75]
        # self.y = [19, 19, 25]
        # self.r = [1.5, 1.4, 1.4]
        # self.theta = [0, 0, 0]
        # self.num_obs = 3

        # four cars
        # self.x = [19.25, 27.75,27.75, 27.75]
        # self.y = [19, 19, 22, 28]
        # self.r = [1.5, 1.4, 1.4,1.4]
        # self.theta = [0, 0, 0, 0]
        # self.num_obs = 4

        # five cars
        self.x = [19.25, 19.25, 19.25, 27.75, 27.75]
        self.y = [19, 22, 25, 19, 25]
        self.r = [1.5, 1.5, 1.4, 1.4, 1.4]
        self.theta = [0, 0, 0, 0, 0]
        self.num_obs = 5
        
        # self.theta = [np.pi/4, np.pi/4, np.pi/4]

        self.lr = 2.4
        self.lf = 1.6
        self.w = 0.5
        
    def car(self, x, y, theta):
        lr = 2.4
        lf = 1.6
        w = 0.5
        car_l = 2
        car_w = 1
        car_x = [x + w*lr*np.cos(theta) - w*lf*np.sin(theta), x + w*lr*np.cos(theta) + w*lf*np.sin(theta), x - w*lr*np.cos(theta) + w*lf*np.sin(theta), x -  w*lr*np.cos(theta) - w*lf*np.sin(theta),x + w*lr*np.cos(theta) - w*lf*np.sin(theta)]
        car_y = [y + w*lr*np.sin(theta) + w*lf*np.cos(theta), y + w*lr*np.sin(theta) - w*lf*np.cos(theta), y - w*lr*np.sin(theta) - w*lf*np.cos(theta), y -  w*lr*np.sin(theta) + w*lf*np.cos(theta),y + w*lr*np.sin(theta) + w*lf*np.cos(theta)]
        return car_x, car_y