import numpy as np
from env.obs import Obstacle
from env.obs_car import Obstacle_car
#from env.inter_area import Inter_area
from shapely.geometry import Polygon
from env.smo import SMO

class Bicycle():
    def __init__(self):
        self.T = 0.2    # sample time
        # self.lr = 1.2
        # self.lf = 0.8
        self.lr = 2.4
        self.lf = 1.6
        self.w = 0.5

        # self.lr = 1
        # self.lf = 1
        
        # target three cars & five cars
        self.target = [27.75, 22, 0, 0] # target x, target y, target heading, target velocity
        # target four cars
        # self.target = [27.75, 25, 0, 0] # target x, target y, target heading, target velocity

        self.threshold = [0.2, np.pi*20/180, 1]
        
        self.dim_state = 4
        self.x = 0
        self.y = 0
        
        self.theta = 0
        self.v = 0
        self.v_range = [-1, 1]

        self.dim_action = 2
        self.a = 0
        self.beta = 0   # the slip angle of vehicle
        self.delta = 0 
        # self.a_range = [-0.5, 0.5]
        # self.a_range = [-1, 1]
        self.a_range = [-2, 0.5]
        self.delta_range = [-np.pi*80/180, np.pi*80/180]
        self.beta_range = [self.get_beta(self.delta_range[0]), self.get_beta(self.delta_range[1])] 

        self.obs = Obstacle()
        self.obs_car = Obstacle_car()
        self.last_state = [self.x, self.y, self.theta, self.v]

        self.t_max = 100.0
        self.dt = 0.01
        self.time = np.arange(0, self.t_max, self.dt)
        self.x_true = np.zeros((len(self.time), 2))
        self.x_hat = np.zeros((len(self.time), 2))
        self.u = 1  # input
        self.x_true[0] = np.array([1.0, 0.0])
        self.x_hat[0] = np.array([0.9, 0.0])
        
        self.disturbance = np.random.normal(-0.01, 0.01, len(self.time))  # 随机扰动

    def reset(self):
        # Example 1
        # self.start_x = 15
        # self.start_y = 15
        # # Example 2
        self.start_x = 0
        self.start_y = 13.75
        # # Example 3
        # self.start_x = 16.25
        # self.start_y = 0
        # Example 4
        # self.start_x = 0
        # self.start_y = 0

        self.x = self.start_x 
        self.y = self.start_y 
        # start heading
        self.theta = 0
        # self.theta = np.pi/4
        # self.theta = np.pi/2

        self.v = 0.1
        self.a = 0
        self.beta = 0
        self.delta = 0

    def get_beta(self, delta):
        temp = np.arctan(self.lr*np.tan(delta)/(self.lr+self.lf))
        return temp
    
    def step(self, action,i):
        self.last_state = [self.x, self.y, self.theta, self.v]

        self.a = action[0]
        self.delta = action[1]
        self.beta = self.get_beta(self.delta)

        for i in range(1, len(self.time)):
            s = SMO.sliding_surface(self.x_true[i-1], self.x_hat[i-1])
            control = SMO.sliding_mode_control(s)
            self.x_true[i] = self.x_true[i-1] + SMO.nonlinear_system_with_disturbance(self.x_true[i-1], self.u, self.disturbance[i-1]) * self.dt
            self.x_hat[i] = self.x_hat[i-1] + (SMO.nonlinear_system_with_disturbance(self.x_hat[i-1], self.u, 0.0) + control) * self.dt
        self.smo= self.x_true- self.x_hat
        #withpoout disturbance observer
        self.x = self.x + self.v * np.cos(self.theta + self.beta) * self.T+self.disturbance[i]
        self.y = self.y + self.v * np.sin(self.theta + self.beta) * self.T+self.disturbance[i]
        #with disturbance observer
        # self.x = self.x + self.v * np.cos(self.theta + self.beta) * self.T+ self.smo[i,0]
        # self.y = self.y + self.v * np.sin(self.theta + self.beta) * self.T+ self.smo[i,1]
        self.theta = self.theta + self.v / self.lr *np.sin(self.beta) * self.T
        self.v = self.v + self.a * self.T
        self.uncertainty()

    def get_reward(self):
        d_last = np.sqrt((self.target[0] - self.last_state[0]) ** 2 + (self.target[1] - self.last_state[1]) ** 2)
        d = np.sqrt((self.target[0] - self.x) ** 2 + (self.target[1] - self.y) ** 2)
        if d_last > d:
            r_pos = 1 - d
        else:
            r_pos = -1 - d  

        # heading reward
        r_heading = 0
        
        if d < 0.5 and  np.sqrt((self.theta-self.target[2] )**2)< np.pi*5/180 and (self.v-self.target[3]) < 1:
            r_heading = 10

        # obstacle reward
        r_o = 0
        for i in range(self.obs.num_obs):
            r_obs = (self.obs.x[i] - self.x) ** 2 + (self.obs.y[i] - self.y) ** 2
            if r_obs < (self.obs.r[i] ** 2) * 1.1: # 1.2 is a safety factor, was 1.69
                r_o -= 20 / np.sqrt(r_obs)
        for i in range(self.obs_car.num_obs):
            r_obs = (self.obs_car.x[i] - self.x) ** 2 + (self.obs_car.y[i] - self.y) ** 2
            if r_obs < (self.obs_car.r[i] ** 2) * 1.1:
                r_o -= 20 / np.sqrt(r_obs)
        # if self.x < self.target[0]+1.2 and self.x > self.target[0]-1.2 and self.y < self.target[1]+0.8 and self.y > self.target[1]-0.8:
        #     r_o += 10
                
        # rode reward
        if self.x < 0 or self.x > 26 or self.y < 0 or self.y > 30:
            r_o -= 20
        if self.y >18 and self.x > 0 and self.x < 12.5: 
            r_o -= 10
        # if self.y > 17 and self.x > 18 and self.x < 21:
        #     r_o -= 10
        if self.y <12.5:
            r_o -= 10

        # area reward
        r_area = 0
        polygon = Polygon([(self.target[0]-1.2,self.target[1]-0.8),(self.target[0]-1.2,self.target[1]+0.8),(self.target[0]+1.2,self.target[1]+0.8),(self.target[0]+1.2,self.target[1]-0.8)])
        other_polygon = Polygon([(self.x + self.w*self.lr*np.cos(self.theta) - self.w*self.lf*np.sin(self.theta),
                                  self.y + self.w*self.lr*np.sin(self.theta) + self.w*self.lf*np.cos(self.theta)),
                                  ( self.x + self.w*self.lr*np.cos(self.theta) + self.w*self.lf*np.sin(self.theta),
                                   self.y + self.w*self.lr*np.sin(self.theta) - self.w*self.lf*np.cos(self.theta)),
                                   ( self.x - self.w*self.lr*np.cos(self.theta) + self.w*self.lf*np.sin(self.theta),
                                    self.y - self.w*self.lr*np.sin(self.theta) - self.w*self.lf*np.cos(self.theta)),
                                    ( self.x -  self.w*self.lr*np.cos(self.theta) - self.w*self.lf*np.sin(self.theta),
                                      self.y -  self.w*self.lr*np.sin(self.theta) + self.w*self.lf*np.cos(self.theta))])
        intersection = polygon.intersection(other_polygon).area


        if intersection> 0.80*self.lr*self.lf:#intersection
            r_area = 10
        
        # reward function
        done = False
        if d < self.threshold[0]: # and r_heading < self.threshold[1] and r_velocity < self.threshold[2]:
            done = True
        #     return float(2 * r_pos + 1 * r_o  +1*r_heading - 0.1 * self.v ** 2 + 10*r_area) + 500, done
        # else:
        #     return float(2 * r_pos + 1 * r_o +1*r_heading - 0.1 * self.v ** 2 + 10*r_area), done
            return float(2 * r_pos + 1 * r_o +1 * r_heading - 1 * self.v ** 2 + 1*r_area)+500, done
        else:
            return float(2 * r_pos + 1 * r_o +1 * r_heading - 1 * self.v ** 2 + 1*r_area), done
        
    
    def get_state(self):
        return [self.x, self.y, self.theta, self.v]
    
    def uncertainty(self):
        x_d = self.v * np.cos(self.theta) * 0.1 * self.T
        y_d = self.v * np.sin(self.theta) * 0.1 * self.T 
        self.x += x_d
        self.y += y_d
        return np.array([x_d, y_d, 0, 0]).reshape(4,1)
    
    def get_fg(self):
        fx = [self.v * np.cos(self.theta),
              self.v * np.sin(self.theta),
              0,
              0]
        
        gx = [[0, -self.v * np.sin(self.theta)],
              [0, self.v * np.cos(self.theta)],
              [0, self.v / self.lr],
              [1, 0]]
        
        return fx, gx

 