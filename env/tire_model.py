import numpy as np
import sys
# set the path to the parking
sys.path.append(r'D:\xxxx\xxxx\parking')
from env.obs import Obstacle
# from obs import Obstacle
import matplotlib.pyplot as plt
np.random.seed(0)
class TireModel():
    def __init__(self):
        self.T = 0.2    # sample time
        self.lr = 1.6  # m
        self.lf = 1.1  # m
        self.Cf = 3200 # N/rad
        self.Cr = 3400 # N/rad
        self.Iz = 2250 # kg/m2
        self.m = 1500.0 # kg
        self.target = [30, 25, 0, 0] # target x, target y, target heading, target velocity
        self.threshold = [2.5, np.pi*20/180, 1]
        
        self.dim_state = 6
        self.x = 0
        self.y = 0
        self.phi = 0
        self.vx = 1
        self.vy = 0.0
        self.w = 0
        self.v_range = [-3, 3]
        self.w_range = [-np.pi/3, np.pi/3]

        self.dim_action = 2
        self.delta = 0 
        self.F = 0
        self.F_range = [-2000, 5000]
        self.delta_range = [-0.6, 0.6]

        self.obs = Obstacle()
        self.last_state = [self.x, self.y, self.phi, self.vx, self.vy, self.w]

    def reset(self):
        self.x = 0
        self.y = 0
        self.phi = 0
        self.vx = 1
        self.vy = 0.0
        self.w = 0
        self.delta = 0
        self.F = 0
    
    def step(self, action):
        self.last_state = [self.x, self.y, self.phi, self.vx, self.vy, self.w]
        self.delta = action[0]
        self.F = action[1]
        self.x = self.x + (self.vx * np.cos(self.phi) - self.vy * np.sin(self.phi)) * self.T
        self.y = self.y + (self.vx * np.sin(self.phi) + self.vy * np.cos(self.phi)) * self.T
        self.phi = self.phi + self.w * self.T

        alpha_f = self.delta - np.arctan((self.w * self.lf + self.vy) / (self.vx))
        alpha_r = np.arctan((self.w * self.lr - self.vy) / (self.vx))

        self.vx = self.vx + (self.F - self.Cf * alpha_f * np.sin(self.delta) + self.m * self.vy * self.w) / self.m * self.T
        self.vy = self.vy + (self.Cr * alpha_r + self.Cf * alpha_f * np.cos(self.delta) - self.m * self.vx * self.w) / self.m * self.T
        self.w = self.w + (self.lf * self.Cf * alpha_f * np.cos(self.delta) - self.lr * self.Cr * alpha_r) / self.Iz * self.T

        self.vx = np.clip(self.vx, self.v_range[0], self.v_range[1])
        self.vy = np.clip(self.vy, self.v_range[0], self.v_range[1])
        
        _ = self.uncertainty()
    
    def step_(self, state, action):
        delta = action[0]
        F = action[1]
        x = state[0] + (state[3] * np.cos(state[2]) - state[4] * np.sin(state[2])) * self.T
        y = state[1] + (state[3] * np.sin(state[2]) + state[4] * np.cos(state[2])) * self.T
        phi = state[2] + state[5] * self.T

        alpha_f = delta - np.arctan((state[5] * self.lf + state[4]) / (state[3]))
        alpha_r = np.arctan((state[5] * self.lr - state[4]) / (state[3]))

        vx = state[3] + (F - self.Cf * alpha_f * np.sin(delta) + self.m * state[4] * state[5]) / self.m * self.T
        vy = state[4] + (self.Cr * alpha_r + self.Cf * alpha_f * np.cos(delta) - self.m * state[3] * state[5]) / self.m * self.T
        w = state[5] + (self.lf * self.Cf * alpha_f * np.cos(delta) - self.lr * self.Cr * alpha_r) / self.Iz * self.T

        vx = np.clip(vx, self.v_range[0], self.v_range[1])
        vy = np.clip(vy, self.v_range[0], self.v_range[1])

        return np.array([x, y, phi, vx, vy, w]).reshape((6,1))
    
    def get_reward(self):
        # r_p = np.sqrt((self.target[0] - self.last_state[0]) ** 2 + (self.target[1] - self.last_state[1]) ** 2)
        r_pos = np.sqrt((self.target[0] - self.x) ** 2 + (self.target[1] - self.y) ** 2)
        r_o = 0
        for i in range(self.obs.num_obs):
            r_obs = (self.obs.x[i] - self.x) ** 2 + (self.obs.y[i] - self.y) ** 2
            if r_obs < (self.obs.r[i] ** 2):
                r_o -= 50
            elif r_obs < (self.obs.r[i] ** 2) * 1.44:
                r_o -=  100 / r_obs
        done = False
        if r_pos < self.threshold[0]: # and r_heading < self.threshold[1] and r_velocity < self.threshold[2]:
            done = True
            return float( -2*r_pos + 200/r_pos + r_o) + 500, done
        else:
            # return float( -np.sqrt(r_pos) / (np.sqrt(self.target[0]**2 + self.target[1]**2))  - r_o), done
            return float( -2*r_pos + 200/r_pos + r_o), done   
         
    def get_state(self):
        return [self.x, self.y, self.phi, self.vx, self.vy, self.w]
    
    def get_fg(self):
        alpha_f = self.delta - np.arctan((self.w * self.lf + self.vy) / (self.vx))
        alpha_r = np.arctan((self.w * self.lr - self.vy) / (self.vx))
        fx = [self.vx * np.cos(self.phi) - self.vy * np.sin(self.phi), 
              self.vx * np.sin(self.phi) + self.vy * np.cos(self.phi),
              self.w,
              self.vy * self.w,
              -self.vx * self.w + self.Cr * alpha_r / self.m - self.Cf * alpha_f / self.m,
              -self.Cr * alpha_r * self.lr / self.Iz - self.Cf * alpha_f * self.lf / self.Iz]
        gx = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [1 / self.m, self.Cf * alpha_f / self.m, -self.Cf /self.m, 0],
              [0, self.Cf / self.m, self.Cf / self.m / 2 * alpha_f, -self.Cf / self.m / 2], 
              [0, self.lf * self.Cf / self.Iz, self.lf * self.Cf * alpha_f / 2 / self.Iz, -self.lf * self.Cf / self.Iz / 2]]
        
        return fx, gx

    def uncertainty(self):
        x_d = np.random.random() * 0.2 
        y_d = np.random.random() * 0.2
        self.x += x_d
        self.y += y_d
        return [x_d, y_d, 0, 0, 0, 0]
    

if __name__ == "__main__":
    env = TireModel()
    env.reset()
    state_x = []
    state_y = []
    for i in range(50):
        action = [0.00, 10]
        env.step(action)
        state_x.append(env.x)
        state_y.append(env.y)
    plt.plot(state_x, state_y)
    plt.xlim([0,5])
    plt.ylim([0,5])
    plt.show()
    