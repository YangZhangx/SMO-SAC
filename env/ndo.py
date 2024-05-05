import numpy as np
import sys
sys.path.append(r'D:\yangzhang\ma\safeRL-main\safeRL-main\parking')
from bicycle_model import Bicycle
from tire_model import TireModel
import matplotlib.pyplot as plt

class NDO:
    def __init__(self, state, env, L=0.1) -> None:
        self.L = L
        self.z = np.zeros((4,1))
        self.env = env
        self.T = self.env.T
        self.hat_w = self.z + self.L * np.array(state).reshape(4,1)

    def estimate(self, state, action):
        self.z = np.array(state).reshape(4,1)
        fx, gx = self.env.get_fg()
        gx = np.array(gx).reshape(4,2)
        a = np.array(action).reshape(2,1)
        fx = np.array(fx).reshape(4,1)
        self.z = self.z + (-self.L * self.z - self.L * (fx + np.dot(gx,a) + self.L * np.array(state).reshape(4,1))) * self.T
        hat_w = self.z + self.L * np.array(state).reshape(4,1)
        return hat_w

class ANDO:
    def __init__(self, init_state, env, A):
        self.state_hat = np.array(init_state)
        self.state_tilde = np.zeros(self.state_hat.shape)
        self.sigma_hat = np.zeros(self.state_hat.shape)
        self.env = env
        self.get_f, self.get_g = env.get_fg()
        self.get_f = np.array(self.get_f)
        self.get_g = np.array(self.get_g)
        self.dt = self.env.T
        self.Ae = A
        self.Mat_expm = np.exp(self.Ae*self.dt)
        self.Phi = (self.Mat_expm - 1.0) / self.Ae
        self.gain = -self.Mat_expm/self.Phi

    def state_predictor(self, action):
        # tire model
        # _, _, self.get_f, self.get_g = env.get_fg()
        # bicycle model
        self.get_f, self.get_g = self.env.get_fg()
        self.get_f = np.array(self.get_f)
        self.get_g = np.array(self.get_g)
        # tire model
        # action_ = [action[0], action[1], action[1] ** 2, action[1] * action[1] * action[1]]
        # bycicle model
        beta = np.arctan(self.env.lr/(self.env.lr + self.env.lf)*np.tan(action[0]))
        action_ = [action[1], beta]
        self.state_hat = self.dt * (self.get_f + self.sigma_hat + self.Ae * self.state_tilde + self.get_g @ action_) + self.state_hat
        return self.state_hat

    def adaptive_law(self, state):
        self.state_tilde = self.state_hat - state
        self.sigma_hat = self.gain * self.state_tilde
        return self.sigma_hat
    
    def disturbance_estimator(self, state, action):
        sigma_hat = self.adaptive_law(state)
        state_hat = self.state_predictor(action)
        # bycicle model
        return sigma_hat, state_hat.reshape((4,1))
        # tire model
        # return sigma_hat, state_hat.reshape((6,1))

if __name__ == "__main__":
    env = TireModel()
    state = env.get_state()
    DOB = ANDO(state, env, 1.0)
    sigma_true_ = []
    sigma_est_ = []
    index = []
    for i in range(1000):
        action = np.array([0.2 + np.random.random()*0.2, 5000])
        sigma_true = env.uncertainty()
        sigma_est = DOB.disturbance_estimator(state, action)
        env.step(action)
        state = env.get_state()
        sigma_true_.append(sigma_true[0])
        sigma_est_.append(sigma_est[0])
        index.append(i)

    plt.plot(index, sigma_est_, label="Est_x")
    plt.plot(index, sigma_true_, label="True_x")
    plt.legend()
    plt.show()