import sys
# set the path to the bicycle model
sys.path.append(r'D:\xxxx\xxxx\parking')
import matplotlib.pyplot as plt
import numpy as np
from env.bicycle_model import Bicycle
from scipy.spatial.transform import Rotation as Rot

class EKF():
    def __init__(self, env):
        self.Q = np.diag([
            0.2,  # variance of location on x-axis
            0.2,  # variance of location on y-axis
            np.deg2rad(10.0),  # variance of yaw angle
            0.2  # variance of velocity
        ]) ** 2
        self.R = np.diag([0.3, 0.3]) ** 2
        self.env = env 
        self.Dt = self.env.T

    # beta a
    def motion_model(self, state, action):
        beta = np.arctan(self.env.lr/(self.env.lr + self.env.lf)*np.tan(action[0][0]))
        self.F = np.array([[1.0, 0, -state[3][0]*np.sin(state[2][0]+beta)*self.Dt, np.cos(state[2][0]+beta)*self.Dt],
                    [0, 1.0, state[3][0]*np.cos(state[2][0]+beta)*self.Dt, np.sin(state[2][0]+beta)*self.Dt],
                    [0, 0, 1.0, np.sin(beta)/self.env.lr*self.Dt],
                    [0, 0, 0, 1]])

        self.B = np.array([[-state[3][0]*np.sin(state[2][0]+beta)*self.Dt, 0],
                    [state[3][0]*np.cos(state[2][0]+beta)*self.Dt, 0],
                    [state[3][0]*np.cos(beta)/self.env.lr*self.Dt, 0],
                    [0, self.Dt]])
        x = state
        u = [[beta], [action[1][0]]]
        x = self.F @ state + self.B @ u

        return x
    
    def motion_model_tire(self, state, action):
        delta = action[1]
        F = action[0]
        x_new = state[0] + (state[3] * np.cos(state[2]) - state[4] * np.sin(state[2])) * self.Dt
        y_new = state[1] + (state[3] * np.sin(state[2]) + state[4] * np.cos(state[2])) * self.Dt
        phi_new = state[2] + state[5] * self.Dt

        alpha_f = delta - np.arctan((state[5] * self.env.lf + state[4]) / (state[3]))
        alpha_r = np.arctan((state[5] * self.env.lr - state[4]) / (state[3]))

        vx_new = state[3] + (F - self.env.Cf * alpha_f * np.sin(delta) + self.env.m * state[4] * state[5]) / self.env.m * self.Dt
        vy_new = state[4] + (self.env.Cr * alpha_r + self.env.Cf * alpha_f * np.cos(delta) - self.m * state[3] * state[5]) / self.env.m * self.Dt
        w_new = state[5] + (self.env.lf * self.env.Cf * alpha_f * np.cos(delta) - self.env.lr * self.env.Cr * alpha_r) / self.env.Iz * self.Dt

        vx_new = np.clip(vx_new, self.env.v_range[0], self.env.v_range[1])
        vy_new = np.clip(vy_new, self.env.v_range[0], self.env.v_range[1])

        return np.array([x_new, y_new, phi_new, vx_new, vy_new, w_new]).reshape((6,1))
    
    def observation_model(self, state):
        x = state
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = self.H @ x
        
        return z
    
    def observation_model_tire(self, state):
        x = state
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        z = H @ x
        
        return z
    
    def ekf_estimation(self, xEst, PEst, z, u):
        #  Predict
        xPred = self.motion_model(xEst, u)
        PPred = self.F @ PEst @ (self.F).T + self.Q

        #  Update
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = self.H @ PPred @ (self.H).T + self.R
        K = PPred @ (self.H).T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ self.H) @ PPred
        return xEst, PEst


def rot_mat_2d(angle):
    return  Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * np.pi + 0.1, 0.1)
    a = np.sqrt(eigval[bigind])
    b = np.sqrt(eigval[smallind])
    x = [a * np.cos(it) for it in t]
    y = [b * np.sin(it) for it in t]
    angle = np.arctan2(eigvec[1, bigind], eigvec[0, bigind])
    
    fx = rot_mat_2d(angle) @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")
        
def main():
    INPUT_NOISE = np.diag([np.deg2rad(10.0), 0.1]) ** 2
    GPS_NOISE = np.diag([0.5, 0.5]) ** 2
    env = Bicycle()
    ekf = EKF(env)

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))
    SIM_TIME = 40

    while SIM_TIME >= time:
        time += env.T
        u = np.array([[(np.random.random()-0.5)*0.5], [np.random.random()*0.1]])
        
        xTrue = ekf.motion_model(xTrue, u)
        # add noise to gps x-y
        z = ekf.observation_model(xTrue) #+ GPS_NOISE @ np.random.randn(2, 1)
        # add noise to input
        ud = u + INPUT_NOISE @ np.random.randn(2, 1)
        xDR = ekf.motion_model(xDR, ud)

        xEst, PEst = ekf.ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(hz[0, :], hz[1, :], ".g")
        plt.plot(hxTrue[0, :].flatten(),
                    hxTrue[1, :].flatten(), "-b")
        plt.plot(hxDR[0, :].flatten(),
                    hxDR[1, :].flatten(), "-k")
        plt.plot(hxEst[0, :].flatten(),
                    hxEst[1, :].flatten(), "-r")
        plot_covariance_ellipse(xEst, PEst)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
    plt.show()


if __name__ == '__main__':
    main()
