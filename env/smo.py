import numpy as np
import sys
# set the path to the parking
sys.path.append(r'D:\xxxx\xxxx\parking')
from env.tire_model import TireModel
import matplotlib.pyplot as plt

class SMO:
    def __init__(self):
        self.t_max = 50.0
        self.dt = 0.01
        self.time = np.arange(0, self.t_max, self.dt)
        self.x_true = np.zeros((len(self.time), 2))
        self.x_hat = np.zeros((len(self.time), 2))
        self.u = 1  # input
        self.x_true[0] = np.array([1.0, 0.0])
        self.x_hat[0] = np.array([0.9, 0.0])
        self.disturbance = np.random.normal(0, 0.01, len(self.time))  # 随机扰动

    
    def nonlinear_system_with_disturbance(x, u, disturbance):
        dx1 = x[1]
        dx2 = -x[0] + u + disturbance  # 加扰动使波形随机
        return np.array([dx1, dx2])

    # 滑动面
    def sliding_surface(x, x_hat):
        return x - x_hat

    # 控制律
    def sliding_mode_control(s):
        k = 0.04  # 增益
        return k * np.sign(s)


#     # 仿真参数
# # 仿真参数
# t_max = 50.0
# dt = 0.01
# time = np.arange(0, t_max, dt)
# x_true = np.zeros((len(time), 2))
# x_hat = np.zeros((len(time), 2))
# u = 1  # input
# # 初始状态
# x_true[0] = np.array([1.0, 0.0])
# x_hat[0] = np.array([0.9, 0.0])
# disturbance = np.random.normal(0, 0.01, len(time))  # 随机扰动

# # 画图 
# plt.plot(time, x_true[:, 0], label="True_x")
# plt.plot(time, x_hat[:, 0], label="Est_x")
# plt.legend()
# plt.show()

# print(disturbance)
# if __name__ == "__main__":
#     env = TireModel()
#     t_max = 50.0
#     dt = 0.01
#     time = np.arange(0, t_max, dt)
#     state = env.get_state()
#     SMO = SMO(state, env, 0.1)
#     sigma_true = [len(time), 1]
#     sigma_est = [len(time), 1]
#     sigma_true_ = []
#     sigma_est_ = []
#     sigma_true[0] = np.array([1.0, 0.0])
#     sigma_est[0] = np.array([0.9, 0.0])
#     index = []
#     for i in range(1000):
#         action = np.array([0.2 + np.random.random()*0.2, 5000])
#         print(action)
#         print(sigma_true)
#         print(sigma_est)
#         s = SMO.sliding_surface(sigma_true, sigma_est)
#         print(s)
#         control = SMO.sliding_mode_control(s)
#         sigma_true = env.uncertainty()
#         sigma_est = SMO.disturbance_estimator(state, action) + control
        
#         env.step(action)
        
#         state = env.get_state()
#         sigma_true_.append(sigma_true[0])
#         sigma_est_.append(sigma_est[0])
#         index.append(i)

    # plt.plot(index, sigma_est_, label="Est_x")
    # plt.plot(index, sigma_true_, label="True_x")
    # plt.legend()
    # plt.show()