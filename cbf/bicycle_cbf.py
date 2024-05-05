import numpy as np
import sys
sys.path.append(r'D:\yangzhang\ma\safeRL-main\safeRL-main\parking')
from env.bicycle_model import Bicycle
from env.obs import Obstacle
from env.obs_car import Obstacle_car
import cvxopt

class Bicycle_CBF:
    def __init__(self) -> None:
        self.penalty = 1.2
        self.peng = 1.2
        self.delta = 0.1
    # alpha is Quadratic + Quadratic
    def cal_1(self, state, action_rl, lr, obs,obs_car, a_range, beta_range, v_range):
        # state: x y theta v
        P = cvxopt.matrix(np.eye(2))
        q = cvxopt.matrix(np.array([-2*action_rl[0], -2*action_rl[1]]).reshape(2,1))
        G = [[-1.0, 1.0, 0.0, 0.0, 1, -1], [0.0, 0.0, -1.0, 1.0, 0, 0]]
        h = [-a_range[0], a_range[1], -beta_range[0], beta_range[1], v_range[1] - state[3], state[3] - v_range[0]]
        
        bx_ = []
        dbx_ = []
        Lfbx_ = []
        Lf2bx_ = []
        LgLfbx_ = []
        for i in range(obs.num_obs):
            bx_.append((state[0]-obs.x[i])**2+(state[1]-obs.y[i])**2-(self.peng * obs.r[i])**2)
            dbx_.append(np.array([2*(state[0]-obs.x[i]), 2*(state[1]-obs.y[i]), 0, 0]).reshape(4,1))
            Lfbx_.append(2*(state[0]-obs.x[i])*state[3]*np.cos(state[2])+2*(state[1]-obs.y[i])*state[3]*np.sin(state[2]))
            Lf2bx_.append(2*state[3]*state[3])
            LgLfbx_.append(np.array([2*(state[0]-obs.x[i])*np.cos(state[2])+2*(state[1]-obs.y[i])*np.sin(state[2]),
                                     (-2*(state[0]-obs.x[i])*state[3]*np.sin(state[2])+2*(state[1]-obs.y[i])*state[3]*np.cos(state[2]))*state[3]/lr]))
            G[0].append(-(2*(state[0]-obs.x[i])*np.cos(state[2])+2*(state[1]-obs.y[i])*np.sin(state[2])))
            G[1].append(-(-2*(state[0]-obs.x[i])*state[3]*np.sin(state[2])+2*(state[1]-obs.y[i])*state[3]*np.cos(state[2]))*state[3]/lr)
            h.append(4*state[3]*state[3]+2*self.penalty*Lfbx_[-1]*bx_[-1]+self.penalty*Lfbx_[-1]*Lfbx_[-1]+
                     self.penalty*self.penalty*self.penalty*(bx_[-1]*bx_[-1])**2 + 2*self.penalty*self.penalty*bx_[-1]*bx_[-1]*Lfbx_[-1] + self.delta)
        G = cvxopt.matrix(G, (6+obs.num_obs,2))
        h = cvxopt.matrix(h)
        for i in range(obs_car.num_obs):
            car_x, car_y = obs_car.car(obs_car.x[i], obs_car.y[i], obs_car.theta[i])
            for j in range(len(car_x)):
                bx_.append((state[0]-car_x[j])**2+(state[1]-car_y[j])**2-(self.peng * obs_car.r[i])**2)
                dbx_.append(np.array([2*(state[0]-car_x[j]), 2*(state[1]-car_y[j]), 0, 0]).reshape(4,1))
                Lfbx_.append(2*(state[0]-car_x[j])*state[3]*np.cos(state[2])+2*(state[1]-car_y[j])*state[3]*np.sin(state[2]))
                Lf2bx_.append(2*state[3]*state[3])
                LgLfbx_.append(np.array([2*(state[0]-car_x[j])*np.cos(state[2])+2*(state[1]-car_y[j])*np.sin(state[2]),
                                        (-2*(state[0]-car_x[j])*state[3]*np.sin(state[2])+2*(state[1]-car_y[j])*state[3]*np.cos(state[2]))*state[3]/lr]))
                G[0].append(-(2*(state[0]-car_x[j])*np.cos(state[2])+2*(state[1]-car_y[j])*np.sin(state[2])))
                G[1].append(-(-2*(state[0]-car_x[j])*state[3]*np.sin(state[2])+2*(state[1]-car_y[j])*state[3]*np.cos(state[2]))*state[3]/lr)
                h.append(4*state[3]*state[3]+2*self.penalty*Lfbx_[-1]*bx_[-1]+self.penalty*Lfbx_[-1]*Lfbx_[-1]+
                        self.penalty*self.penalty*self.penalty*(bx_[-1]*bx_[-1])**2 + 2*self.penalty*self.penalty*bx_[-1]*bx_[-1]*Lfbx_[-1] + self.delta)
        G = cvxopt.matrix(G, (6+obs.num_obs+obs_car.num_obs,2))
        h = cvxopt.matrix(h)       
        return P,q,G,h
    
    # alpha is linear+linear
    def cal_2(self, state, action_rl, lr, obs,obs_car, a_range, beta_range, v_range):
        # state: x y theta v
        P = cvxopt.matrix(np.eye(2))
        q = cvxopt.matrix(np.array([-2*action_rl[0], -2*action_rl[1]]).reshape(2,1))
        G = [[-1.0, 1.0, 0.0, 0.0, 1, -1], [0.0, 0.0, -1.0, 1.0, 0, 0]]
        h = [-a_range[0], a_range[1], -beta_range[0], beta_range[1], v_range[1] - state[3], state[3] - v_range[0]]
        
        bx_ = []
        dbx_ = []
        Lfbx_ = []
        Lf2bx_ = []
        LgLfbx_ = []
        for i in range(obs.num_obs):
            bx_.append((state[0]-obs.x[i])**2+(state[1]-obs.y[i])**2-obs.r[i]**2)
            dbx_.append(np.array([2*(state[0]-obs.x[i]), 2*(state[1]-obs.y[i]), 0, 0]).reshape(4,1))
            Lfbx_.append(2*(state[0]-obs.x[i])*state[3]*np.cos(state[2])+2*(state[1]-obs.y[i])*state[3]*np.sin(state[2]))
            Lf2bx_.append(2*state[3]*state[3])
            LgLfbx_.append(np.array([2*(state[0]-obs.x[i])*np.cos(state[2])+2*(state[1]-obs.y[i])*np.sin(state[2]),
                                     (-2*(state[0]-obs.x[i])*state[3]*np.sin(state[2])+2*(state[1]-obs.y[i])*state[3]*np.cos(state[2]))*state[3]/lr]))
            G[0].append(-(2*(state[0]-obs.x[i])*np.cos(state[2])+2*(state[1]-obs.y[i])*np.sin(state[2])))
            G[1].append(-(-2*(state[0]-obs.x[i])*state[3]*np.sin(state[2])+2*(state[1]-obs.y[i])*state[3]*np.cos(state[2]))*state[3]/lr)
            h.append(2*state[3]*state[3]+2*self.penalty*Lfbx_[-1]+self.penalty*self.penalty*bx_[-1] + self.delta)
        G = cvxopt.matrix(G, (6+obs.num_obs,2))
        h = cvxopt.matrix(h)
        for i in range(obs_car.num_obs):
            car_x, car_y = obs_car.car(obs_car.x[i], obs_car.y[i], obs_car.theta[i])
            for j in range(len(car_x)):
                bx_.append((state[0]-car_x[j])**2+(state[1]-car_y[j])**2-obs_car.r[i]**2)
                dbx_.append(np.array([2*(state[0]-car_x[j]), 2*(state[1]-car_y[j]), 0, 0]).reshape(4,1))
                Lfbx_.append(2*(state[0]-car_x[j])*state[3]*np.cos(state[2])+2*(state[1]-car_y[j])*state[3]*np.sin(state[2]))
                Lf2bx_.append(2*state[3]*state[3])
                LgLfbx_.append(np.array([2*(state[0]-car_x[j])*np.cos(state[2])+2*(state[1]-car_y[j])*np.sin(state[2]),
                                        (-2*(state[0]-car_x[j])*state[3]*np.sin(state[2])+2*(state[1]-car_y[j])*state[3]*np.cos(state[2]))*state[3]/lr]))
                G[0].append(-(2*(state[0]-car_x[j])*np.cos(state[2])+2*(state[1]-car_y[j])*np.sin(state[2])))
                G[1].append(-(-2*(state[0]-car_x[j])*state[3]*np.sin(state[2])+2*(state[1]-car_y[j])*state[3]*np.cos(state[2]))*state[3]/lr)
                h.append(2*state[3]*state[3]+2*self.penalty*Lfbx_[-1]+self.penalty*self.penalty*bx_[-1] + self.delta)
        G = cvxopt.matrix(G, (6+obs.num_obs+obs_car.num_obs,2))
        h = cvxopt.matrix(h)
        return P,q,G,h

    # alpha is linear + square root
    def cal_3(self, state, action_rl, lr, obs,obs_car, a_range, beta_range, v_range):
        # state: x y theta v
        P = cvxopt.matrix(np.eye(2))
        q = cvxopt.matrix(np.array([-2*action_rl[0], -2*action_rl[1]]).reshape(2,1))
        G = [[-1.0, 1.0, 0.0, 0.0, 1, -1], [0.0, 0.0, -1.0, 1.0, 0, 0]]
        h = [-a_range[0], a_range[1], -beta_range[0], beta_range[1], v_range[1] - state[3], state[3] - v_range[0]]
        
        bx_ = []
        dbx_ = []
        Lfbx_ = []
        Lf2bx_ = []
        LgLfbx_ = []
        for i in range(obs.num_obs):
            bx_.append((state[0]-obs.x[i])**2+(state[1]-obs.y[i])**2-obs.r[i]**2)
            dbx_.append(np.array([2*(state[0]-obs.x[i]), 2*(state[1]-obs.y[i]), 0, 0]).reshape(4,1))
            Lfbx_.append(2*(state[0]-obs.x[i])*state[3]*np.cos(state[2])+2*(state[1]-obs.y[i])*state[3]*np.sin(state[2]))
            Lf2bx_.append(2*state[3]*state[3])
            LgLfbx_.append(np.array([2*(state[0]-obs.x[i])*np.cos(state[2])+2*(state[1]-obs.y[i])*np.sin(state[2]),
                                     (-2*(state[0]-obs.x[i])*state[3]*np.sin(state[2])+2*(state[1]-obs.y[i])*state[3]*np.cos(state[2]))*state[3]/lr]))
            G[0].append(-(2*(state[0]-obs.x[i])*np.cos(state[2])+2*(state[1]-obs.y[i])*np.sin(state[2])))
            G[1].append(-(-2*(state[0]-obs.x[i])*state[3]*np.sin(state[2])+2*(state[1]-obs.y[i])*state[3]*np.cos(state[2]))*state[3]/lr)
            h.append(2*state[3]*state[3]+self.penalty*Lfbx_[-1]+self.penalty*np.sqrt(Lfbx_[-1]+self.penalty*bx_[-1]) + self.delta)
        G = cvxopt.matrix(G, (6+obs.num_obs,2))
        h = cvxopt.matrix(h)
        for i in range(obs_car.num_obs):
            car_x, car_y = obs_car.car(obs_car.x[i], obs_car.y[i], obs_car.theta[i])
            for j in range(len(car_x)):
                bx_.append((state[0]-car_x[j])**2+(state[1]-car_y[j])**2-obs_car.r[i]**2)
                dbx_.append(np.array([2*(state[0]-car_x[j]), 2*(state[1]-car_y[j]), 0, 0]).reshape(4,1))
                Lfbx_.append(2*(state[0]-car_x[j])*state[3]*np.cos(state[2])+2*(state[1]-car_y[j])*state[3]*np.sin(state[2]))
                Lf2bx_.append(2*state[3]*state[3])
                LgLfbx_.append(np.array([2*(state[0]-car_x[j])*np.cos(state[2])+2*(state[1]-car_y[j])*np.sin(state[2]),
                                        (-2*(state[0]-car_x[j])*state[3]*np.sin(state[2])+2*(state[1]-car_y[j])*state[3]*np.cos(state[2]))*state[3]/lr]))
                G[0].append(-(2*(state[0]-car_x[j])*np.cos(state[2])+2*(state[1]-car_y[j])*np.sin(state[2])))
                G[1].append(-(-2*(state[0]-car_x[j])*state[3]*np.sin(state[2])+2*(state[1]-car_y[j])*state[3]*np.cos(state[2]))*state[3]/lr)
                h.append(2*state[3]*state[3]+self.penalty*Lfbx_[-1]+self.penalty*np.sqrt(Lfbx_[-1]+self.penalty*bx_[-1]) + self.delta)
        G = cvxopt.matrix(G, (6+obs.num_obs+obs_car.num_obs,2))
        h = cvxopt.matrix(h)
        return P,q,G,h

    def solve_qp(self, state, action_rl, lr, obs,obs_car, a_range, beta_range, v_range):
        cvxopt.solvers.options['show_progress'] = False
        try:
            P,q,G,h = self.cal_1(state, action_rl, lr, obs,obs_car,  a_range, beta_range, v_range)
            sol = cvxopt.solvers.qp(P,q,G,h)
            return [sol['x'][0], sol['x'][1]]
        except:
            try:
                P,q,G,h = self.cal_3(state, action_rl, lr, obs,obs_car,  a_range, beta_range, v_range)
                sol = cvxopt.solvers.qp(P,q,G,h)
                # print("Using alpha form with linear + square root")
                return [sol['x'][0], sol['x'][1]]
            except:
                try: 
                    P,q,G,h = self.cal_2(state, action_rl, lr, obs, obs_car, a_range, beta_range, v_range)
                    sol = cvxopt.solvers.qp(P,q,G,h)
                    # print("Using alpha form with linear + linear root")
                    return [sol['x'][0], sol['x'][1]]
                except:
                    # print("can not solve !!!")
                    return action_rl
        
