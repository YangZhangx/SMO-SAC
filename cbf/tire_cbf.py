import numpy as np
import sys
# set the path to the parking
sys.path.append(r'D:\xxxx\xxxx\parking')
from env.tire_model import TireModel
from env.obs import Obstacle
from env.obs_car import Obstacle_car
import cvxopt

class Tire_CBF:
    def __init__(self) -> None:
        self.penalty = 1

    # alpha is Quadratic + Quadratic
    def cal_1(self, env, obs,obs_car, action_rl):
        alpha_f = action_rl[0] - np.arctan((env.w * env.lf + env.vy) / (env.vx))
        alpha_r = np.arctan((env.w * env.lr - env.vy) / (env.vx))
        fx = [env.vx * np.cos(env.phi) - env.vy * np.sin(env.phi), 
              env.vx * np.sin(env.phi) + env.vy * np.cos(env.phi),
              env.w,
              env.vy * env.w,
              -env.vx * env.w + env.Cr * alpha_r / env.m - env.Cf * alpha_f / env.m,
              -env.Cr * alpha_r * env.lr / env.Iz - env.Cf * alpha_f * env.lf / env.Iz]
        gx = [[0,0], [0,0], [0,0], [env.Cf * alpha_f / env.m, 1 / env.m],
              [env.Cf / env.m, 0], 
              [env.lf * env.Cf / env.Iz, 0]]
        P = cvxopt.matrix(np.eye(2))
        q = cvxopt.matrix(np.array([-2*action_rl[0], -2*action_rl[1]]).reshape(2,1))
        G = [[-1.0, 1.0, 0.0, 0.0, env.Cf*alpha_f/env.m, -env.Cf*alpha_f/env.m, env.Cf/env.m, -env.Cf/env.m, env.lf*env.Cf/env.Iz, -env.lf*env.Cf/env.Iz], 
             [0.0, 0.0, -1.0, 1.0, 1/env.m, -1/env.m, 0, 0, 0, 0]]
        h = [-env.delta_range[0], env.delta_range[1], -env.F_range[0], env.F_range[1],
             env.v_range[1] - env.vx - env.vy*env.w, env.vx - env.v_range[0] + env.vy*env.w, 
             env.v_range[1] - env.vy + env.vx*env.w - env.Cr*alpha_r/env.m + env.Cf*alpha_f/env.m,
             -env.v_range[0] + env.vy - env.vx*env.w + env.Cr*alpha_r/env.m - env.Cf*alpha_f/env.m,
             env.Cr*alpha_r*env.lr/env.Iz + env.Cf*env.lf*alpha_f/env.Iz + env.w_range[1] - env.w,
             -env.Cr*alpha_r*env.lr/env.Iz - env.Cf*env.lf*alpha_f/env.Iz + env.w - env.w_range[0]]
        bx_ = []
        dbx_ = []
        Lfbx_ = []
        Lf2bx_ = []
        LgLfbx_ = []
        for i in range(obs.num_obs):
            bx_.append((env.x-obs.x[i])**2+(env.y-obs.y[i])**2-obs.r[i]**2)
            dbx_.append(np.array([2*(env.x-obs.x[i]), 2*(env.y-obs.y[i]), 0, 0, 0, 0]).reshape((6,1)))
            Lfbx_.append(2*(env.x-obs.x[i])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)) + 2*(env.y-obs.y[i])*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)))
            pas_LfBx_ = [2*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)), 2*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)),
                         2*(env.x-obs.x[i])*(-env.vx*np.sin(env.phi)-env.vy*np.cos(env.phi))+2*(env.y-obs.y[i])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)),
                         2*(env.x-obs.x[i])*np.cos(env.phi)+2*(env.y-obs.y[i])*np.sin(env.phi),
                         -2*(env.x-obs.x[i])*np.sin(env.phi)+2*(env.y-obs.y[i])*np.cos(env.phi),
                         0]
        
            Lf2bx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(fx).reshape((6,1)))[0][0])
            LgLfbx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(gx).reshape((6,2)))[0])

            G[0].append(-LgLfbx_[-1][0])
            G[1].append(-LgLfbx_[-1][1])

            h.append(2*Lf2bx_[-1] + 2*self.penalty*Lfbx_[-1]*bx_[-1] + self.penalty*Lfbx_[-1]*Lfbx_[-1]
                     + self.penalty*self.penalty*self.penalty*(bx_[-1]**2)**2 + self.penalty*(bx_[-1]**2))
        G = cvxopt.matrix(G, (10+obs.num_obs,2))
        h = cvxopt.matrix(h)
        for i in range(obs_car.num_obs):
            car_x, car_y = obs_car.car(obs_car.x[i], obs_car.y[i], obs_car.theta[i])
            for j in range(len(car_x)):
                bx_.append((env.x-car_x[j])**2+(env.y-car_y[j])**2-obs_car.r[i]**2)
                dbx_.append(np.array([2*(env.x-car_x[j]), 2*(env.y-car_y[j]), 0, 0, 0, 0]).reshape((6,1)))
                Lfbx_.append(2*(env.x-car_x[j])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)) + 2*(env.y-car_y[j])*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)))
                pas_LfBx_ = [2*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)), 2*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)),
                             2*(env.x-car_x[j])*(-env.vx*np.sin(env.phi)-env.vy*np.cos(env.phi))+2*(env.y-car_y[j])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)),
                             2*(env.x-car_x[j])*np.cos(env.phi)+2*(env.y-car_y[j])*np.sin(env.phi),
                             -2*(env.x-car_x[j])*np.sin(env.phi)+2*(env.y-car_y[j])*np.cos(env.phi),
                             0]
                Lf2bx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(fx).reshape((6,1)))[0][0])
                LgLfbx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(gx).reshape((6,2)))[0])

                G[0].append(-LgLfbx_[-1][0])
                G[1].append(-LgLfbx_[-1][1])

                h.append(2*Lf2bx_[-1] + 2*self.penalty*Lfbx_[-1]*bx_[-1] + self.penalty*Lfbx_[-1]*Lfbx_[-1]
                         + self.penalty*self.penalty*self.penalty*(bx_[-1]**2)**2 + self.penalty*(bx_[-1]**2))
        G = cvxopt.matrix(G, (10+obs.num_obs+obs_car.num_obs,2))
        h = cvxopt.matrix(h)
        return P,q,G,h
    
    # alpha is linear+linear
    def cal_2(self, env, obs,obs_car,  action_rl):
        alpha_f = action_rl[0] - np.arctan((env.w * env.lf + env.vy) / (env.vx))
        alpha_r = np.arctan((env.w * env.lr - env.vy) / (env.vx))
        fx = [env.vx * np.cos(env.phi) - env.vy * np.sin(env.phi), 
              env.vx * np.sin(env.phi) + env.vy * np.cos(env.phi),
              env.w,
              env.vy * env.w,
              -env.vx * env.w + env.Cr * alpha_r / env.m - env.Cf * alpha_f / env.m,
              -env.Cr * alpha_r * env.lr / env.Iz - env.Cf * alpha_f * env.lf / env.Iz]
        gx = [[0,0], [0,0], [0,0], [env.Cf * alpha_f / env.m, 1 / env.m],
              [env.Cf / env.m, 0], 
              [env.lf * env.Cf / env.Iz, 0]]
        P = cvxopt.matrix(np.eye(2))
        q = cvxopt.matrix(np.array([-2*action_rl[0], -2*action_rl[1]]).reshape(2,1))
        G = [[-1.0, 1.0, 0.0, 0.0, env.Cf*alpha_f/env.m, -env.Cf*alpha_f/env.m, env.Cf/env.m, -env.Cf/env.m, env.lf*env.Cf/env.Iz, -env.lf*env.Cf/env.Iz], 
             [0.0, 0.0, -1.0, 1.0, 1/env.m, -1/env.m, 0, 0, 0, 0]]
        h = [-env.delta_range[0], env.delta_range[1], -env.F_range[0], env.F_range[1],
             env.v_range[1] - env.vx - env.vy*env.w, env.vx - env.v_range[0] + env.vy*env.w, 
             env.v_range[1] - env.vy + env.vx*env.w - env.Cr*alpha_r/env.m + env.Cf*alpha_f/env.m,
             -env.v_range[0] + env.vy - env.vx*env.w + env.Cr*alpha_r/env.m - env.Cf*alpha_f/env.m,
             env.Cr*alpha_r*env.lr/env.Iz + env.Cf*env.lf*alpha_f/env.Iz + env.w_range[1] - env.w,
             -env.Cr*alpha_r*env.lr/env.Iz - env.Cf*env.lf*alpha_f/env.Iz + env.w - env.w_range[0]]
        bx_ = []
        dbx_ = []
        Lfbx_ = []
        Lf2bx_ = []
        LgLfbx_ = []
        for i in range(obs.num_obs):
            bx_.append((env.x-obs.x[i])**2+(env.y-obs.y[i])**2-obs.r[i]**2)
            dbx_.append(np.array([2*(env.x-obs.x[i]), 2*(env.y-obs.y[i]), 0, 0, 0, 0]).reshape((6,1)))
            Lfbx_.append(2*(env.x-obs.x[i])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)) + 2*(env.y-obs.y[i])*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)))
            pas_LfBx_ = [2*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)), 2*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)),
                         2*(env.x-obs.x[i])*(-env.vx*np.sin(env.phi)-env.vy*np.cos(env.phi))+2*(env.y-obs.y[i])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)),
                         2*(env.x-obs.x[i])*np.cos(env.phi)+2*(env.y-obs.y[i])*np.sin(env.phi),
                         -2*(env.x-obs.x[i])*np.sin(env.phi)+2*(env.y-obs.y[i])*np.cos(env.phi),
                         0]
            Lf2bx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(fx).reshape((6,1)))[0][0])
            LgLfbx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(gx).reshape((6,2)))[0])

            G[0].append(-LgLfbx_[-1][0])
            G[1].append(-LgLfbx_[-1][1])

            h.append(2*Lf2bx_[-1] + 2*self.penalty*Lfbx_[-1] + self.penalty*self.penalty*bx_[-1])
        G = cvxopt.matrix(G, (10+obs.num_obs,2))
        h = cvxopt.matrix(h)
        for i in range(obs_car.num_obs):
            car_x, car_y = obs_car.car(obs_car.x[i], obs_car.y[i], obs_car.theta[i])
            for j in range(len(car_x)):
                bx_.append((env.x-car_x[j])**2+(env.y-car_y[j])**2-obs_car.r[i]**2)
                dbx_.append(np.array([2*(env.x-car_x[j]), 2*(env.y-car_y[j]), 0, 0, 0, 0]).reshape((6,1)))
                Lfbx_.append(2*(env.x-car_x[j])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)) + 2*(env.y-car_y[j])*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)))
                pas_LfBx_ = [2*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)), 2*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)),
                             2*(env.x-car_x[j])*(-env.vx*np.sin(env.phi)-env.vy*np.cos(env.phi))+2*(env.y-car_y[j])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)),
                             2*(env.x-car_x[j])*np.cos(env.phi)+2*(env.y-car_y[j])*np.sin(env.phi),
                             -2*(env.x-car_x[j])*np.sin(env.phi)+2*(env.y-car_y[j])*np.cos(env.phi),
                             0]
                Lf2bx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(fx).reshape((6,1)))[0][0])
                LgLfbx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(gx).reshape((6,2)))[0])

                G[0].append(-LgLfbx_[-1][0])
                G[1].append(-LgLfbx_[-1][1])

                h.append(2*Lf2bx_[-1] + 2*self.penalty*Lfbx_[-1] + self.penalty*self.penalty*bx_[-1])
        G = cvxopt.matrix(G, (10+obs.num_obs+obs_car.num_obs,2))
        h = cvxopt.matrix(h)
        return P,q,G,h

    # alpha is linear + square root
    def cal_3(self, env, obs, obs_car, action_rl):
        alpha_f = action_rl[0] - np.arctan((env.w * env.lf + env.vy) / (env.vx))
        alpha_r = np.arctan((env.w * env.lr - env.vy) / (env.vx))
        fx = [env.vx * np.cos(env.phi) - env.vy * np.sin(env.phi), 
              env.vx * np.sin(env.phi) + env.vy * np.cos(env.phi),
              env.w,
              env.vy * env.w,
              -env.vx * env.w + env.Cr * alpha_r / env.m - env.Cf * alpha_f / env.m,
              -env.Cr * alpha_r * env.lr / env.Iz - env.Cf * alpha_f * env.lf / env.Iz]
        gx = [[0,0], [0,0], [0,0], [env.Cf * alpha_f / env.m, 1 / env.m],
              [env.Cf / env.m, 0], 
              [env.lf * env.Cf / env.Iz, 0]]
        P = cvxopt.matrix(np.eye(2))
        q = cvxopt.matrix(np.array([-2*action_rl[0], -2*action_rl[1]]).reshape(2,1))
        G = [[-1.0, 1.0, 0.0, 0.0, env.Cf*alpha_f/env.m, -env.Cf*alpha_f/env.m, env.Cf/env.m, -env.Cf/env.m, env.lf*env.Cf/env.Iz, -env.lf*env.Cf/env.Iz], 
             [0.0, 0.0, -1.0, 1.0, 1/env.m, -1/env.m, 0, 0, 0, 0]]
        h = [-env.delta_range[0], env.delta_range[1], -env.F_range[0], env.F_range[1],
             env.v_range[1] - env.vx - env.vy*env.w, env.vx - env.v_range[0] + env.vy*env.w, 
             env.v_range[1] - env.vy + env.vx*env.w - env.Cr*alpha_r/env.m + env.Cf*alpha_f/env.m,
             -env.v_range[0] + env.vy - env.vx*env.w + env.Cr*alpha_r/env.m - env.Cf*alpha_f/env.m,
             env.Cr*alpha_r*env.lr/env.Iz + env.Cf*env.lf*alpha_f/env.Iz + env.w_range[1] - env.w,
             -env.Cr*alpha_r*env.lr/env.Iz - env.Cf*env.lf*alpha_f/env.Iz + env.w - env.w_range[0]]
        bx_ = []
        dbx_ = []
        Lfbx_ = []
        Lf2bx_ = []
        LgLfbx_ = []
        for i in range(obs.num_obs):
            bx_.append((env.x-obs.x[i])**2+(env.y-obs.y[i])**2-obs.r[i]**2)
            dbx_.append(np.array([2*(env.x-obs.x[i]), 2*(env.y-obs.y[i]), 0, 0, 0, 0]).reshape((6,1)))
            Lfbx_.append(2*(env.x-obs.x[i])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)) + 2*(env.y-obs.y[i])*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)))
            pas_LfBx_ = [2*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)), 2*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)),
                         2*(env.x-obs.x[i])*(-env.vx*np.sin(env.phi)-env.vy*np.cos(env.phi))+2*(env.y-obs.y[i])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)),
                         2*(env.x-obs.x[i])*np.cos(env.phi)+2*(env.y-obs.y[i])*np.sin(env.phi),
                         -2*(env.x-obs.x[i])*np.sin(env.phi)+2*(env.y-obs.y[i])*np.cos(env.phi),
                         0]
            Lf2bx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(fx).reshape((6,1)))[0][0])
            LgLfbx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(gx).reshape((6,2)))[0])

            G[0].append(-LgLfbx_[-1][0])
            G[1].append(-LgLfbx_[-1][1])

            h.append(2*Lf2bx_[-1] + self.penalty*Lfbx_[-1] + self.penalty*np.sqrt(Lfbx_[-1] + self.penalty*bx_[-1]))
        G = cvxopt.matrix(G, (10+obs.num_obs,2))
        h = cvxopt.matrix(h)
        for i in range(obs_car.num_obs):
            car_x, car_y = obs_car.car(obs_car.x[i], obs_car.y[i], obs_car.theta[i])
            for j in range(len(car_x)):
                bx_.append((env.x-car_x[j])**2+(env.y-car_y[j])**2-obs_car.r[i]**2)
                dbx_.append(np.array([2*(env.x-car_x[j]), 2*(env.y-car_y[j]), 0, 0, 0, 0]).reshape((6,1)))
                Lfbx_.append(2*(env.x-car_x[j])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)) + 2*(env.y-car_y[j])*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)))
                pas_LfBx_ = [2*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)), 2*(env.vx*np.sin(env.phi)+env.vy*np.cos(env.phi)),
                             2*(env.x-car_x[j])*(-env.vx*np.sin(env.phi)-env.vy*np.cos(env.phi))+2*(env.y-car_y[j])*(env.vx*np.cos(env.phi)-env.vy*np.sin(env.phi)),
                             2*(env.x-car_x[j])*np.cos(env.phi)+2*(env.y-car_y[j])*np.sin(env.phi),
                             -2*(env.x-car_x[j])*np.sin(env.phi)+2*(env.y-car_y[j])*np.cos(env.phi),
                             0]
                Lf2bx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(fx).reshape((6,1)))[0][0])
                LgLfbx_.append(np.dot(np.array(pas_LfBx_).reshape((1,6)),np.array(gx).reshape((6,2)))[0])

                G[0].append(-LgLfbx_[-1][0])
                G[1].append(-LgLfbx_[-1][1])

                h.append(2*Lf2bx_[-1] + self.penalty*Lfbx_[-1] + self.penalty*np.sqrt(Lfbx_[-1] + self.penalty*bx_[-1]))
        G = cvxopt.matrix(G, (10+obs.num_obs+obs_car.num_obs,2))
        h = cvxopt.matrix(h)
        return P,q,G,h

    def solve_qp(self, env, obs, obs_car, action_rl):
        cvxopt.solvers.options['show_progress'] = False
        try:
            P,q,G,h = self.cal_1(env, obs, obs_car, action_rl)
            sol = cvxopt.solvers.qp(P,q,G,h)
            return [sol['x'][0], sol['x'][1]]
        except:
            try:
                P,q,G,h = self.cal_3(env, obs,obs_car,  action_rl)
                sol = cvxopt.solvers.qp(P,q,G,h)
                print("Using alpha form with linear + square root")
                return [sol['x'][0], sol['x'][1]]
            except:
                try: 
                    P,q,G,h = self.cal_2(env, obs, obs_car, action_rl)
                    sol = cvxopt.solvers.qp(P,q,G,h)
                    print("Using alpha form with linear + linear root")
                    return [sol['x'][0], sol['x'][1]]
                except:
                    print("can not solve !!!")
                    return action_rl

tire_model = TireModel()
obs = Obstacle()
obs_car = Obstacle_car()
tire_cBF = Tire_CBF()
print(tire_cBF.solve_qp(tire_model, obs, [-0.1, -500]))