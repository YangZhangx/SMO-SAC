
import sys
# set the path to the parking
sys.path.append(r'D:\xxxx\xxxx\parking')
import casadi as ca
from env.bicycle_model import Bicycle
from env.obs import Obstacle
from env.obs_car import Obstacle_car
import numpy as np

class SafeSet():
    def __init__(self, env, obs,obs_car) -> None:
        self.p = 2
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        states = ca.vertcat(x, y)
        states = ca.vertcat(states, theta)
        states = ca.vertcat(states, v)
        n_states = states.size()[0]

        a = ca.SX.sym('a')
        beta = ca.SX.sym('beta')
        controls = ca.vertcat(a, beta)
        n_controls = controls.size()[0]

        rhs_1 = v * ca.cos(theta + beta)
        rhs_2 = v * ca.sin(theta + beta)
        rhs_3 = v / env.lr * ca.sin(beta)
        rhs_4 = a

        rhs = ca.vertcat(rhs_1, rhs_2)
        rhs = ca.vertcat(rhs, rhs_3)
        rhs = ca.vertcat(rhs, rhs_4)

        f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

        # for safe control
        U = ca.SX.sym('U', n_controls)
        X = ca.SX.sym('X', n_states*2)
        P = ca.SX.sym('P', n_controls + n_states)

        obj = 0
        g = []
        g.append(X[:n_states] - P[:n_states])
        obj = obj + (U[0] - P[n_states]) * (U[0] - P[n_states]) + (U[1] - P[n_states+1]) * (U[1] - P[n_states+1])
        x_next = f(X[:n_states], U[:]) * env.T + P[:n_states]
        g.append(X[n_states:] - x_next)
        
        for i in range(obs.num_obs):
            g.append(ca.sqrt((X[n_states] - obs.x[i])**2 + (X[n_states+1] - obs.y[i])**2))

        # for i in range(obs_car.num_obs):
        #     car_x, car_y = obs_car.car(obs_car.x[i], obs_car.y[i], obs_car.theta[i])
        #     for j in range(len(car_x)):
        #         g.append(ca.sqrt((X[n_states] - car_x[j])**2 + (X[n_states+1] - car_y[j])**2))
        for i in range(obs_car.num_obs):
            g.append(ca.sqrt((X[n_states] - obs_car.x[i])**2 + (X[n_states+1] - obs_car.y[i])**2))
            
        
        opt_variables = ca.vertcat(U, X)
        nlp_prob = {'f': obj, 'x': opt_variables, 'p':P, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        for _ in range(n_states):
            self.lbg.append(0.0)
            self.ubg.append(0.0)
        for _ in range(n_states):
            self.lbg.append(0.0)
            self.ubg.append(0.0)
        for i in range(obs.num_obs):
            self.lbg.append(obs.r[i] * self.p)
            self.ubg.append(ca.inf)
        for i in range(obs_car.num_obs):
            self.lbg.append(obs_car.r[i] * self.p)
            self.ubg.append(ca.inf)
        self.lbx.append(env.a_range[0])
        self.ubx.append(env.a_range[1])
        self.lbx.append(env.beta_range[0])
        self.ubx.append(env.beta_range[1])
        for _ in range(n_states*2):
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)

env = Bicycle()
obs = Obstacle()
obs_car = Obstacle_car()
safe_act = SafeSet(env, obs, obs_car)
init_ = np.array([0.0, 0.0, 0, 0, np.pi/4, 0, 0, 0, np.pi/4, 0])
c_p = np.array([3.5, 3, np.pi/4, 0, 1, env.get_beta(np.pi/12)])
res = safe_act.solver(x0=init_, p=c_p, lbg=safe_act.lbg, lbx=safe_act.lbx, ubg=safe_act.ubg, ubx=safe_act.ubx)
estimated_opt = res['x'].full()
print(estimated_opt)
print(estimated_opt[0][0])
print(estimated_opt[1][0])