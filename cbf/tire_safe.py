
import sys
# set the path to the parking
sys.path.append(r'D:\xxxx\xxxx\parking')
import casadi as ca
from env.tire_model import TireModel
from env.obs import Obstacle
import numpy as np

class SafeSet():
    def __init__(self, env, obs) -> None:
        self.p = 1.5
        self.N = 5
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('phi')
        vx = ca.SX.sym('vx')
        vy = ca.SX.sym('vy')
        w = ca.SX.sym('w')
        states = ca.vertcat(x, y)
        states = ca.vertcat(states, phi)
        states = ca.vertcat(states, vx)
        states = ca.vertcat(states, vy)
        states = ca.vertcat(states, w)
        n_states = states.size()[0]

        delta = ca.SX.sym('delta')
        Fd = ca.SX.sym('Fd')
        controls = ca.vertcat(delta, Fd)
        n_controls = controls.size()[0]

        rhs_1 = vx * ca.cos(phi) - vy * ca.sin(phi)
        rhs_2 = vx * ca.sin(phi) + vy * ca.cos(phi)
        rhs_3 = w
        rhs_4 = (Fd - env.Cf * (delta - ca.arctan((w * env.lf + vy) / vx)) * ca.sin(delta) + env.m * vy * w) / env.m
        rhs_5 = (env.Cr * ca.arctan((w * env.lr - vy) / vx) + env.Cf * (delta - ca.arctan((w * env.lf + vy) / vx)) * ca.cos(delta) - env.m * vx * w) / env.m
        rhs_6 = (env.lf * env.Cf * (delta - ca.arctan((w * env.lf + vy) / vx)) * ca.cos(delta) - env.lr * env.Cr * ca.arctan((w * env.lr - vy) / vx)) / env.Iz
        rhs = ca.vertcat(rhs_1, rhs_2)
        rhs = ca.vertcat(rhs, rhs_3)
        rhs = ca.vertcat(rhs, rhs_4)
        rhs = ca.vertcat(rhs, rhs_5)
        rhs = ca.vertcat(rhs, rhs_6)

        f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

        # for safe control
        U = ca.SX.sym('U', n_controls*(self.N-1))
        X = ca.SX.sym('X', n_states*self.N)
        P = ca.SX.sym('P', n_controls + n_states)

        obj = 0
        g = []
        g.append(X[:n_states] - P[:n_states])
        obj = obj + (U[0] - P[n_states]) * (U[0] - P[n_states]) + (U[1] - P[n_states+1]) * (U[1] - P[n_states+1])
        obj = obj + ((X[6] - 60) ** 2 + (X[7] - 50) ** 2) * 0.0001
        for i in range(self.N-1):
            x_next = f(X[i*n_states:(i+1)*n_states], U[i*n_controls:(i+1)*n_controls]) * env.T + X[i*n_states:(i+1)*n_states]
            g.append(X[(i+1)*n_states:(i+2)*n_states] - x_next)
        # x_next2 = f(X[n_states:2*n_states], U[n_controls:2*n_controls]) * env.T + X[n_states:2*n_states]
        # g.append(X[2*n_states:3*n_states] - x_next2)
        # x_next3 = f(X[2*n_states:3*n_states], U[2*n_controls:]) * env.T + X[2*n_states:3*n_states]
        # g.append(X[3*n_states:] - x_next3)

        for j in range(self.N-1):
            for i in range(obs.num_obs):
                g.append(ca.sqrt((X[(j+1)*n_states] - obs.x[i])**2 + (X[(j+1)*n_states+1] - obs.y[i])**2))
        # for i in range(obs.num_obs):
        #     g.append(ca.sqrt((X[2*n_states] - obs.x[i])**2 + (X[2*n_states+1] - obs.y[i])**2))
        # for i in range(obs.num_obs):
        #     g.append(ca.sqrt((X[3*n_states] - obs.x[i])**2 + (X[3*n_states+1] - obs.y[i])**2))

        opt_variables = ca.vertcat(U, X)
        nlp_prob = {'f': obj, 'x': opt_variables, 'p':P, 'g':ca.vertcat(*g)}
        opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':0, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        for _ in range(self.N):
            for _ in range(n_states):
                self.lbg.append(0.0)
                self.ubg.append(0.0)
        for j in range(self.N-1):
            for i in range(obs.num_obs):
                self.lbg.append(obs.r[i]*self.p)
                self.ubg.append(ca.inf)
        for i in range(self.N-1):
            self.lbx.append(env.delta_range[0])
            self.ubx.append(env.delta_range[1])
            self.lbx.append(env.F_range[0])
            self.ubx.append(env.F_range[1])
        for _ in range(self.N):
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)
            self.lbx.append(-ca.inf)
            self.ubx.append(ca.inf)
            self.lbx.append(env.v_range[0])
            self.ubx.append(env.v_range[1])
            self.lbx.append(env.v_range[0])
            self.ubx.append(env.v_range[1])
            self.lbx.append(env.w_range[0])
            self.ubx.append(env.w_range[1])

env = TireModel()
obs = Obstacle()
safe_act = SafeSet(env, obs)
init_ = np.array([0.0, 0.0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0, 0])
c_p = np.array([0, 0, 0, 0.1, 0, 0, 0.1, 2000])
res = safe_act.solver(x0=init_, p=c_p, lbg=safe_act.lbg, lbx=safe_act.lbx, ubg=safe_act.ubg, ubx=safe_act.ubx)
estimated_opt = res['x'].full()
print(estimated_opt[0][0])
print(estimated_opt[1][0])