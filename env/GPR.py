import sys
# set the path to the bicycle model
sys.path.append(r'D:\xxxx\xxxx\parking')
from env.bicycle_model import Bicycle
import GPy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPR1(): 
    def __init__(self, env) -> None:
        self.K = GPy.kern.Matern32(6)
        self.icm = GPy.util.multioutput.ICM(input_dim=1,num_outputs=4,kernel=self.K)
        self.kern = GPy.kern.RBF(2) ** GPy.kern.Coregionalize(input_dim=1,output_dim=4, rank=1)

    def train(self, train_in, train_out):
        self.lcm1 = GPy.util.multioutput.ICM(input_dim=6,num_outputs=4, kernel=self.kern)
        self.m1 = GPy.models.GPRegression(train_in, train_out, self.lcm1)
        self.m = GPy.models.GPCoregionalizedRegression(train_in, train_out, kernel=self.icm)
        self.m['.*Mat32.var'].constrain_fixed(1) #For this kernel, B.kappa encodes the variance now.
        # self.m.optimize()
        self.m1.optimize()

    def predict(self, test_in):
        out_ = self.m.predict(test_in)
        print(out_)

class GPR():
    def __init__(self):
        self.kernel = C(0.1, (0.001, 1e4)) * RBF(0.5, (1e-4, 1e4))
        self.reg = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=1e-10)
    
    def train(self, train_in, train_out):
        self.reg.fit(train_in, train_out)
        
    def predict(self, test_in):
        output, err = self.reg.predict(test_in, return_std=True)
        return output
    
if __name__ == "__main__":
    env = Bicycle()
    GP_ = GPR()
    train_in = []
    train_out = []
    for i in range(1000):
        state = env.get_state()
        action = [np.random.random()*0.1, np.random.random()*0.1]
        train_in.append(state+action)
        env.step(action)
        temp = env.get_state()
        train_out.append(env.get_state())
    GP_.train(np.array(train_in), np.array(train_out))
    
    sigma_true_x = []
    sigma_est_x = []
    index = []
    for i in range(100):
        state = env.get_state()
        action = [np.random.random()*0.1, np.random.random()*0.1]
        test_in = state + action
        env.step(action)
        sigma_true = env.uncertainty()
        sigma_true_x.append(sigma_true[0])
        test_in = np.array(test_in)
        temp = GP_.predict([test_in])[0]
        sigma_est_ = env.get_state()[0] - temp[0]
        sigma_est_x.append(sigma_est_)
        index.append(i)

    plt.plot(index, sigma_est_x, label="Est_x")
    plt.plot(index, sigma_true_x, label="True_x")
    plt.legend()
    plt.show()