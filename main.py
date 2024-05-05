import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1,2'

import sys
# set the path to the current directory
sys.path.append(r'D:\xxxx\xxxx\parking')
from sac.SAC import SAC
from env.obs import Obstacle
from env.obs_car import Obstacle_car
from env.bicycle_model import Bicycle
from env.smo import SMO
from cbf.bicycle_cbf import Bicycle_CBF
from cbf.bicycle_safe import SafeSet #NLP safe set
import matplotlib.pyplot as plt
import torch
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')

torch.cuda.set_device (0)

env = Bicycle()
tau = 0.01
gamma = 0.99
q_lr = 2e-3
value_lr = 3e-3
policy_lr = 3e-3
buffer_maxlen = 10000000

Episode = 1000
batch_size = 64

agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr)
obs = Obstacle()
obs_car = Obstacle_car()
cbf = Bicycle_CBF()
safe_act = SafeSet(env, obs, obs_car)
Return = []
return_ = 0

# set the save path for log and plot
train_writer = SummaryWriter(log_dir='logs/test_parking_test4_SAC-PC_smo/')

# plt_save = 'D:/yangzhang/ma/result/SSAC/parking/5cars3_safe_NLP_dis/'


for episode in range(Episode):
    score = 0
    env.reset() 
    state = env.get_state()
    trajectory_x = [state[0]]
    trajectory_y = [state[1]]
    for i in range(400):
        action = agent.get_action(state)
        action[0] = action[0] * env.a_range[1]
        action[1] = env.get_beta(action[1] * env.delta_range[1])

        # two types of safety constraints CBF-QP or PC-NLP

        # CBF-QP safety constraints 
        # action_safe = cbf.solve_qp(state, action, env.lr, obs, obs_car,env.a_range, env.beta_range, env.v_range)
        
        # PC-NLP safety constraints 
        # safe_act = SafeSet(env, obs,obs_car)
        init_ = np.hstack((np.array(action), np.array(state)))
        init_ = np.hstack((init_, np.array(state)))
        c_p = np.hstack((np.array(state), np.array(action)))
        res = safe_act.solver(x0=init_, p=c_p, lbg=safe_act.lbg, lbx=safe_act.lbx, ubg=safe_act.ubg, ubx=safe_act.ubx)
        estimated_opt = res['x'].full()
        action_safe = [estimated_opt[0][0], np.arctan(np.tan(estimated_opt[1][0])*env.lr/(env.lr+env.lf))]

        env.step(action_safe,i)
        next_state = env.get_state()
        reward, done = env.get_reward()
        done_mask = 0.0 if done else 1.0
        agent.buffer.push((state, action_safe, reward, next_state, done_mask))
        state = next_state
        trajectory_x.append(state[0])
        trajectory_y.append(state[1])
        score += reward
        if np.sqrt((env.target[0] - env.x) ** 2 + (env.target[1] - env.y) ** 2) < 0.2:
            # env.theta = 0
            state[2] = 0
            done = True
        if env.x < 0 or env.x > 28.5 or env.y < 0 or env.y > 30:
            done = True
        if done:
            break
        if agent.buffer.buffer_len() > 128:
            agent.update(batch_size)
        


    if episode % 1 == 0:

        # plot figure
        plt.figure(figsize=(5, 5),)
        plt.axis('equal')

        # plot trajectory
        plt.plot(trajectory_x, trajectory_y)

        # plot target
        # plt.plot([env.target[0]], [env.target[1]], '*')
        target_x = [env.target[0]-1.2,env.target[0]-1.2,env.target[0]+1.2,env.target[0]+1.2,env.target[0]-1.2]
        target_y = [env.target[1]-0.8,env.target[1]+0.8,env.target[1]+0.8,env.target[1]-0.8,env.target[1]-0.8]
        plt.plot(target_x, target_y, color='y', linewidth=2.0)
        
        # plt.gcf().gca().add_artist(target)

        # plot start
        plt.plot(env.start_x, env.start_y, '*')

        # plot wall
        # wall_x = [0, 0, 30, 30, 0]
        # wall_y = [0, 30, 30, 0, 0]
        # plt.plot(wall_x, wall_y, color='k', linewidth=2.0)

        #plot obstacle car
        for i in range(obs_car.num_obs):
            car_x, car_y = obs_car.car(obs_car.x[i], obs_car.y[i], obs_car.theta[i])
            plt.plot(car_x, car_y, color='r', linewidth=2.0)
            
        # plot mid roads
        # road_x = [0, 30]
        # road_y = [15, 15]
        # plt.plot(road_x, road_y, color='b', linewidth=1.0, linestyle='--')
        # road_x = [15, 15]
        # road_y = [0,30]
        # plt.plot(road_x, road_y, color='b', linewidth=1.0, linestyle='--')

        # plot road
        road_x = [0, 12.5]
        road_y = [12.5, 12.5]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)
        road_x = [12.5, 12.5]
        road_y = [0, 12.5]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)

        road_x = [0, 12.5]
        road_y = [17.5, 17.5]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)
        road_x = [12.5, 12.5]
        road_y = [17.5, 30]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)

        road_x = [17.5, 17.5]
        road_y = [17.5, 30]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)
        road_x = [17.5,21 ]
        road_y = [17.5, 17.5]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)
        road_x = [26,30 ]
        road_y = [17.5, 17.5]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)

        road_x = [17.5,17.5 ]
        road_y = [0, 12.5]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)
        road_x = [17.5,30 ]
        road_y = [12.5, 12.5]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)

        # plot road branch
        road_x = [21, 21]
        road_y = [17.5, 30]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)
        road_x = [26, 26]
        road_y = [17.5, 30]
        plt.plot(road_x, road_y, color='b', linewidth=1.0)
        # road_x = [23.5, 23.5]
        # road_y = [15, 30]
        # plt.plot(road_x, road_y, color='b', linewidth=1.0, linestyle='--')

        # plot parking left
        parking_x = [17.5,21]
        parking_y = [17.75,17.75]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [17.5,21]
        parking_y = [20.25,20.25]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [17.5,21]
        parking_y = [20.75,20.75]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [17.5,21]
        parking_y = [23.25,23.25]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [17.5,21]
        parking_y = [23.75,23.75]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [17.5,21]
        parking_y = [26.25,26.25]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [17.5,21]
        parking_y = [26.75,26.75]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [17.5,21]
        parking_y = [29.25,29.25]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')

        # plot parking right
        parking_x = [26,29.5]
        parking_y = [17.75,17.75]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [26,29.5]
        parking_y = [20.25,20.25 ]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [26,29.5]
        parking_y = [20.75,20.75]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [26,29.5]
        parking_y = [23.25,23.25]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [26,29.5]
        parking_y = [23.75,23.75]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [26,29.5]
        parking_y = [26.25,26.25]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [26,29.5]
        parking_y = [26.75,26.75]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')
        parking_x = [26,29.5]
        parking_y = [29.25,29.25]
        plt.plot(parking_x, parking_y, color='k', linewidth=0.5, linestyle='--')

        #plot car
        x = trajectory_x[-1]
        y = trajectory_y[-1]
        theta = state[2]
        beta = state[3]

        lr = env.lr
        lf = env.lf
        w = 0.5
        
        # x_circle = [x + lr*np.cos(theta), x , x -  lr*np.cos(theta)]
        # y_circle = [y + lr*np.sin(theta), y , y -  lr*np.sin(theta)]
        r_circle = 1
        car_circle = plt.Circle((x + 0.6*np.cos(theta),y + 0.6*np.sin(theta)), r_circle, color='g', fill=False)
        plt.gcf().gca().add_artist(car_circle)
        car_circle = plt.Circle((x, y), r_circle, color='g', fill=False)
        plt.gcf().gca().add_artist(car_circle)
        car_circle = plt.Circle(( x - 0.6*np.cos(theta),  y -0.6*np.sin(theta)), r_circle, color='g', fill=False)
        plt.gcf().gca().add_artist(car_circle)

        car_x = [x + w*lr*np.cos(theta) - w*lf*np.sin(theta), x + w*lr*np.cos(theta) + w*lf*np.sin(theta), x - w*lr*np.cos(theta) + w*lf*np.sin(theta), x -  w*lr*np.cos(theta) - w*lf*np.sin(theta),x + w*lr*np.cos(theta) - w*lf*np.sin(theta)]
        car_y = [y + w*lr*np.sin(theta) + w*lf*np.cos(theta), y + w*lr*np.sin(theta) - w*lf*np.cos(theta), y - w*lr*np.sin(theta) - w*lf*np.cos(theta), y -  w*lr*np.sin(theta) + w*lf*np.cos(theta),y + w*lr*np.sin(theta) + w*lf*np.cos(theta)]
        
        # car_x = [x + lr*np.cos(beta) - w*np.sin(beta), x + lr*np.cos(beta) + w*np.sin(beta), x - lf*np.cos(beta) + w*np.sin(beta), x - lf*np.cos(beta) - w*np.sin(beta),x + lr*np.cos(beta) - w*np.sin(beta)]
        # car_y = [y + lr*np.sin(beta) + w*np.cos(beta), y + lr*np.sin(beta) - w*np.cos(beta), y - lf*np.sin(beta) - w*np.cos(beta), y - lf*np.sin(beta) + w*np.cos(beta),y + lr*np.sin(beta) + w*np.cos(beta)]
        # car_x = [x + lr*np.cos(beta) , x + lr*np.cos(beta) , x - lf*np.cos(beta) , x - lf*np.cos(beta) ,x + lr*np.cos(beta) ]
        # car_y = [y + lr*np.sin(beta) , y + lr*np.sin(beta) , y - lf*np.sin(beta) , y - lf*np.sin(beta) ,y + lr*np.sin(beta) ]
        # car_x = [x + lr*np.cos(theta) , x + lr*np.cos(theta) , x - lf*np.cos(theta) , x - lf*np.cos(theta) ,x + lr*np.cos(theta) ]
        # car_y = [y + lr*np.sin(theta) , y + lr*np.sin(theta) , y - lf*np.sin(theta) , y - lf*np.sin(theta) ,y + lr*np.sin(theta) ]
        plt.plot(car_x, car_y, color='b', linewidth=2.0)

        #plot obstacles cars     
        for i in range(obs.num_obs):
            circle = plt.Circle((obs.x[i], obs.y[i]), obs.r[i], color='r', fill=True)
            plt.gcf().gca().add_artist(circle)
        plt.title("Episode="+ str(episode))

        # test complex example
        # plt.savefig(plt_save+str(episode)+'.png')
        plt.savefig('D:/xxxx/test_parking/test_SAC-PC_smo/'+str(episode)+'.png')

        plt.cla()
        plt.close('all')

        # # plot disturbance
        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(env.time, env.x_true[:, 0], label='True State (x)')
        # plt.xlabel('Time')
        # plt.ylabel('State')
        # plt.subplot(2, 1, 2)
        # plt.plot(env.time, env.x_true[:, 0], label='True State (x)')
        # plt.plot(env.time, env.x_hat[:, 0], label='Estimated State (x)')
        # plt.xlabel('Time')
        # plt.ylabel('State')
        # plt.legend()
        # # plt.title("Episode="+ str(episode))
        # # plt.savefig(plt_save+'disturbance_x'+str(episode)+'.png')
        # plt.savefig('D:/yangzhang/ma/result/SSAC/test_parking/5cars_SAC-CBF/'+'disturbance_x'+str(episode)+'.png')
        # # plt.savefig('D:/yangzhang/ma/result/SSAC/test_complex/test2_SAC-CBF/'+str(episode)+'.png')
        # # plt.savefig('D:/yangzhang/ma/result/SSAC/test_complex/test2-SAC-SMO-NLP0.1/'+str(episode)+'.png')
        # plt.cla()
        # plt.close('all')

        # plt.figure()
        # plt.subplot(2, 1, 1)
        # plt.plot(env.time, env.x_true[:, 1], label='True State (y)')
        # plt.subplot(2, 1, 2)
        # plt.plot(env.time, env.x_true[:, 1], label='True State (y)')
        # plt.plot(env.time, env.x_hat[:, 1], label='Estimated State (y)')
        # plt.xlabel('Time')
        # plt.ylabel('State')
        # plt.legend()
        # #plt.show()
        # # plt.title("Episode="+ str(episode))
        # # plt.savefig(plt_save+'disturbance_y'+str(episode)+'.png')
        # plt.savefig('D:/yangzhang/ma/result/SSAC/test_parking/5cars_SAC-CBF/'+'disturbance_y'+str(episode)+'.png')
        # # plt.savefig('D:/yangzhang/ma/result/SSAC/test_complex/test2_SAC-CBF/'+str(episode)+'.png')
        # # plt.savefig('D:/yangzhang/ma/result/SSAC/test_complex/test2-SAC-SMO-NLP0.1/'+str(episode)+'.png')
        # plt.cla()
        # plt.close('all')

    print("episode:{}, Return:{},Reward:{}, buffer_capacity:{}".format(episode, score,reward, agent.buffer.buffer_len()))
    Return.append(score)

    train_writer.add_scalar('Return', score, episode)
    train_writer.add_scalar('buffer_capacity', agent.buffer.buffer_len(), episode)
    train_writer.add_scalar('reward', reward, episode)
    train_writer.add_scalar('action', action[0], episode)
    train_writer.add_scalar('action', action[1], episode)
    train_writer.add_scalar('disturbance',  np.average(env.disturbance), episode)
    train_writer.add_scalar('smo', np.average(env.smo), episode)

    



    # logdir = 'D:/yangzhang/ma/result/SSAC/test_complex/test1_safe_NLP/'
    # logdir = 'D:/yangzhang/ma/result/SSAC/test_simple/test2_SAC-CBF/'
    # logdir = 'D:/yangzhang/ma/result/SSAC/test_simple/test2-SAC-SMO-NLP0.1/'
    # logdir = 'D:/yangzhang/ma/result/SSAC/test_complex/test2_SAC-CBF/'
    # logdir = 'D:/yangzhang/ma/result/SSAC/test_complex/test2-SAC-SMO-NLP0.1/'
    # logdir = 'D:/yangzhang/ma/result/SSAC/test_complex/test2-SAC-SMO-NLP0.1/'
    # logdir = 'D:/yangzhang/ma/result/SSAC/test_complex/test2-SAC-SMO-NLP0.1/'
    
    # print("car length:", lr, lf)
    return_ += score
    score = 0
 