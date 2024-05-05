# SMO-SAC
Code for paper "Safety Enhancement for Reinforcement Learning with Sliding Mode Observer-based Control for Automatic Parking"

And master's thesis "Safety Enhancing for Reinforcement Learning with Sliding Mode Observer-based Control"


### SMO-SAC
Sliding mode observer-based soft Soft Actor-Critic

# Introduction
A method to improve the security of reinforcement learning with the reinforcement learning algorithm sac combined with a sliding mode observer. 
A specific parking environment is designed in the paper. The environment has parking spaces, roads, and cars as obstacles. The algorithm enables the control of vehicle speed, acceleration and steering angle. 
After continuous training, the vehicle can eventually be made to park accurately in the target parking space.

## Requirements
python 3.9
pythorch 2.0.1

## How To Run

1. Set the main file path
2. 'pip install -r requirements.txt'
3. set obstacles in 'obstacle.py'
4. set obstacle cars in 'obs_car.py'
5. set the start position and target parking place of ego car in 'bicycle_model.py'
6. adjust Reward functions in 'bicycle_model.py'
7. set the plots saving file in 'main.py'
8. run 'main.py'

#### Result

To get 5-car scenario, set obs_car with 5 cars
Similarly, to get 4-car or 3-car scenario


