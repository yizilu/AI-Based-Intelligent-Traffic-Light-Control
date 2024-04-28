import pandas as pd
import numpy as np
from sumolib import checkBinary  
import matplotlib.pyplot as plt
from environment import SumoEnvironment
from RouteGenerator import RouteGenerator
from RandomController import RandomController
from FixedTimeController import FixedTimeController
from MaxQueueController import MaxQueueController
from MaxPressureController import MaxPressureController
from PhaseController import PhaseController
from DQNController import DQNController
from plot import plot

# Set the route generator parameters
cars_generated_steps = 1000
cool_down_steps = 100
max_steps = cars_generated_steps + cool_down_steps
seed = 0

# Set the network, episodes, reward_type and demand
network = '31grid'
episodes = 200
demand = 1000
reward_type = 'Queue'


# Set net file and route file
net_file = "source/road_network/{network}.net.xml".format(network = network)
route_file = "source/road_network/{network}.rou.xml".format(network = network)

# Set DQN save path
dqn_save_path = "source/dqn_model_save/{network}/{vehicle_numbers}vehicles_rewardtype_{reward_type}_{episodes}episodes_trained_model.keras"\
    .format(network = network, vehicle_numbers = demand, reward_type = reward_type, episodes = episodes)

# Set the save path of traffic demand distribution plot 
traffic_demand_plot_path = "source/result_plot/Traffic_Demand_distribution/{steps}_steps_{vehicle_numbers}vehicles.png"\
    .format(steps = cars_generated_steps, vehicle_numbers = demand)

# Set the save path of episodic reward 
reward_save_path = 'source/reward_save/{network}/{vehicle_numbers}vehicles_rewardtype_{reward_type}_{episodes}episodes.csv'\
    .format(network = network, vehicle_numbers = demand, reward_type= reward_type, episodes = episodes)

# Set the save path of result plot
result_plot_save_path = 'source/result_plot/{network}/{vehicle_numbers}vehicles_rewardtype_{reward_type}_{episodes}episodes.png'\
    .format(network = network, vehicle_numbers = demand, reward_type = reward_type, episodes = episodes)


# Set the environment
env = SumoEnvironment(
    net_file = net_file,
    route_file = route_file,
    use_gui = False,
    max_steps = max_steps
)


# Parameters for dqn network
state_size =  37*env._num_traffic_lights 
action_size = 4 

hidden_layers = 2
units = 256
learning_rate = 0.001
buffer_size = 10000
batch_size = 128
gamma = 0.95


# Generate the route file
RG = RouteGenerator(cars_generated_steps, demand, seed, \
                    route_file, env.start_end_points_x)
RG.generate_routefile()
# Plot the traffic demand
plot.plot_demand_distribution(RG.timings, traffic_demand_plot_path)
#

# Main function for non-reinforcement-learning controllers
def main_non_rl_controllers(controller, env, reward_type):
    env.total_reward = 0
    env.time = 0
    env._start_simulation()
    done = False
    while not done:
        reward, done = env.non_rl_step(reward_type)
        time = env.time
        TrafficLight_states = controller.get_actions(time)
        env.set_Traffic_Lights_States(env._traffic_lights, TrafficLight_states)
    total_reward = env.get_total_reward()
    env.close()
    return total_reward


# Main function for DQN
def main_dqn_single_agent(controller, env, episodes, dqn_save_path, reward_type):
    reward_episodic = []
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, controller.state_size])
        done = False
        
        while not done:
            action = controller.choose_action(state)
            next_state, reward, done = env.step(action, reward_type)
            next_state = np.reshape(next_state, [1, controller.state_size])
            controller.store_experience(state, action, reward, next_state, done)
            state = next_state
            
            
            if done:
                total_reward = env.get_total_reward()
                reward_episodic.append(total_reward)
                print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}")
                
        if len(controller.replay_buffer) > controller.batch_size:
            controller.replay()
        
    controller.save_dqn_model(dqn_save_path)
    env.close()

    return reward_episodic


# Get the baselines
controller1 = MaxPressureController(env)
MaxPressure_reward = main_non_rl_controllers(controller1, env, reward_type)
controller2 = RandomController(env)
Random_reward = main_non_rl_controllers(controller2, env, reward_type)
controller3 = MaxQueueController(env)
MaxQueue_reward = main_non_rl_controllers(controller3, env, reward_type)
controller4 = FixedTimeController(env)
FixedTime_reward = main_non_rl_controllers(controller4, env, reward_type)
controller5 = PhaseController(env)
Phase_reward = main_non_rl_controllers(controller5, env, reward_type)

# Set DQN controller
controller = DQNController(state_size, action_size, env, \
                        hidden_layers, units, learning_rate, buffer_size, batch_size, gamma)


# DQN total reward 
dqn_reward_episodic = main_dqn_single_agent(controller, env, episodes, dqn_save_path, reward_type)
df_dqn_reward_episodic = pd.DataFrame(dqn_reward_episodic,columns=['dqn_reward_episodic'])

# Save DQN Rewaed
df_dqn_reward_episodic.to_csv(reward_save_path)

# Plot the reward result
plot.plot_dqn_training_reward(
    Random_reward, 
    MaxPressure_reward, 
    MaxQueue_reward,
    FixedTime_reward,
    Phase_reward,
    dqn_reward_episodic,
    result_plot_save_path)