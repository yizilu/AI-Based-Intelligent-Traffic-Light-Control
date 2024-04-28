import pandas as pd
import numpy as np
from sumolib import checkBinary  
import matplotlib.pyplot as plt
from environment import SumoEnvironment
from DQNController import DQNController
from RouteGenerator import RouteGenerator

cars_generated_steps = 1000
cool_down_steps = 100
max_steps = cars_generated_steps + cool_down_steps
n_cars_generated = 50
seed = 0

net_file = "road_network/21grid.net.xml"
route_file = "road_network/21grid.rou.xml"
traffic_demand_plot_path = "result_plot/Traffic_Demand_distribution.png"


env = SumoEnvironment(
    net_file = net_file,
    route_file = route_file,
    use_gui = False,
    max_steps = max_steps
)

# Generate the route file
RG = RouteGenerator(cars_generated_steps, n_cars_generated, seed, \
                    route_file, env.start_end_points_x)
RG.generate_routefile()
RG.plot_demand_distribution(traffic_demand_plot_path)

# Main function for DQN
def main_dqn(controller, env, episodes, dqn_save_path, reward_type):
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

    return reward_episodic


# Parameters for dqn network
state_size =  37*env._num_traffic_lights 
action_size = 4 

# Set reward type
reward_type = 'Delay'

# Set the search grid of hyperparameters
hidden_layers_list = [2,3,4]
units_list = [64,128,256]
buffer_size_list = [100, 1000, 10000]
batch_size_list = [4, 16, 64, 128]
learning_rate_list = [0.002, 0.001, 0.0005, 0.0001]
gamma_list = [0.9, 0.95, 0.99]

# Set default hyperparameters
default_hidden_layers = 2
default_units = 128
default_buffer_size = 1000
default_batch_size = 16
default_learning_rate = 0.001
default_gamma = 0.95



# hidden_layers tuning
hidden_layers_performance = []
for hidden_layers in hidden_layers_list:
    controller = DQNController(state_size, action_size, env, \
                               hidden_layers, default_units, default_learning_rate, default_buffer_size, default_batch_size, default_gamma)
    dqn_save_path = "dqn_model_save/Hyperparameter_Tuning/hidden_layer_{}.keras".format(hidden_layers)
    episodes = 100

    # DQN total reward 
    dqn_reward_episodic = main_dqn(controller, env, episodes, dqn_save_path, reward_type)
    df_dqn_reward_episodic = pd.DataFrame(dqn_reward_episodic,columns=['dqn_reward_episodic'])

    # plot the result and compare
    plt.plot(dqn_reward_episodic,color='orange',label='dqn_reward_episodic')
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.legend()
    plt.savefig('result_plot/Hyperparameter_Tuning/hidden_layer_{}.png'.format(hidden_layers))
    plt.close()

    hidden_layers_performance.append(df_dqn_reward_episodic.tail(20)['dqn_reward_episodic'].mean())
hidden_layers_index = hidden_layers_performance.index(max(hidden_layers_performance))
default_hidden_layers = hidden_layers_list[hidden_layers_index]



# units tuning
units_performance = []
for units in units_list:
    controller = DQNController(state_size, action_size, env, \
                               default_hidden_layers, units, default_learning_rate, default_buffer_size, default_batch_size, default_gamma)
    dqn_save_path = "dqn_model_save/Hyperparameter_Tuning/units_{}.keras".format(units)
    episodes = 100

    # DQN total reward 
    dqn_reward_episodic = main_dqn(controller, env, episodes, dqn_save_path, reward_type)
    df_dqn_reward_episodic = pd.DataFrame(dqn_reward_episodic,columns=['dqn_reward_episodic'])

    # plot the result and compare
    plt.plot(dqn_reward_episodic,color='orange',label='dqn_reward_episodic')
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.legend()
    plt.savefig('result_plot/Hyperparameter_Tuning/units_{}.png'.format(units))
    plt.close()

    units_performance.append(df_dqn_reward_episodic.tail(20)['dqn_reward_episodic'].mean())
units_index = units_performance.index(max(units_performance))
default_units = units_list[units_index]



# buffer_size tuning
buffer_size_performance = []
for buffer_size in buffer_size_list:
    controller = DQNController(state_size, action_size, env, \
                               default_hidden_layers, default_units, default_learning_rate, buffer_size, default_batch_size, default_gamma)
    dqn_save_path = "dqn_model_save/Hyperparameter_Tuning/buffer_size_{}.keras".format(buffer_size)
    episodes = 100

    # DQN total reward 
    dqn_reward_episodic = main_dqn(controller, env, episodes, dqn_save_path, reward_type)
    df_dqn_reward_episodic = pd.DataFrame(dqn_reward_episodic,columns=['dqn_reward_episodic'])

    # plot the result and compare
    plt.plot(dqn_reward_episodic,color='orange',label='dqn_reward_episodic')
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.legend()
    plt.savefig('result_plot/Hyperparameter_Tuning/buffer_size_{}.png'.format(buffer_size))
    plt.close()

    buffer_size_performance.append(df_dqn_reward_episodic.tail(20)['dqn_reward_episodic'].mean())
buffer_size_index = buffer_size_performance.index(max(buffer_size_performance))
default_buffer_size = buffer_size_list[buffer_size_index]



# batch_size tuning
batch_size_performance = []
for batch_size in batch_size_list:
    # 1
    controller = DQNController(state_size, action_size, env, \
                               default_hidden_layers, default_units, default_learning_rate, default_buffer_size, batch_size, default_gamma)
    # 2
    dqn_save_path = "dqn_model_save/Hyperparameter_Tuning/batch_size_{}.keras".format(batch_size)
    episodes = 100

    # DQN total reward 
    dqn_reward_episodic = main_dqn(controller, env, episodes, dqn_save_path, reward_type)
    df_dqn_reward_episodic = pd.DataFrame(dqn_reward_episodic,columns=['dqn_reward_episodic'])

    # plot the result and compare
    plt.plot(dqn_reward_episodic,color='orange',label='dqn_reward_episodic')
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.legend()
    # 3
    plt.savefig('result_plot/Hyperparameter_Tuning/batch_size_{}.png'.format(batch_size))
    plt.close()

    #4
    batch_size_performance.append(df_dqn_reward_episodic.tail(20)['dqn_reward_episodic'].mean())
#5
batch_size_index = batch_size_performance.index(max(batch_size_performance))
#6
default_batch_size = batch_size_list[batch_size_index]



# learning_rate tuning
learning_rate_performance = []
for learning_rate in learning_rate_list:
    # 1
    controller = DQNController(state_size, action_size, env, \
                               default_hidden_layers, default_units, learning_rate, default_buffer_size, default_batch_size, default_gamma)
    # 2
    dqn_save_path = "dqn_model_save/Hyperparameter_Tuning/learning_rate_{}.keras".format(learning_rate)
    episodes = 100

    # DQN total reward 
    dqn_reward_episodic = main_dqn(controller, env, episodes, dqn_save_path, reward_type)
    df_dqn_reward_episodic = pd.DataFrame(dqn_reward_episodic,columns=['dqn_reward_episodic'])

    # plot the result and compare
    plt.plot(dqn_reward_episodic,color='orange',label='dqn_reward_episodic')
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.legend()
    # 3
    plt.savefig('result_plot/Hyperparameter_Tuning/learning_rate_{}.png'.format(learning_rate))
    plt.close()

    #4
    learning_rate_performance.append(df_dqn_reward_episodic.tail(20)['dqn_reward_episodic'].mean())
#5
learning_rate_index = learning_rate_performance.index(max(learning_rate_performance))
#6
default_learning_rate = learning_rate_list[learning_rate_index]



# gamma tuning
gamma_performance = []
for gamma in gamma_list:
    # 1
    controller = DQNController(state_size, action_size, env, \
                               default_hidden_layers, default_units, default_learning_rate, default_buffer_size, default_batch_size, gamma)
    # 2
    dqn_save_path = "dqn_model_save/Hyperparameter_Tuning/gamma_{}.keras".format(gamma)
    episodes = 100

    # DQN total reward 
    dqn_reward_episodic = main_dqn(controller, env, episodes, dqn_save_path, reward_type)
    df_dqn_reward_episodic = pd.DataFrame(dqn_reward_episodic,columns=['dqn_reward_episodic'])

    # plot the result and compare
    plt.plot(dqn_reward_episodic,color='orange',label='dqn_reward_episodic')
    plt.xlabel('episodes')
    plt.ylabel('total reward per episode')
    plt.legend()
    # 3
    plt.savefig('result_plot/Hyperparameter_Tuning/gamma_{}.png'.format(gamma))
    plt.close()

    #4
    gamma_performance.append(df_dqn_reward_episodic.tail(20)['dqn_reward_episodic'].mean())
#5
gamma_index = gamma_performance.index(max(gamma_performance))
#6
default_gamma = gamma_list[gamma_index]


Hyperparameter_Tuning_result = pd.DataFrame(
    {'hidden_layers' : default_hidden_layers,
    'units' : default_units,
    'buffer_size' : default_buffer_size,
    'batch_size' : default_batch_size,
    'learning_rate' : default_learning_rate,
    'gamma' : default_gamma}, index = 0
)
Hyperparameter_Tuning_result.to_csv('Hyperparameter_Tuning_result.csv')