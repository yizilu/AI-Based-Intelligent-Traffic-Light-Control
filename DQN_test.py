import pandas as pd
import numpy as np
import seaborn as sns
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
from DQNController import DQNController_test
from plot import plot

cars_generated_steps = 1000
cool_down_steps = 100
max_steps = cars_generated_steps + cool_down_steps
seed = 0

network = '31grid'
demands = [200, 500, 1000]
episodes = 200
reward_types = ['Queue', 'Delay']

for demand in demands:
    for reward_type in reward_types:
        # Set net file and route file
        net_file = "source/road_network/{network}.net.xml".format(network = network)
        route_file = "source/road_network/{network}.rou.xml".format(network = network)

        # Set DQN load path
        dqn_load_path = "source/dqn_model_save/{network}/rewardtype_{reward_type}_{episodes}episodes_trained_model.keras"\
            .format(network = network, reward_type = reward_type, episodes = episodes)

        # Set the save path of result plot
        result_plot_save_path = 'source/result_plot/{network}/rewardtype_{reward_type}_demand{demand}_test.png'\
            .format(network = network, reward_type = reward_type, demand = demand)

        # Set the save path of traffic demand distribution plot 
        traffic_demand_plot_path = "source/result_plot/Traffic_Demand_distribution/{steps}_steps_{vehicle_numbers}vehicles.png"\
            .format(steps = cars_generated_steps, vehicle_numbers = demand)

        test_reward_save_path = "source/reward_save/{network}/rewardtype_{reward_type}_demand{demand}_test.csv"\
            .format(network = network, reward_type = reward_type, demand = demand)

        env = SumoEnvironment(
            net_file = net_file,
            route_file = route_file,
            use_gui = False,
            max_steps = max_steps
        )

        # Generate the route file
        RG = RouteGenerator(cars_generated_steps, demand, seed, \
                            route_file, env.start_end_points_x)
        RG.generate_routefile()
        # Plot the traffic demand
        plot.plot_demand_distribution(RG.timings, traffic_demand_plot_path)

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
        def main_dqn_test(controller, env, reward_type):
            reward_test = []
            state = env.reset()
            state = np.reshape(state, [1, controller.state_size])
            done = False
                
            while not done:
                action = controller.choose_action(state)
                next_state, reward, done = env.step(action, reward_type)
                next_state = np.reshape(next_state, [1, controller.state_size])
                state = next_state
                
                if done:
                    reward_test = env.get_total_reward()
            env.close()
            return reward_test



        # Parameters for dqn network
        state_size =  37*env._num_traffic_lights 
        action_size = 4 

        # Load DQN
        controller = DQNController_test(env, dqn_load_path, state_size, action_size)
        dqn_reward_test = main_dqn_test(controller, env, reward_type)


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

        # Plot the baseline reward and dqn reward
        Rewards = [-dqn_reward_test,-MaxPressure_reward,-MaxQueue_reward,-FixedTime_reward, -Random_reward, -Phase_reward]#Removed Phase
        data = pd.DataFrame()
        data['Controller'] = ['dqn','MaxPressure','MaxQueue','FixedTime', 'Random', 'Phase']#Removed Phase
        data['Rewards'] = Rewards
        sns.barplot(data=data, x="Controller", y="Rewards", palette ='tab10')
        plt.xlabel('Controllers')
        plt.ylabel('Reward ({reward_type})'.format(reward_type = reward_type))
        plt.legend()
        plt.savefig(result_plot_save_path)
        data.to_csv(test_reward_save_path)

