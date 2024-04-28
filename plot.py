import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class plot():
    def __init__(self) -> None:
        pass

    def plot_demand_distribution(
            timings, 
            traffic_demand_plot_path):
        
        g=sns.displot(x=timings, kde=True)
        g.set_axis_labels("time/second", "vehicle numbers")
        g.set_titles("Traffic Demand Distribution")
        plt.savefig(traffic_demand_plot_path)
        plt.close()

    def plot_dqn_training_reward(
            Random_reward, 
            MaxPressure_reward, 
            MaxQueue_reward,
            FixedTime_reward,
            Phase_reward,
            dqn_reward_episodic,
            result_plot_save_path):


        # Plot the baseline reward and dqn reward
        plt.axhline(y=Random_reward,color='blue',label='RandomController')
        plt.axhline(y=MaxPressure_reward,color='red',label='MaxPressureController')
        plt.axhline(y=MaxQueue_reward,color='red',linestyle=':',label='MaxQueueController')
        plt.axhline(y=FixedTime_reward,color='c',label='FixedTimeController')
        plt.axhline(y=Phase_reward,color='y',label='PhaseController')

        plt.plot(dqn_reward_episodic,color='orange',label='dqn_reward_episodic')

        plt.xlabel('episodes')
        plt.ylabel('total reward per episode')
        plt.legend()
        plt.savefig(result_plot_save_path)
        plt.close()

    
    def plot_dqn_test_reward(
            Random_reward, 
            MaxPressure_reward, 
            MaxQueue_reward,
            FixedTime_reward,
            Phase_reward,
            dqn_reward_test,
            result_plot_save_path):

        # Plot the baseline reward and dqn reward
        Rewards = [dqn_reward_test, Random_reward, MaxPressure_reward, MaxQueue_reward, FixedTime_reward, Phase_reward]
        data = pd.DataFrame()
        data['Controller'] = ['DQN','Random','MaxPressure','MaxQueue','FixedTime','Phase']
        data['Rewards'] = Rewards
        sns.barplot(data=data, x="Controller", y="Rewards")
        plt.xlabel('Controllers')
        plt.ylabel('Reward (Total queue vehicle number)')
        plt.legend()
        plt.savefig(result_plot_save_path)
        plt.close()
