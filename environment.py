import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import math
import traci
import random
import pandas as pd
import numpy as np
from sumolib import checkBinary  
import matplotlib.pyplot as plt

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

class SumoEnvironment():
    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        max_steps: int = 10000
        ):

        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = checkBinary("sumo-gui")
        else:
            self._sumo_binary = checkBinary("sumo")

        self._net_file = net_file
        self._route_file = route_file
        self.out_csv_name = out_csv_name
        self.max_steps = max_steps
        
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(tools)
        else:
            raise ImportError("Please declare the environment variable 'SUMO_HOME'")
       
        traci.start([self._sumo_binary, "-n", self._net_file])
        traci_retrieve = traci
        self._lane_IDs = self.get_lane_IDs(traci_retrieve)
        self._junction_IDs = self.get_junction_IDs(traci_retrieve)
        self.start_end_points_x, self.start_end_points_y = self.get_start_end_points_x_y_axis(traci_retrieve, self._junction_IDs)
        self._traffic_lights = self.get_Traffic_Lights(traci_retrieve)
        self._num_traffic_lights = len(self._traffic_lights)
        traci_retrieve.close()

        # Record the episode of q learning
        self.episode = 0
        # Initial the waiting time for q learning reward
        self.waiting_time = 0
        # Set the interval for each action step 
        self.action_step = 5
        # Initial the time step in each episode
        self.time = 0
        # Initial the total reward in each episode
        self.total_reward = 0
        # Initial last step waiting time
        self.last_step_waiting_time = 0

        

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net_file,
            "-r",
            self._route_file,
            "--junction-taz"
        ]
        traci.start(sumo_cmd)
        self.sumo = traci
        

    def _sumo_step(self): # Do 1 step simulation
        self.sumo.simulationStep()

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        self.sumo.close()


    def get_lane_IDs(self, sumo_traci):
        lanes = sumo_traci.lane.getIDList()
        return lanes
        
    def get_junction_IDs(self, sumo_traci):
        junctions = sumo_traci.junction.getIDList()
        return junctions
        
    def get_start_end_points_x_y_axis(self, sumo_traci, junctions): 
        # use get_dead_end_junctions to help generate route file, because we assume that the car depart from one of the dead end points and leave at one of them
        junction_Positions = pd.DataFrame()
        junction_Positions['ID'] = junctions
        junction_Positions['x'] = [sumo_traci.junction.getPosition(junction)[0] for junction in junctions]
        junction_Positions['y'] = [sumo_traci.junction.getPosition(junction)[1] for junction in junctions]
        min_x = junction_Positions['x'].min()
        max_x = junction_Positions['x'].max()
        min_y = junction_Positions['y'].min()
        max_y = junction_Positions['y'].max()
        start_end_points_x = junction_Positions[(junction_Positions['x']==min_x)|(junction_Positions['x']==max_x)]['ID']
        start_end_points_y = junction_Positions[(junction_Positions['y']==min_y)|(junction_Positions['y']==max_y)]['ID']
        return list(start_end_points_x),list(start_end_points_y) 
    
    def get_Traffic_Lights(self, sumo_traci):
        traffic_lights = sumo_traci.trafficlight.getIDList()
        return traffic_lights
    
    def set_Traffic_Lights_States(self, traffic_lights, TrafficLight_states):
        for i in range(len(traffic_lights)):
            self.sumo.trafficlight.setRedYellowGreenState(traffic_lights[i], TrafficLight_states[i])

    def get_Traffic_Lights_States(self, traffic_lights):
        traffic_lights_states = []
        for i in range(len(traffic_lights)):
            traffic_lights_states.append(self.sumo.trafficlight.getRedYellowGreenState(traffic_lights[i]))
        return traffic_lights_states

    def get_Vehicle_Numbers(self,lanes):
        Vehicle_Numbers = np.array([self.sumo.lane.getLastStepVehicleNumber(lane) for lane in lanes])
        return Vehicle_Numbers

    def get_Waiting_Time(self,lanes):
        Waiting_Time = np.array([self.sumo.lane.getWaitingTime(lane) for lane in lanes])
        return Waiting_Time

    def get_Halting_Numbers(self,lanes):
        Halting_Numbers = np.array([self.sumo.lane.getLastStepHaltingNumber(lane) for lane in lanes])
        return Halting_Numbers
    
    def get_Occupancy(self, lanes):
        Occupancy = np.array([self.sumo.lane.getLastStepOccupancy(lane) for lane in lanes])
        return Occupancy
    
    def get_Mean_Speed(self, lanes):
        Mean_Speed = np.array([self.sumo.lane.getLastStepMeanSpeed(lane) for lane in lanes])
        return Mean_Speed
 
    def get_Traffic_Light_Controlled_Lanes(self, traffic_light):
        Controlled_Lanes = self.sumo.trafficlight.getControlledLanes(traffic_light)
        return Controlled_Lanes
    
    def get_Traffic_Light_Controlled_Links(self, traffic_light):
        Controlled_Links = self.sumo.trafficlight.getControlledLinks(traffic_light)
        return Controlled_Links
    
    def get_Traffic_Light_Out_Lanes(self, traffic_light):
        Out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(traffic_light) if link]
        return Out_lanes

    def get_Pressure(self, Traffic_Light_Controlled_Lanes, Traffic_Light_Out_Lanes):
        Controlled_Lanes = Traffic_Light_Controlled_Lanes
        Out_lanes = Traffic_Light_Out_Lanes
        Controlled_Lanes_Vehicle_Numbers = self.get_Vehicle_Numbers(Controlled_Lanes)
        Out_lanes_Vehicle_Numbers = self.get_Vehicle_Numbers(Out_lanes)
        return sum(Controlled_Lanes_Vehicle_Numbers)-sum(Out_lanes_Vehicle_Numbers)
    
    # Here are the functions we need for reinforcement learning

    # get_states: return the traffic light phase(one-hot encoding), traffic light controled/Out lanes vehicle numbers(Pressure)
    def get_states(self):
        states = np.array([])

        #The firse part of states is the traffic light states
        traffic_lights = self._traffic_lights
        traffic_lights_states = self.get_Traffic_Lights_States(traffic_lights)
        traffic_lights_state_mapping = {
            'GGrrrrGGrrrr': 0,
            'rrGrrrrrGrrr': 1,
            'rrrGGrrrrGGr': 2,
            'rrrrrGrrrrrG': 3
        }
        states = np.array([traffic_lights_state_mapping.get(traffic_lights_states[i], -1) for i in range(len(traffic_lights))])
        
        for tl in traffic_lights:
            tl_Controlled_Lanes = self.get_Traffic_Light_Controlled_Lanes(tl)
            tl_Controlled_Lanes_Halting_Numbers = self.get_Halting_Numbers(tl_Controlled_Lanes)
            tl_Controlled_Lanes_Occupancy = self.get_Occupancy(tl_Controlled_Lanes)
            tl_Controlled_Lanes_Mean_Speed = self.get_Mean_Speed(tl_Controlled_Lanes)
            states = np.hstack((states, tl_Controlled_Lanes_Halting_Numbers, tl_Controlled_Lanes_Occupancy, tl_Controlled_Lanes_Mean_Speed))
        return states

    # get_reward:

    # 1. Queue reward
    def get_queue_reward(self):
        reward = -sum(self.get_Halting_Numbers(self._lane_IDs))
        self.total_reward+=reward
        return reward
        
    # 2. Delay reward
    def get_delay_reward(self):
        reward = -sum(self.get_Waiting_Time(self._lane_IDs))
        self.total_reward+=reward
        return reward
    
    # 3. Mean Speed reward
    def get_mean_speed_reward(self):
        if sum(self.get_Vehicle_Numbers(self._lane_IDs))==0:
            reward = 0
        else:
            total_speed = sum(self.get_Mean_Speed(self._lane_IDs)*self.get_Vehicle_Numbers(self._lane_IDs))
            vehicle_numbers = sum(self.get_Vehicle_Numbers(self._lane_IDs))
            reward = total_speed/vehicle_numbers
        self.total_reward+=reward
        return reward
    
    def get_total_reward(self):
        return self.total_reward
    
    # reset(return state) 
    def reset(self):
        if self.episode!=0:
            self.close()
        self.time = 0
        self.episode += 1
        self.total_reward = 0
        self._start_simulation()
        states = self.get_states()
        return states

    # step(return state, reward, done)
    def step(self, action, reward_type):
        if self.time!=0:
            self.apply_action(action)
        
        if self.time + self.action_step <= self.max_steps:
            for steps in range(self.action_step):
                self._sumo_step()
            self.time += self.action_step
        else:
            for steps in range(self.max_steps-self.time):
                self._sumo_step()
            self.time = self.max_steps

        states = self.get_states()

        if reward_type == 'Queue':
            reward = self.get_queue_reward()
        elif reward_type == 'Delay':
            reward = self.get_delay_reward()
        elif reward_type == 'MeanSpeed':
            reward = self.get_mean_speed_reward()
        else:
            print('Error reward type')
            reward = 0
        done = (self.time==self.max_steps)
        return states, reward, done
    
    # step for non reinforcement learning controller
    def non_rl_step(self, reward_type):
        if self.time + self.action_step <= self.max_steps:
            for steps in range(self.action_step):
                self._sumo_step()
            self.time += self.action_step
        else:
            for steps in range(self.max_steps-self.time):
                self._sumo_step()
            self.time = self.max_steps
        if reward_type == 'Queue':
            reward = self.get_queue_reward()
        elif reward_type == 'Delay':
            reward = self.get_delay_reward()
        elif reward_type == 'MeanSpeed':
            reward = self.get_mean_speed_reward()
        else:
            print('Error reward type')
            reward = 0
        done = (self.time==self.max_steps)
        return reward, done
    
    def apply_action(self, action):
        traffic_lights = self._traffic_lights
        TrafficLight_states = []
        actions_list = ['GGrrrrGGrrrr', 'rrGrrrrrGrrr', 'rrrGGrrrrGGr', 'rrrrrGrrrrrG']
        #action_remain = action
        for i in range(len(self._traffic_lights)):
            #action_index = action_remain % 4
            TrafficLight_states.append(actions_list[action])
            #action_remain = action_remain // 4
        self.set_Traffic_Lights_States(traffic_lights,TrafficLight_states)
        return

    
