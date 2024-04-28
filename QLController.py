from Controller import Controller
import random

class QLController(Controller):

    def __init__(self, env):
        Controller.__init__(self,env)
        self.num_traffic_lights = len(self.env._traffic_lights)
        self.TrafficLight_states = ['GGGrrrrrrrrr' for _ in range(self.num_traffic_lights)]
        self.state_space, self.action_space = self.define_space()
        self.state_transitions = {
            0: 'GGGrrrrrrrrr',
            1: 'rrrGGGrrrrrr',
            2: 'rrrrrrGGGrrr',
            3: 'rrrrrrrrrGGG'
        }


        # Initialize the Q-table for Q-learning
        # The Q-table maps each state to a list of values for each action, initially set to 0
        self.q_table = {state: [0 for _ in self.action_space] for state in self.state_space}
        self.alpha = 0.5
        self.gamma = 0.95
        self.epsilon = 0.9

    def define_space(self):
        state_space = ['GGGrrrrrrrrr', 'rrrGGGrrrrrr', 'rrrrrrGGGrrr', 'rrrrrrrrrGGG']
        action_space = [0, 1, 2, 3]
        return state_space, action_space



    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)
        else:
            # Enumerate states, compare Q values, and return the action (first element)
            return self.action_space[max(enumerate(self.q_table[state]), key=lambda x: x[1])[0]]

    def learn(self, state, next_state, reward, action):
        current_action_index = self.action_space.index(action)
        max_future_q = max(self.q_table[next_state])
        current_q = self.q_table[state][current_action_index]
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][current_action_index] = new_q
    

    def apply_action_to_state(self, current_state, action):
        # Map action to state transition
        return self.state_transitions[action]

# get actions for traffic light control
    def get_actions(self, time):
        new_states = []

        traffic_lights = self.env._traffic_lights
        
        for i, traffic_light in enumerate(traffic_lights):
            Traffic_Light_Controlled_Lanes = self.env.get_Traffic_Light_Controlled_Lanes(traffic_light)
            Traffic_Light_Controlled_Vehicle_Numbers = self.env.get_Vehicle_Numbers(Traffic_Light_Controlled_Lanes)
            Edge_Queue_Sum = sum([sum(Traffic_Light_Controlled_Vehicle_Numbers[j:j+3]) for j in range(10)if j % 3 == 0])

            current_state = self.TrafficLight_states[i]
            action = self.act(current_state)
            next_state = self.apply_action_to_state(current_state, action)

            reward = -Edge_Queue_Sum  
            self.learn(current_state, next_state, reward, action)
            new_states.append(next_state)

        self.TrafficLight_states = new_states
        return self.TrafficLight_states
