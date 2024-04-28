import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import h5py
from Controller import Controller 
from keras.models import load_model
from keras.losses import mean_squared_error

class DQNController(Controller):
    def __init__(self, state_size, action_size, env, \
                 hidden_layers, units, learning_rate, buffer_size, batch_size, gamma):
        Controller.__init__(self,env)
        self.traffic_lights = self.env._traffic_lights
        self.num_traffic_lights = len(self.traffic_lights)
        self.TrafficLight_states = ['GGrrrrGGrrrr' for _ in range(self.num_traffic_lights)]
        
        self.state_size = state_size
        self.action_size = action_size

        # The number of hidden layers in DQN 
        self.hidden_layers = hidden_layers
        # The dimensionality of the output space for each hidden layer
        self.units = units
        # The replay buffer stores experiences (state, action, reward, next state, done flag)
        self.replay_buffer = []
        # The maximum size of the replay buffer. Older experiences are discarded when the buffer is full
        self.buffer_size = buffer_size
        # The size of the batch used for training the DQN. A small batch is sampled from the replay buffer for each training step
        self.batch_size = batch_size
        self.gamma = gamma
        # Initially, it is always exploring
        self.epsilon = 1.0 
        # The minimum value of epsilon after decay. This ensures there is always some exploration
        self.epsilon_min = 0.01 
        # The decay rate of epsilon after each episode, reducing the amount of exploration over time
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self.build_dqn_model()

    
    def build_dqn_model(self):
        model = Sequential()
        model.add(Dense(self.units, input_dim=self.state_size, activation='relu'))
        for _ in range(self.hidden_layers-1):
            model.add(Dense(self.units, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # Compile the model with the Adam optimizer and mean squared error loss function.
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def store_experience(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0) # Remove the oldest experience if buffer is full
        self.replay_buffer.append((state, action, reward, next_state, done)) # Add new experience

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Use the DQN model to predict the Q-values for all possible actions given the current state.
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        # Check if the replay buffer has enough experiences for training
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a minibatch of experiences randomly from the replay buffer.
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Set the target to the immediate reward by default.
            target = reward
             # If the next state is not a terminal state, update the target with discounted future rewards.
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            # Bellman Equation update the target reward by adding the current reward and max reward from next state
            # Predict the Q-values for the current state.
            target_f = self.model.predict(state)
            # Update the Q-value for the taken action to the calculated target value.
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) # Train the model on the current state and updated Q-values.
            # Training process will pass through the sampled minibatch once and will not output any information

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # Reduces the exploration rate over time

    def save_dqn_model(self, save_path):
        self.model.save(save_path)

class DQNController_test(Controller):
    def __init__(self, env, load_path, state_size, action_size):
        Controller.__init__(self,env)
        self.load_path = load_path
        self.model = self.load_dqn_model(self.load_path)
        self.state_size = state_size
        self.action_size = action_size

    def load_dqn_model(self, load_path):
        # Load the model with the custom objects specified
        model = load_model(load_path)
        return model
    
    def choose_action(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
