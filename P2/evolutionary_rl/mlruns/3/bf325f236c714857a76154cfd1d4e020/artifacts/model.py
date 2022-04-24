from collections import deque
import random
from tabnanny import verbose
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from collections import deque

from country import Country
from environment import Environment

class Model():

    model: object
    target_model: object
    replay_memory: deque
    state_shape: tuple
    action_shape: int

    country_index: int

    epsilon: float = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
    max_epsilon: int = 1 # You can't explore more than 100% of the time - Makes sense
    min_epsilon: float = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
    episode: int = 0
    actions: list = []
    total_reward: int = 0

    steps_taken: int = 0

    def __init__(self, state_shape: int, action_shape: int, country_index: int) -> None:
        
        physical_devices = tf.config.list_physical_devices('GPU')
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        self.strategy = tf.distribute.MirroredStrategy()
        
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.country_index = country_index

        self.model = self.create_agent(state_shape, action_shape)
        self.target_model = self.create_agent(state_shape, action_shape)

        self.replay_memory = deque(maxlen=10000)
        
    def create_agent(self, state_shape: int, action_shape: int) -> object:
        """ Takes the current state shape and the action space shape
        to create neural network paramterized for this. Yses relu activation
        and HEUniform transformation normalizer. 36 Hidden layers all together 
        
        Notes:
            The agent maps X-states to Y-actions
            e.g. The neural network output is [.1, .7, .1, .3]      # Is this the q value then?
            The highest value 0.7 is the Q-Value.
            The index of the highest action (0.7) is action #1.     # So q value for all possible actions, highest is chosen

        Arguments:
            state_shape (int): The shape of the current state space

            action_shape (int): The shape of the current action space
        
        Return:
            model (keras.neural_net): Neural network by keras
        
        Side Effects:
            None
        """

        with self.strategy.scope():

            model = keras.Sequential()      
            model.add(LSTM(40, input_shape=state_shape, return_sequences=True, activation='relu'))
            model.add(LSTM(40, return_sequences=True, activation='relu'))       
            model.add(LSTM(20, activation='relu'))  
            model.add(keras.layers.Dense(action_shape, activation='softmax'))
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'], run_eagerly=True)
            return model

    def step(self, environment: Environment) -> int:
        
        random_number = np.random.rand()
        current_state = environment.current_state(self.country_index)

        if random_number <= self.epsilon:  # Explore  
            action = random.randint(0, 6)

        else: 
            current_reshaped = np.array(current_state).reshape([1, self.state_shape[0], self.state_shape[1]])
            predicted = self.model(current_reshaped).numpy()[0]          # Predicting best action, not sure why flatten (pushing 2d into 1d)
            action = np.argmax(predicted) 
        
        reward, done = environment.step(action, self.country_index)      # Executing action on current state and getting reward, this also increments out current state
        new_state = environment.current_state(self.country_index)               
        self.replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory
        self.total_reward += reward

        self.steps_taken += 1

        if done:
            self.train(True)
            self.episode += 1
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-0.01 * self.episode)
            self.update_target()

        elif self.steps_taken % 10 == 0:
            self.train(False)

    def train(self, done: bool) -> None:
        """ Thes the current enviroment, replay memeory, model and target model
        to test if there is enoguh memory cached. If there is, takes a random 128 
        examples from the memory and uses that to retrain the target model 
        
        Arguments:
            env (TrainingEnviroment): The current TrainingEnviroment object assoicated with the training

            replay_memory (deque): The current cached memeory associated with the training

            model (object): The given neural network

            target_model (object): The given nerual network to train

            done (bool): Whether training has finished or not
        
        Return:
            None
        
        Side Effects:
            None
        """

        learning_rate = 0.7         
        discount_factor = 0.618     

        MIN_REPLAY_SIZE = 500      
        if len(self.replay_memory) < MIN_REPLAY_SIZE:        # Only do this function when we've gone through atleast 1000 steps?
            return 0, 0

        data_set_size = 256     # Getting random 128 batch sample from 
        mini_batch = random.sample(self.replay_memory, data_set_size)       # Grabbing said random sample
        
        current_states = np.array([transition[0] for transition in mini_batch])     # Getting all the states from your sampled mini batch, because index 0 is the observation
        current_states = np.array(current_states).reshape([len(current_states), 5, 15])
        current_qs_list = self.model(current_states).numpy()     # Predict the q values based on all the historical state

        new_current_states = np.array([transition[3] for transition in mini_batch]) # Getting all of the states after we executed our action? 
        new_current_states = np.array(new_current_states).reshape([len(new_current_states), 5, 15])
        future_qs_list = self.target_model(new_current_states).numpy()       # the q values resulting in our action

        X = []
        Y = []

        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):       # Looping through our randomly sampled batch
            
            if not done:                                                                                # If we havent finished the game or died?
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])                 # Calculuting max value for each step using the discount factor
            else:
                max_future_q = reward                                                                   # if we finished then max is just the given reqard
            
            current_qs = current_qs_list[index]     # Getting current value of q's
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q        # Updating the q values

            X.append(observation)           # Creating model input based off this
            Y.append(current_qs)             

        train_data = tf.data.Dataset.from_tensor_slices((X,Y))
        train_data = train_data.batch(32, drop_remainder=True)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options)

        history = self.model.fit(train_data, verbose=1, shuffle=False) 
        return history.history['loss'][-1], history.history['accuracy'][-1]

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())