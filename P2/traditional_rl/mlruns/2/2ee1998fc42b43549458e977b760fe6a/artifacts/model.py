from collections import deque
import random
from tabnanny import verbose
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM

from country import Country

class Model():

    model: object
    target_model: object
    
    def __init__(self, state_shape: int, action_shape: int) -> None:
        
        physical_devices = tf.config.list_physical_devices('GPU')
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        self.strategy = tf.distribute.MirroredStrategy()

        self.model = self.create_agent(state_shape, action_shape)
        self.target_model = self.create_agent(state_shape, action_shape)

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
            model.add(LSTM(40, input_shape=(5, 15), return_sequences=True, activation='relu'))
            model.add(LSTM(40, return_sequences=True, activation='relu'))       
            model.add(LSTM(20, activation='relu'))  
            model.add(keras.layers.Dense(action_shape, activation='softmax'))
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'], run_eagerly=True)
            return model

    def predict(self, state: np.array) -> np.array:

        current_reshaped = np.array(state).reshape([1, 5, 15])
        action = self.model(current_reshaped).numpy()[0] 
        return np.argmax(action) 

    def train(self, replay_memory: deque, done: bool) -> None:
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
        if len(replay_memory) < MIN_REPLAY_SIZE:        # Only do this function when we've gone through atleast 1000 steps?
            return 0, 0

        data_set_size = 256     # Getting random 128 batch sample from 
        mini_batch = random.sample(replay_memory, data_set_size)       # Grabbing said random sample
        
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