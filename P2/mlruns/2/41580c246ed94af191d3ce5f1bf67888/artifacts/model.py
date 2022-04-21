from collections import deque
import random
from tabnanny import verbose
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import sys

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
            model.add(keras.layers.Dense(10, input_shape=(state_shape,), activation='relu'))     
            model.add(keras.layers.Dense(10, activation='relu'))
            model.add(keras.layers.Dense(action_shape, activation='softmax'))
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'], run_eagerly=True)
            return model

    def create_dataset(self, data: list) -> object:
        """ Takes the data and creates a dataset for the model to train on.
        The dataset is a list of tuples. Each tuple is a state and action.
        The dataset is shuffled and then split into training and testing sets.
        The training set is used to train the model. The testing set is used
        to test the model.

        Notes:
            The dataset is a list of tuples. Each tuple is a state and action.
            The dataset is shuffled and then split into training and testing sets.
            The training set is used to train the model. The testing set is used
            to test the model.

        Arguments:
            data (list): A list of tuples. Each tuple is a state and action.
        
        Return:
            dataset (tuple): A tuple of training and testing datasets.
        
        Side Effects:
            None
        """

        

        return train_data

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

        # learning_rate = 0.7         
        # discount_factor = 0.618     

        # MIN_REPLAY_SIZE = 500      
        # if len(replay_memory) < MIN_REPLAY_SIZE:        # Only do this function when we've gone through atleast 1000 steps?
        #     return 0, 0

        # data_set_size = 256     # Getting random 128 batch sample from 
        # mini_batch = random.sample(replay_memory, data_set_size)       # Grabbing said random sample
        
        # current_states = np.array([transition[0] for transition in mini_batch])     # Getting all the states from your sampled mini batch, because index 0 is the observation
        # current_states = np.array(current_states).reshape([len(current_states), len(current_states[0])])
        # current_qs_list = self.model(current_states).numpy()     # Predict the q values based on all the historical state

        # new_current_states = np.array([transition[3] for transition in mini_batch]) # Getting all of the states after we executed our action? 
        # new_current_states = np.array(new_current_states).reshape([len(new_current_states), len(new_current_states[0])])
        # future_qs_list = self.target_model(new_current_states).numpy()       # the q values resulting in our action

        # X = []
        # Y = []

        # for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):       # Looping through our randomly sampled batch
            
        #     if not done:                                                                                # If we havent finished the game or died?
        #         max_future_q = reward + discount_factor * np.max(future_qs_list[index])                 # Calculuting max value for each step using the discount factor
        #     else:
        #         max_future_q = reward                                                                   # if we finished then max is just the given reqard
            
        #     action -= 1
        #     current_qs = current_qs_list[index]     # Getting current value of q's
        #     current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q        # Updating the q values

        #     X.append(observation)           # Creating model input based off this
        #     Y.append(current_qs)            

        with open('big_X.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('big_Y.pkl', 'rb') as f:
            Y = pickle.load(f)   

        train_data = tf.data.Dataset.from_tensor_slices((X,Y))
        train_data = train_data.batch(16, drop_remainder=True)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options)

        history = self.model.fit(train_data, verbose=1)  
        return 0, 0

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())


def main():

    with open('big_X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('big_Y.pkl', 'rb') as f:
        Y = pickle.load(f)

    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])

    with strategy.scope():

        model = keras.Sequential()      
        model.add(keras.layers.Dense(10, input_shape=(15,), activation='relu'))     
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(7, activation='softmax'))
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    
    train_data = tf.data.Dataset.from_tensor_slices((X, Y))
    train_data = train_data.batch(16, drop_remainder=True)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_data = train_data.with_options(options)

    history = model.fit(train_data)   

if __name__ == "__main__":
    main()