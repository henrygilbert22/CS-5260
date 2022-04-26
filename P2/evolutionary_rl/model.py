from collections import deque
import random
from tabnanny import verbose
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from collections import deque
from threading import Thread

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
        """ Initializes the model and target model. Create multi-gpu strategy to 
        train model in parallel
        
        Arguments:
            state_shape (int): The shape of the current state space

            action_shape (int): The shape of the current action space

            country_index (int): The index of the country to be trained

        Return:
            None
        """
        
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
        to create neural network paramterized for this 

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
            model.add(LSTM(150, input_shape=state_shape, return_sequences=True, activation='relu'))
            model.add(LSTM(150, return_sequences=True, activation='relu'))       
            model.add(LSTM(80, activation='relu')) 
            model.add(keras.layers.Dense(40, activation='relu')) 

            model.add(keras.layers.Dense(action_shape, activation='softmax'))
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), 
            metrics=['accuracy'], run_eagerly=True)

            return model

    def predict(self, state: np.array) -> np.array:
        """ Predict the action to take given the current state
        
        Arguments:
            state (np.array): The current state
            
        Return:
            action (np.array): The action to take
        """

        current_reshaped = np.array(state).reshape([1, self.state_shape[0], self.state_shape[1]])
        action = self.model(current_reshaped).numpy()[0] 
        return np.argmax(action) 

    def test_step(self, environment: Environment) -> int:
        """ Test step is used to test the model on the test set
        
        Arguments:
            environment (Environment): The environment to test the model on
            
        Return:
            None
        """

        current_state = environment.current_state(self.country_index)
        action = self.predict(current_state)

        reward, done = environment.step(action, self.country_index)      
        self.total_reward += reward

    def step(self, environment: Environment) -> int:
        """ Step is used to take a step in the environment and update the model
        
        Arguments:
            environment (Environment): The environment to take a step in
            
        Return:
            None
        """

        random_number = np.random.rand()
        current_state = environment.current_state(self.country_index)

        if random_number <= self.epsilon:  # Explore  
            action = random.randint(0, 6)

        else: 
            action = self.predict(current_state)       
        
        reward, done = environment.step(action, self.country_index)     
        new_state = environment.current_state(self.country_index)               
        self.replay_memory.append([current_state, action, reward, new_state, done])      
        self.total_reward += reward

        self.steps_taken += 1

        if done:
            self.train(done, self.state_shape)
            self.episode += 1
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
                np.exp(-0.01 * self.episode)
            self.update_target(self.model.get_weights())

        elif self.steps_taken % 10 == 0:
            self.train(done, self.state_shape)


    def train(self, done: bool, state_space: list) -> None:
        """ Thes the current enviroment, replay memeory, model and target model
        to test if there is enoguh memory cached. If there is, takes a random 128 
        examples from the memory and uses that to retrain the target model 
        
        Arguments:
            done (bool): Whether training has finished or not

            state_space (list): The current state space
        
        Return:
            None
        
        Side Effects:
            None
        """
        
        learning_rate = 0.7         
        discount_factor = 0.618     

        MIN_REPLAY_SIZE = 500      
        if len(self.replay_memory) < MIN_REPLAY_SIZE:       
            return
        
        data_set_size = 256      
        mini_batch = random.sample(self.replay_memory, data_set_size)       
        
        current_states = np.array([transition[0] for transition in mini_batch])     
        current_states = np.array(current_states).reshape(
            [len(current_states), 
            state_space[0], 
            state_space[1]]
        )
        current_qs_list = self.model(current_states).numpy()     

        new_current_states = np.array([transition[3] for transition in mini_batch]) 
        new_current_states = np.array(new_current_states).reshape(
            [len(new_current_states), 
            state_space[0], 
            state_space[1]]
        )
        future_qs_list = self.target_model(new_current_states).numpy()       

        X = []
        Y = []

        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):       
            
            if not done:                                                                                
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])                 
            else:
                max_future_q = reward                                                                 
            
            current_qs = current_qs_list[index]     
            current_qs[action] = (1 - learning_rate) * current_qs[action] + \
            learning_rate * max_future_q        

            X.append(observation)           
            Y.append(current_qs)             

        train_data = tf.data.Dataset.from_tensor_slices((X,Y))
        train_data = train_data.batch(data_set_size)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options)

        history = self.model.fit(train_data, verbose=1, shuffle=False) 

        with open(f'processes/{self.country_index}', 'a') as f:
            f.write(f'loss: {history.history["loss"][-1]} \
            - acc: {history.history["accuracy"][-1]} - epsilon: {self.epsilon}\n')

    def update_target(self, model_weights: np.array) -> None:
        """ Updates the target model with the weights of the current model

        Arguments:
            model_weights (np.array): The weights of the current model
        
        Return:
            None
        """
        
        self.target_model.set_weights(model_weights)

       