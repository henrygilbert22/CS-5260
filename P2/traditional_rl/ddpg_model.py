from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from keras import backend as BK
import mlflow

class OUActionNoise:

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        """ Initializes the OUActionNoise object
        
        Arguments:
            mean (np.array): The mean of the OU process
            
            std_deviation (np.array): The standard deviation of the OU process
            
            theta (float): The theta parameter of the OU process
            
            dt (float): The time step of the OU process
            
            x_initial (np.array): The initial state of the OU process
            
        Returns:
            None    
        """

        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        """ Generates a random action with noise

        Arguments:
            None

        Returns:
            action (np.array): The action with noise
        """

        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        
        self.x_prev = x
        return x

    def reset(self):
        """ Resets the OU process to its initial state
        
        Arguments:
            None
            
        Returns:
            None
        """

        if self.x_initial is not None:
            self.x_prev = self.x_initial

        else:
            self.x_prev = np.zeros_like(self.mean)



class DDPGModel():

    actor_model: object
    target_actor: object

    critic_model: object
    target_critic: object

    critic_lr: int = 0.002
    actor_lr:int = 0.001
    std_dev: int = 0.2

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    gamma = 0.99
    tau = 0.005

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    buffer: object
    strategy: object
    
    def __init__(self, state_shape: int, action_shape: int) -> None:
        """ Initializes the DDPGModel object
        
        Arguments:
            state_shape (int): The shape of the state
            
            action_shape (int): The shape of the action
            
        Returns:
            None
        """
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
 
        self.actor_model = self.get_actor(state_shape, action_shape)
        self.target_actor = self.get_actor(state_shape, action_shape)

        self.critic_model = self.get_critic(state_shape, action_shape)
        self.target_critic = self.get_critic(state_shape, action_shape)

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())


    def mapping_to_target_range(x, target_min=0, target_max=3):
        """ Maps the input to the target range
        
        Arguments:
            x (float): The input
            
            target_min (float): The minimum of the target range
            
            target_max (float): The maximum of the target range
            
        Returns:
            x (float): The mapped input
        """

        x02 = BK.tanh(x) + 1 # x in range(0,2)
        scale = ( target_max-target_min )/2.
        return  x02 * scale + target_min

    def get_actor(self, state_shape: int, action_shape: int):
        """ Initializes the actor model
        
        Arguments:
            state_shape (int): The shape of the state
            
            action_shape (int): The shape of the action
            
        Returns:
            model (object): The actor model
        """
       
        init = tf.keras.initializers.HeUniform() 
        model = keras.Sequential()

        model.add(LSTM(40, input_shape=(5, 15), return_sequences=True, 
            activation='relu', kernel_initializer=init))
        model.add(LSTM(40, return_sequences=True, activation='relu', 
            kernel_initializer=init))       
        model.add(LSTM(20, activation='relu', kernel_initializer=init))  

        model.add(keras.layers.Dense(action_shape, activation='softmax'))

        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
            metrics=['accuracy'], 
            run_eagerly=True
        )
        
        return model
    
    def get_critic(self, state_shape, action_shape):
        """ Initializes the critic model
        
        Arguments:
            state_shape (int): The shape of the state
            
            action_shape (int): The shape of the action
            
        Returns:
            model (object): The critic model
        """
       
        state_input = keras.layers.Input(shape=(5, 15))
        state_out = LSTM(50, activation="relu", return_sequences=True)(state_input)
        state_out = LSTM(50, activation="relu")(state_out)

        action_input = keras.layers.Input(shape=(7))
        action_out = keras.layers.Dense(25, activation="relu")(action_input)

        concat = keras.layers.Concatenate()([state_out, action_out])
        outputs = keras.layers.Dense(1)(concat)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model
            
            

    def learn(self, replay_memory: deque) -> None:
        """ Thes the current enviroment, replay memeory, model and target model
        to test if there is enoguh memory cached. If there is, takes a random 128 
        examples from the memory and uses that to retrain the target model 
        
        Arguments:
            env (TrainingEnviroment): The current TrainingEnviroment object assoicated 
                with the training

            replay_memory (deque): The current cached memeory associated with the training

            model (object): The given neural network

            target_model (object): The given nerual network to train

            done (bool): Whether training has finished or not
        
        Return:
            None
        """

        if len(replay_memory) < 512:
            return

        mini_batch = random.sample(replay_memory, 256)  

        state_batch = [transition[0] for transition in mini_batch]
        action_batch = [transition[1] for transition in mini_batch]
        reward_batch = [transition[2] for transition in mini_batch]
        next_state_batch = [transition[3] for transition in mini_batch]

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

        self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def create_dataset(self, input: np.array):
        """ Creates a dataset from the given input
        
        Arguments:
            input (np.array): The input to create a dataset from
            
        Returns:
            dataset (tf.data.Dataset): The dataset created from the input
        """

        input = [[i] for i in input]
        train_data = tf.data.Dataset.from_tensor_slices((input))   

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        data = train_data.with_options(options)

        return data

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        """ Updates the model with the given state, action, reward and next state
        
        Arguments:
            state_batch (np.array): The state batch to update the model with
            
            action_batch (np.array): The action batch to update the model with
            
            reward_batch (np.array): The reward batch to update the model with
            
            next_state_batch (np.array): The next state batch to update the model with
            
        Returns:
            None
        """
            
        next_state_batch = np.array(next_state_batch)
        next_state_batch = next_state_batch.reshape(len(next_state_batch), 5, 15)
        
        state_batch = np.array(state_batch)
        state_batch = state_batch.reshape(len(state_batch), 5, 15)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)

            target_actions = target_actions.numpy().reshape(256, 7)
            action_batch = np.array(action_batch).reshape(256, 7)

            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
                )

            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            mlflow.log_metric("critic_loss", critic_loss)

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
            mlflow.log_metric("actor_loss", actor_loss)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )  

    @tf.function
    def update_target(self, target_weights, weights, tau):
        """ Updates the target model with the given target weights and weights
        
        Arguments:
            target_weights (np.array): The target weights to update the target model with
            
            weights (np.array): The weights to update the target model with
            
            tau (float): The tau value to update the target model with
        
        Returns:
            None
        """
        
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def policy(self, state):
        """ Returns the action to take given the state
        
        Arguments:
            state (np.array): The state to return the action for
            
        Returns:
            action (np.array): The action to take given the state
        """
        
        state = np.array(state)
        state = state.reshape(1, 5, 15)
        sampled_actions = self.actor_model(state).numpy()[0]

        noise = self.ou_noise() # Adding noise to action
        transformed_action = ((3/2) * (sampled_actions + 1)) + noise
        
        return np.argmax(transformed_action)

    def test_policy(self, state):
        """ Returns the action to take given the state
        
        Arguments:
            state (np.array): The state to return the action for
            
        Returns:
            action (np.array): The action to take given the state
        """
        
        state = np.array(state)
        state = state.reshape(1, 5, 15)

        sampled_actions = self.actor_model(state).numpy()[0][0]
        noise = self.ou_noise() # Adding noise to action
        transformed_action = ((3/2) * (sampled_actions + 1))

        if transformed_action > 0:
            return transformed_action
        else:
            return 0