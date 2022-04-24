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
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):

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
        
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        #self.strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
 
        self.actor_model = self.get_actor(state_shape, action_shape)
        self.target_actor = self.get_actor(state_shape, action_shape)

        self.critic_model = self.get_critic(state_shape, action_shape)
        self.target_critic = self.get_critic(state_shape, action_shape)

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())


    def mapping_to_target_range(x, target_min=0, target_max=3 ) :
        x02 = BK.tanh(x) + 1 # x in range(0,2)
        scale = ( target_max-target_min )/2.
        return  x02 * scale + target_min

    def get_actor(self, state_shape: int, action_shape: int):

       # with self.strategy.scope():

        init = tf.keras.initializers.HeUniform() 
        model = keras.Sequential()

        model.add(LSTM(40, input_shape=(5, 15), return_sequences=True, activation='relu', kernel_initializer=init))
        model.add(LSTM(40, return_sequences=True, activation='relu', kernel_initializer=init))       
        model.add(LSTM(20, activation='relu', kernel_initializer=init))  
        model.add(keras.layers.Dense(action_shape, activation='softmax'))
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'], run_eagerly=True)
        
        return model
    
    def get_critic(self, state_shape, action_shape):
        # Initialize weights between -3e-3 and 3-e3

        #with self.strategy.scope():
            
        # State as input
        state_input = keras.layers.Input(shape=(5, 15))
        state_out = LSTM(50, activation="relu", return_sequences=True)(state_input)
        state_out = LSTM(50, activation="relu")(state_out)

        # Action as input
        action_input = keras.layers.Input(shape=(7))
        action_out = keras.layers.Dense(25, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = keras.layers.Concatenate()([state_out, action_out])
        outputs = keras.layers.Dense(1)(concat)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        return model
            
            

    def learn(self, replay_memory: deque) -> None:
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

        input = [[i] for i in input]
        train_data = tf.data.Dataset.from_tensor_slices((input))   

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        data = train_data.with_options(options)

        return data

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        
            
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
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def policy(self, state):
        
        state = np.array(state)
        state = state.reshape(1, 5, 15)
        sampled_actions = self.actor_model(state).numpy()[0]

        # noise = self.ou_noise() # Adding noise to action
        # transformed_action = ((3/2) * (sampled_actions + 1)) + noise
        
        return np.argmax(sampled_actions)

    def test_policy(self, state):
        
        state = np.array(state)
        state = state.reshape(1, 5, 15)

        sampled_actions = self.actor_model(state).numpy()[0][0]
        noise = self.ou_noise() # Adding noise to action
        transformed_action = ((3/2) * (sampled_actions + 1))

        if transformed_action > 0:
            return transformed_action
        else:
            return 0