from cProfile import label
from environment import Environment
from model import Model
from ddpg_model import DDPGModel

from collections import deque
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import silence_tensorflow.auto
import atexit
import multiprocessing as mp

class Simulator:

    env: Environment
    model: DDPGModel

    def __init__(self) -> None:
        """ Initialize the simulator and configure the experimentation
        
        Arguments:
            None
            
        Returns:   
            None
        """
        
        self.env = Environment()
        self.configure_mlflow()

    def configure_mlflow(self):
        """ Configure the MLflow environment
        
        Arguments:
            None
            
        Returns:
            None
        """

        #notes = input("Enter a note: ")
        notes = 'test'
        if notes != "x":
            
            self.log = True
            mlflow.set_tracking_uri("http://10.0.0.206:5000")
            mlflow.set_experiment('Deep-Q-Learning')
            mlflow.start_run(run_name="Henry G") 
            MlflowClient().set_tag(mlflow.active_run().info.run_id, "mlflow.note.content", notes)
            
            mlflow.log_artifact(f'simulator.py')
            mlflow.log_artifact(f'model.py')
            mlflow.log_artifact(f'environment.py')


    def graph_action_distribution(self, actions: list, title: str):
        """ Graph the action distribution

        Arguments:
            actions (list) -- List of actions taken
        
        Returns:
            None
        """

        data = {}
        for action in actions:

            if action in data:
                data[action] += 1
            else:
                data[action] = 1

        action = list(data.keys())
        values = list(data.values())
        
        plt.bar(action, values)
        plt.savefig(f'{title}_bar.png')
        mlflow.log_artifact(f'{title}_bar.png')

    def ddpg_simulate(self):
        """ Simulate the DDPG algorithm
        
        Arguments:
            None
            
        Returns:
            None
        """

        self.model = DDPGModel(15, 7)

        epsilon = 1 
        max_epsilon = 1 
        min_epsilon = 0.01 
        episode = 0
        replay_memory = deque(maxlen=10000)
        actions = []

        for i in tqdm(range(100)):
            
            done = False
            self.env.reset()
            total_reward = 0

            while(not done):

                random_number = np.random.rand()
                current_state = self.env.current_state()

                if random_number <= epsilon:  # Explore  
                    action = self.env.random_action() # Just randomly choosing an action
                
                else: #Exploitting
                    action = self.model.policy(current_state)
                
                actions.append(action)
                reward, done = self.env.step(action)      # Executing action on current state and getting reward, this also increments out current state
                new_state = self.env.current_state()               
                total_reward += reward
                
                action_set = [0] * 7
                action_set[action] = 1

                replay_memory.append([current_state, action_set, reward, new_state])      # Adding everything to the replay memory
                self.model.learn(replay_memory)

            mlflow.log_metric("reward", total_reward)
            episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-0.01 * episode)

        self.graph_action_distribution(actions, "training")

    def DQN_simulate(self):
        """ Simulate the DQN algorithm
        
        Arguments:
            None
            
        Returns:   
            None
        """

        self.model = Model(15,7)

        epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
        max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
        episode = 0
        replay_memory = deque(maxlen=10000)
        actions = []

        for i in tqdm(range(100)):
            
            done = False
            steps_to_update_target_model = 0
            self.env.reset()

            total_reward = 0

            while(not done):

                steps_to_update_target_model += 1 
                random_number = np.random.rand()
                current_state = self.env.current_state()

                if random_number <= epsilon:  # Explore  
                    action = self.env.random_action() # Just randomly choosing an action
                
                else: #Exploitting
                    
                    current_reshaped = np.array(current_state).reshape([1, 5, 15])
                    predicted = self.model.model(current_reshaped).numpy()[0]          # Predicting best action, not sure why flatten (pushing 2d into 1d)
                    action = np.argmax(predicted) 
                    
                actions.append(action)

                reward, done = self.env.step(action)      # Executing action on current state and getting reward, this also increments out current state
                new_state = self.env.current_state()               
                replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory
                total_reward += reward

                if steps_to_update_target_model % 10 == 0 or done:                   # If we've done 4 steps or have lost/won, update the main neural net, not target
                    self.model.train(replay_memory, done)            # training the main model
 
            episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-0.01 * episode)
            self.model.update_target()
        
    def test(self):
        """ Test a given model against the first 100

        Arguments:
            None
            
        Returns:
            None
        """

        self.env.reset()
        total_reward = 0
        done = False

        while(not done):

            current_state = self.env.current_state()
            action = self.model.predict(current_state)          
            reward, done = self.env.step(action)      
            total_reward += reward
        
        return total_reward
    

def run_test(i: int, results: list):
    """ Run a test

    Arguments:
        i (int): The test number
        results (list): The list of results

    Returns:
        None
    """

    s = Simulator()
    s.DQN_simulate()
    reward = s.test(i)
    results.append(reward)

def test():
    """ Test the simulator
    
    Arguments:
        None
        
    Returns:
        None
    """

    pool = mp.Pool(10)
    results = mp.Manager().list()

    for i in range(5):
        pool.apply_async(run_test, args=(i, results,))

    pool.close()
    pool.join()

    with open(f'test/{0}.txt', 'w') as f:
        f.write(f'Made it to: {sum(results)/len(results)}')

def main():
    test()
    
    
if __name__ == '__main__':
    main()