from cProfile import label
from environment import Environment
from model import Model

from collections import deque
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import silence_tensorflow.auto
import atexit
import time
import multiprocessing as mp

class Simulator:

    env: Environment
    models: list

    def __init__(self) -> None:
        """ Initialize the simulator and configure the experimentation
        
        Arguments:
            None
            
        Returns:   
            None
        """
                
        self.env = Environment()
        self.models = []

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
            mlflow.set_experiment('Deep-RL-Evolution')
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

    def simulate_parallel(self, replicas: int, test: bool = False):
        """ Simulate the environment in parallel
        
        Arguments:
            replicas (int) -- Number of replicas to run

            test (bool) -- Whether to test the model or not

        Returns:
            None
        """

        pool = mp.Pool(processes=replicas)
        shared_total_rewards = mp.Manager().list()
        shared_weights = mp.Manager().list()

        for i in range(replicas):
            
            shared_total_rewards.append(False)
            pool.apply_async(simulate_single_country, args=(i, shared_total_rewards, 
                shared_weights, test,))
        
        pool.close()
        pool.join()

        if test: return shared_weights[-1]

def simulate_single_country(country_index: int, shared_total_rewards: list, 
    shared_weights: list, test: bool):
    """ Simulate a single country. Executes the model for a single country
    for each epoch independently. At the end of the epoch, syncs with the other
    model to dictate best performance and update the weights.
    
    Arguments:
        country_index (int) -- The index of the country to simulate
        
        shared_total_rewards (list) -- Shared list of total rewards 
        
        shared_weights (list) -- Shared list of weights
        
        test (bool) -- Whether to test the model or not
        
    Returns:
        None
    """

    env = Environment()
    model = Model((5, 15), 7, 4)        # Only using the one country
    waiting_for_model = lambda rewards: [1 for reward in rewards if not reward]

    for i in tqdm(range(100)):
        
        env.reset()
        model.total_reward = 0

        while(not env.country_finished(4)):
            model.step(env)

        shared_total_rewards[country_index] = model.total_reward

        while waiting_for_model(shared_total_rewards): continue

        if np.argmax(shared_total_rewards) == country_index:
            shared_weights.append(model.model.get_weights())
            mlflow.log_metric('total_reward', model.total_reward)

        while len(shared_weights) <= i: continue

        model.update_target(shared_weights[-1])
        shared_total_rewards[country_index] = False
    
    if test:

        shared_total_rewards[country_index] = model.total_reward
        while waiting_for_model(shared_total_rewards): continue
        
        if np.argmax(shared_total_rewards) == country_index:
            mlflow.keras.log_model(model.model, "best_model")
            test_model(model, shared_weights)
            shared_total_rewards = mp.Manager().list()


def test_model(model: Model, shared_list: list):
    """ Test the model from a spawned parallel process

    Arguments:
        model (Model) -- The model to test

        shared_list (list) -- The list of weights to test

    Returns:
        None
    """

    env = Environment()
    env.reset()
    model.total_reward = 0

    while(not env.country_finished(4)):
        model.step(env)

    shared_list.append(model.total_reward)

def main():
    
    start = time.time()
    
    s = Simulator()
    test_reward = s.simulate_parallel(10, True)
    mlflow.log_metric('test_reward', test_reward)

    print(f"took: {time.time() - start}")

if __name__ == '__main__':
    main()

"""
Non threaded for 10: 241.09997153282166
Threaded for 10: 343.917240858078
Parallel for 10: 53.768577337265015
"""