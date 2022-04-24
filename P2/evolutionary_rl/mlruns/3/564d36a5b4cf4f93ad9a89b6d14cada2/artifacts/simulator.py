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
        
        self.env = Environment()
        self.models = []

        self.configure_mlflow()

    def configure_mlflow(self):

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

    def simulate(self):

        [self.models.append(Model((5, 15), 7, i)) for i in range(5)]
        most_successful_models = []

        for i in tqdm(range(10)):
            
            self.env.reset()

            while(not self.env.all_countries_finished()):
                
                training_threads = []
                for model in self.models:
                    training_threads.append(model.step(self.env))

                if training_threads[0]:
                    [t.join() for t in training_threads]
                        
            total_rewards = [self.models[i].total_reward for i in range(len(self.models))]
            most_successful_model = np.argmax(total_rewards)
            most_successful_models.append(most_successful_model)
            print(f'Most successful model: {most_successful_model}')
            
            for i in range(len(self.models)):
                self.models[i].total_reward = 0
                self.models[i].update_target(self.models[most_successful_model].model)

        print(f'Most successful models: {most_successful_models}')
        return most_successful_models[-1]
                
    def test(self, best_model: int):

        actions = []
        rewards = []
          
        done = False
        steps_to_update_target_model = 0
        self.env.reset()

        total_reward = 0

        while(not done):

            current_state = self.env.current_state()
                
            current_reshaped = np.array(current_state).reshape([1, 5, 15])
            predicted = self.model.model(current_reshaped).numpy()[0]          # Predicting best action, not sure why flatten (pushing 2d into 1d)
            action = np.argmax(predicted) 
        
            actions.append(action)
            reward, done = self.env.step(action)      # Executing action on current state and getting reward, this also increments out current state
            total_reward += reward

        
        print(f'Made it to: {total_reward}')
        rewards.append(total_reward)

        mlflow.log_metric("avg_test_reward", sum(rewards)/len(rewards))
        self.graph_action_distribution(actions, "test")

    def simulate_parallel(self, replicas: int, test: bool = False):

        pool = mp.Pool(processes=replicas)
        shared_total_rewards = mp.Manager().list()
        shared_weights = mp.Manager().list()

        for i in range(replicas):
            shared_total_rewards.append(False)
            pool.apply_async(simulate_single_country, args=(i, shared_total_rewards, shared_weights, test,))
        
        pool.close()
        pool.join()

        if test: return shared_weights[-1]

def simulate_single_country(country_index: int, shared_total_rewards: list, shared_weights: list, test: bool):

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
                test_model(model, shared_weights)
                shared_total_rewards = mp.Manager().list()


def test_model(model: Model, shared_list: list):

    env = Environment()
    env.reset()
    model.total_reward = 0

    while(not env.country_finished(4)):
        model.test_step(env)

    shared_list.append(model.total_reward)

def temp(results: list):

    s = Simulator()
    results.append(s.simulate_parallel(True))

def test(num_tests: int):

    results = mp.Manager().list()
    processes = []

    for i in range(num_tests):
        p = mp.Process(target=temp, args=(results,))
        p.daemon = False
        processes.append(p)
        processes[-1].start()

    [p.join() for p in processes]

    with open(f'test/{0}.txt', 'w') as f:
        f.write(f'Made it to: {sum(results)/len(results)}')

def run():

    start = time.time()
    s = Simulator()
    print(s.simulate_parallel(10, True))
    print(f"took: {time.time() - start}")

def main():
    
    run()
    #test(1)

if __name__ == '__main__':
    main()

"""
Non threaded for 10: 241.09997153282166
Threaded for 10: 343.917240858078
Parallel for 10: 53.768577337265015
"""