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

        for i in tqdm(range(100)):
            
            self.env.reset()

            while(not self.env.all_countries_finished()):
                
                for model in self.models:
                    model.step(self.env)

            total_rewards = [self.models[i].total_reward for i in range(len(self.models))]
            most_successful_model = np.argmax(total_rewards)
            print(f'Most successful model: {most_successful_model}')
            
            for i in range(len(self.models)):
                self.models[i].total_reward = 0
                self.models[i].update_target(self.models[most_successful_model].model)


                
    def test(self):

        actions = []
        rewards = []
          
        done = False
        steps_to_update_target_model = 0
        self.env.reset()

        total_reward = 0

        while(not done):

            steps_to_update_target_model += 1 
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


        
            
            
def main():

    s = Simulator()
    s.simulate()

if __name__ == '__main__':
    main()