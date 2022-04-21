from cProfile import label
from environment import Environment
from model import Model

from collections import deque
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
#import silence_tensorflow.auto
import atexit

class Simulator:

    env: Environment
    model: Model

    def __init__(self) -> None:
        
        self.env = Environment()
        self.model = Model(15,7)

        self.configure_mlflow()

    def configure_mlflow(self):

        #notes = input("Enter a note: ")
        notes = 'test'
        if notes != "x":
            
            print("in here")
            self.log = True
            mlflow.set_tracking_uri("http://10.0.0.206:5000")
            mlflow.set_experiment('Deep-Q-Learning')
            mlflow.start_run(run_name="Henry G") 
            mlflow.tensorflow.autolog() 
            MlflowClient().set_tag(mlflow.active_run().info.run_id, "mlflow.note.content", notes)
            
            mlflow.log_artifact(f'simulator.py')
            mlflow.log_artifact(f'model.py')
            mlflow.log_artifact(f'environment.py')

    def graph_performance(self, results: list, averages: list, actions):

        x = [i for i in range(len(results))]
        plt.plot(x, results, label='Reward')
        plt.plot(x, averages, label="Average")

        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.savefig('graph.png')
        plt.clf()

        data = {}
        for action in actions:
            if action in data:
                data[action] += 1
            else:
                data[action] = 1

        action = list(data.keys())
        values = list(data.values())
        
        # creating the bar plot
        plt.bar(action, values)
        plt.savefig('bar.png')
        
    def simulate(self):

        epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
        max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
        episode = 0
        replay_memory = deque(maxlen=10_000)
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

                    #current_reshaped = np.array(current_state).reshape([1, np.array(current_state).shape[0]])
                    predicted = self.model.model.predict(self.model.create_dataset([current_state])).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
                    action = np.argmax(predicted) 
                    action += 1
                
                actions.append(action)

                reward, done = self.env.step(action)      # Executing action on current state and getting reward, this also increments out current state
                new_state = self.env.current_state()               
                replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory
                total_reward += reward

                if steps_to_update_target_model % 20 == 0 or done:                   # If we've done 4 steps or have lost/won, update the main neural net, not target

                    loss, accuracy = self.model.train(replay_memory, done)            # training the main model
                    mlflow.log_metric("loss", loss)
                    mlflow.log_metric("accuracy", accuracy)

            print(f'Made it to: {total_reward} - {epsilon}')
            mlflow.log_metric("reward", total_reward)
            mlflow.log_metric("epsilon", epsilon)

            episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-0.05 * episode)
            self.model.update_target()
            
            
def main():

    s = Simulator()
    s.simulate()

    atexit.register(s.strategy._extended._collective_ops._pool.close) # type: ignore

if __name__ == '__main__':
    main()