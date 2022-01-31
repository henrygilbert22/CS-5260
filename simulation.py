from dataclasses import dataclass
from math import ceil
import pandas as pd
from rx import empty
import copy
import time
import multiprocessing as mp
import numpy as np
from more_itertools import chunked

from priority_queue import PriorityQueue, Solution
from transforms import HousingTransform, AlloyTransform, ElectronicTransform
from country import Country, ResourceWeights
from transfer import Transfer

# Schedule reward is differnce of inititial state value and final state value, this is the undiscounted reward
# Discounted schedule reward: gamma^N * (Q_end(c_i, s_j) – Q_start(c_i, s_j))
# We must use the expected utility of a scheudle which is the dicounted * the probility that it will actually hapen
    # This probobility is calculated by taking into account how the schedule affects the other countries, and how benificial it would be
    # for them
    
# Search algo can not be recursive

                       
class Simulation:
    
    countries: list[Country]
    r_weights: ResourceWeights
    
    countries_file_name: str
    weights_file_name: str
    
    country: Country
    frontier: PriorityQueue
    solutions: PriorityQueue
    
    depth: int
    gamma: float
    state_reduction: int
    
    def __init__(self, countries_file_name: str, weights_file_name: str, country: int, depth: int, gamma: float, state_reduction: int) -> None:
        
        self.countries_file_name = countries_file_name
        self.weights_file_name = weights_file_name
        self.depth = depth
        self.gamma = gamma
        self.state_reduction = state_reduction
        self.countries = {}
        self.frontier = PriorityQueue()
        self.solutions = PriorityQueue()
        
        self.load()
        
        self.country = self.countries[country]
        del self.countries[country]
    
    def load(self):
        
        self.load_countries(self.countries_file_name)
        self.load_weights(self.weights_file_name)
    
    def load_weights(self, file_name: str):
        
        df = pd.read_excel(file_name)
        args = pd.Series(df.Weight.values).to_list()
        
        self.r_weights = ResourceWeights(*args)        
           
           
    def load_countries(self, file_name: str):
    
        df = pd.read_excel(file_name)
        
        for index, row in df.iterrows(): 
            args = list(row.values) + [self.state_reduction]
            self.countries[args[0]] = Country(*args)
    
    def calculate_reward(self, new_state: Country, solution: Solution):
        
        curr_quality = new_state.state_value()
        og_quality = solution.path[0][1].state_value()
        
        # (Probobility * this) + (1-Probility)*C        C is negative function for cost of failure
        return pow(self.gamma, len(solution.path)+1) * (curr_quality - og_quality)          # Need to take into account other states when were trading
        
        
    # We need to limit the queue size for smaller computations
    
    def generate_transform_succesors(self, solution: Solution):
        
        curr_state = solution.path[-1][1]
        
        housing_scalers = curr_state.can_housing_transform()
        alloy_scalers = curr_state.can_alloys_transform()
        electronics_scalers = curr_state.can_electronics_transform()
        
        for scaler in housing_scalers:
            
            trans = HousingTransform(curr_state, scaler)
            new_state = curr_state.housing_transform(scaler)
            new_solution = Solution(self.calculate_reward(new_state, solution), solution.path + [[trans, new_state, self.countries]])        
            self.frontier.push(new_solution)
            
        for scaler in alloy_scalers:
            
            trans = AlloyTransform(curr_state, scaler)
            new_state = curr_state.alloys_transform(scaler)
            new_solution = Solution(self.calculate_reward(new_state, solution), solution.path + [[trans, new_state, self.countries]])        
            self.frontier.push(new_solution)
        
        for scaler in electronics_scalers:
            
            trans = ElectronicTransform(curr_state, scaler)
            new_state = curr_state.electronics_transform(scaler)
            new_solution = Solution(self.calculate_reward(new_state, solution), solution.path + [[trans, new_state, self.countries]])        
            self.frontier.push(new_solution)
        
    
    def generate_transfer_succesors(self, solution: Solution):
        
        curr_state = solution.path[-1][1]
        curr_countries = solution.path[-1][2]
        countries_elms = {}    
            
        curr_elms = {
                'metalic_elm': curr_state.metalic_elm,
                'timber': curr_state.timber,
                'metalic_alloys': curr_state.metalic_alloys,
                'electronics': curr_state.electronics,
                'housing': curr_state.housing
            }
        
        for c in curr_countries:
            countries_elms[c] = {
                'metalic_elm': curr_countries[c].metalic_elm,
                'timber': curr_countries[c].timber,
                'metalic_alloys': curr_countries[c].metalic_alloys,
                'electronics': curr_countries[c].electronics,
                'housing': curr_countries[c].housing
            }
        
        for c in countries_elms:
            for elm in countries_elms[c]:
                
                for curr_elm in curr_elms:
                    
                    amounts = []
                    poss_trades = [i+1 for i in range(min(countries_elms[c][elm], curr_elms[curr_elm]))]  
                    num_buckets = ceil(len(poss_trades) / self.state_reduction)                         # To push 0.1 to 1 as min is 1 bucket
                    
                    if num_buckets < 0 or len(poss_trades) == 0:
                        continue
                    
                    buckets = np.array_split(poss_trades, num_buckets)
                    for bucket in buckets:
                        if len(bucket) > 0:                  # Takes care if state_reduction is larger than starting buckets
                            amounts.append(int(sum(bucket)/len(bucket)))
                    
                    for amount_1 in amounts:
                        for amount_2 in amounts:
                        
                            trade = Transfer(elm, curr_elm, amount_1, amount_2, c, curr_state.name)
                            new_curr_state = curr_state.make_trade(curr_elm, amount_2)
                            new_countries = copy.deepcopy(curr_countries)
                            new_countries[c] = curr_countries[c].make_trade(elm, amount_1)
                            new_solution = Solution(self.calculate_reward(new_curr_state, solution), solution.path + [[trade, new_curr_state, new_countries]]) 
                            self.frontier.push(new_solution)
                
    def generate_succesors(self, solution: Solution):           # Paralize this for extra credit
                
        self.generate_transform_succesors(solution)
        self.generate_transfer_succesors(solution)
    
    def parallel_generate_succesors(self, state: Country, solution: Solution):
        
        housing_scalers = state.can_housing_transform()
        alloy_scalers = state.can_alloys_transform()
        electronics_scalers = state.can_electronics_transform()
        
        housing_chunks = []
        for i in range(0, len(housing_scalers), int(len(housing_scalers)/3)):
            housing_chunks.append(housing_scalers[i:i+int(len(housing_scalers)/3)])
        
        allow_chunks = []
        for i in range(0, len(alloy_scalers), int(len(alloy_scalers)/3)):
            allow_chunks.append(alloy_scalers[i:i+int(len(alloy_scalers)/3)])
            
        electronic_chunks = []
        for i in range(0, len(electronics_scalers), int(len(electronics_scalers)/3)):
            electronic_chunks.append(electronics_scalers[i:i+int(len(electronics_scalers)/3)])
            
        pool = mp.Pool
        
    def search(self):
        
        initial_solution = Solution(self.country.state_value(), [[None, self.country, self.countries]])
        self.frontier.push(initial_solution)
        
        while not self.frontier.empty():
                        
            solution = self.frontier.pop()
            
            if len(solution.path) > self.depth:
                self.solutions.push(solution)
                continue
            
            self.generate_succesors(solution)
            


def generate_new_states(scalers: list, transform_type: str, solution: Solution, shared_list: list, state: Country):

        local_solutions = []
        
        for scaler in scalers:
             
            if transform_type == 'housing':
                trans = HousingTransform(state, scaler)
                new_state = state.housing_transform(scaler)
                new_solution = Solution(new_state.state_value(), solution.path + [[trans, new_state]])
                local_solutions.append(new_solution)
            
            elif transform_type == 'alloy':
                trans = AlloyTransform(state, scaler)
                new_state = state.alloys_transform(scaler)
                new_solution = Solution(new_state.state_value(), solution.path + [[trans, new_state]])
                local_solutions.append(new_solution)
            
            elif transform_type == 'electronic':
                trans = ElectronicTransform(state, scaler)
                new_state = state.electronics_transform(scaler)
                new_solution = Solution(new_state.state_value(), solution.path + [[trans, new_state]])
                local_solutions.append(new_solution)
                
        
        shared_list += local_solutions
                 
            
# With depth of 1, country 4 took 0.0017349720001220703 - 70 solutions
# With depth of 2, country 4 took 0.6941020488739014 - 4900 solutions
    # Time scale = 400x longer for each depth
    # State space scale = 70x more states
# Depth of 3, country 4 took 2756.87561917305 - 343000 solutions
    # Time scale = 3900x longer
    # State space scale = 70x

        
def main():
    
    s = Simulation('Example-Initial-Countries.xlsx', 'Example-Sample-Resources.xlsx', 'Erewhon', 1, 0.8, 2)
    
    start = time.time()
    s.search()
    end = time.time()
    
    print(f"Took: {end-start}")
    
    print(len(s.solutions.priority_queue))
    best_solution = s.solutions.pop()
    print(best_solution)
    

if __name__ == '__main__':
    main()