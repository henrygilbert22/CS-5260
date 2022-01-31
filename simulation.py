from dataclasses import dataclass
import pandas as pd
from rx import empty
import copy
import time
import multiprocessing as mp
import numpy as np
from more_itertools import chunked

from priority_queue import PriorityQueue
from transforms import HousingTransform, AlloyTransform, ElectronicTransform
from country import Country, ResourceWeights


# Schedule reward is differnce of inititial state value and final state value, this is the undiscounted reward
# Discounted schedule reward: gamma^N * (Q_end(c_i, s_j) – Q_start(c_i, s_j))
# We must use the expected utility of a scheudle which is the dicounted * the probility that it will actually hapen
    # This probobility is calculated by taking into account how the schedule affects the other countries, and how benificial it would be
    # for them
    
# Search algo can not be recursive


@dataclass
class Solution:
    
    expected_utility: float
    path: list
    
                          
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
        self.countries = []
        self.frontier = PriorityQueue()
        self.solutions = PriorityQueue()
        
        self.load()
        
        self.country = self.countries[country]
    
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
            self.countries.append(Country(*args))
    
    def calculate_reward(self, new_state: Country, solution: Solution):
        
        curr_quality = new_state.state_value()
        og_quality = solution.path[0][1].state_value()
        
        # (Probobility * this) + (1-Probility)*C        C is negative function for cost of failure
        return pow(self.gamma, len(solution.path)+1) * (curr_quality - og_quality)
        
        
    # We need to limit the queue size for smaller computations
    
    def generate_succesors(self, state: Country, solution: Solution):           # Paralize this for extra credit
                
        housing_scalers = state.can_housing_transform()
        alloy_scalers = state.can_alloys_transform()
        electronics_scalers = state.can_electronics_transform()
        
        for scaler in housing_scalers:
            
            trans = HousingTransform(state, scaler)
            new_state = state.housing_transform(scaler)
            new_solution = Solution(self.calculate_reward(new_state, solution), solution.path + [[trans, new_state]])        
            self.frontier.push(new_solution)
            
        for scaler in alloy_scalers:
            
            trans = AlloyTransform(state, scaler)
            new_state = state.alloys_transform(scaler)
            new_solution = Solution(self.calculate_reward(new_state, solution), solution.path + [[trans, new_state]])        
            self.frontier.push(new_solution)
        
        for scaler in electronics_scalers:
            
            trans = ElectronicTransform(state, scaler)
            new_state = state.electronics_transform(scaler)
            new_solution = Solution(self.calculate_reward(new_state, solution), solution.path + [[trans, new_state]])        
            self.frontier.push(new_solution)
    
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
        
        initial_solution = Solution(self.country.state_value(), [[None, self.country]])
        self.frontier.push(initial_solution)
        
        while not self.frontier.empty():
                        
            solution = self.frontier.pop()
            
            if len(solution.path) > self.depth:
                self.solutions.push(solution)
                continue
            
            self.generate_succesors(solution.path[-1][1], solution)
            


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
    
    s = Simulation('Example-Initial-Countries.xlsx', 'Example-Sample-Resources.xlsx', 4, 1, 0.8, 2)
    
    start = time.time()
    s.search()
    end = time.time()
    
    print(f"Took: {end-start}")
    
    print(len(s.solutions.priority_queue))
    best_solution = s.solutions.pop()
    print(best_solution)
    

if __name__ == '__main__':
    main()