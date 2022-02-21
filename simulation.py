from dataclasses import dataclass
from math import ceil
import pandas as pd
from rx import empty
import copy
import time
import multiprocessing as mp
import numpy as np
from more_itertools import chunked
import math
import os

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
    max_frontier_size: int
    gamma: float
    state_reduction: int
    
    def __init__(self, countries_file_name: str, weights_file_name: str, country: int, depth: int, gamma: float, state_reduction: int, max_frontier_size: int) -> None:
        
        self.countries_file_name = countries_file_name
        self.weights_file_name = weights_file_name
        self.depth = depth
        self.max_frontier_size = max_frontier_size
        self.gamma = gamma
        self.state_reduction = state_reduction
        self.countries = {}
        self.frontier = PriorityQueue(max_frontier_size)
        self.solutions = PriorityQueue(3)
        
        self.load()
        
        self.country = self.countries[country]
        del self.countries[country]
    
    def load(self):
        
        self.load_countries(self.countries_file_name)
        self.load_weights(self.weights_file_name)
    
    def load_weights(self, file_name: str):
        
        df = pd.read_csv(file_name)
        args = pd.Series(df.Weight.values).to_list()
        self.r_weights = ResourceWeights(*args)        
                  
    def load_countries(self, file_name: str):
    
        df = pd.read_excel(file_name)
        
        for index, row in df.iterrows(): 
            args = list(row.values) + [self.state_reduction]
            self.countries[args[0]] = Country(*args)
    
    def calculate_reward(self, solution: Solution):
        
        new_state = solution.path[-1][1]
        curr_quality = new_state.state_value()
        og_quality = solution.path[0][1].state_value()
        
        other_country_probobility = []
        for step in solution.path:
            
            if type(step[0]) is Transfer:
                other_c_utility = self.countries[step[0].c_1_name].state_value()
                other_country_probobility.append(math.log(other_c_utility))
        
        if other_country_probobility:
            other_c_prob = sum(other_country_probobility) / len(other_country_probobility)
        else:
            other_c_prob = 1
            
        discounted_reward =  round(pow(self.gamma, len(solution.path)+1) * (curr_quality - og_quality), 3)        # Need to take into account other states when were trading
        expected_utility = (other_c_prob * discounted_reward) + ((1 - other_c_prob) * 0.1)      # 0.1 encourages it to take risks and not give too much weight to the probobility
        
        return expected_utility
        
    def generate_transform_succesors(self, solution: Solution):
        
        curr_state = solution.path[-1][1]
        
        housing_scalers = curr_state.can_housing_transform()
        alloy_scalers = curr_state.can_alloys_transform()
        electronics_scalers = curr_state.can_electronics_transform()
        
        for scaler in housing_scalers:
            
            trans = HousingTransform(curr_state, scaler)
            new_state = curr_state.housing_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, self.countries]]
            new_solution.priority = self.calculate_reward(new_solution)       
            self.frontier.push(new_solution)
            
        for scaler in alloy_scalers:
            
            trans = AlloyTransform(curr_state, scaler)
            new_state = curr_state.alloys_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, self.countries]]
            new_solution.priority = self.calculate_reward(new_solution)       
            self.frontier.push(new_solution)
        
        for scaler in electronics_scalers:
            
            trans = ElectronicTransform(curr_state, scaler)
            new_state = curr_state.electronics_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, self.countries]]
            new_solution.priority = self.calculate_reward(new_solution)       
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
                    
                    other_elm_scale = 1 / self.r_weights[elm]               #0.2 - 5 are needed to be $1
                    self_elm_scale = 1 / self.r_weights[curr_elm]    #0.5 - 2 are needed to be $1
                    max_amount = min(int(countries_elms[c][elm]/other_elm_scale), int(curr_elms[curr_elm]/self_elm_scale))      # 500/5 = 100, 100 / 0.5 = 200, max swap is 100 (equivalent value)
                    
                    poss_trades = [i+1 for i in range(max_amount)]
                    num_buckets = ceil(len(poss_trades) / self.state_reduction)
                    
                    if num_buckets < 1 or len(poss_trades) == 0:
                        continue
                    
                    amounts = []
                    buckets = np.array_split(poss_trades, num_buckets)
                    for bucket in buckets:
                        if len(bucket) > 0:                  # Takes care if state_reduction is larger than starting buckets
                            amounts.append(int(sum(bucket)/len(bucket)))
                    
                    for amount in amounts:
                        
                        other_elm_amount = ceil(amount / other_elm_scale)
                        self_elm_amount = ceil(amount / self_elm_scale)
                        
                        trade = Transfer(elm, curr_elm, other_elm_amount, self_elm_amount, c, curr_state.name)
                        new_curr_state = curr_state.make_trade(curr_elm, self_elm_amount)
                        new_countries = copy.deepcopy(curr_countries)
                        new_countries[c] = curr_countries[c].make_trade(elm, other_elm_amount)
                        new_solution = copy.deepcopy(solution)
                        new_solution.path += [[trade, new_curr_state, new_countries]]
                        new_solution.priority = self.calculate_reward(new_solution)       
                        self.frontier.push(new_solution)
                          
    def generate_succesors(self, solution: Solution):           # Paralize this for extra credit
                
        self.generate_transform_succesors(solution)
        self.generate_transfer_succesors(solution)
     
    def search(self):
        
        total = 0
        initial_solution = Solution(self.country.state_value(), [[None, self.country, self.countries]])
        self.frontier.push(initial_solution)
        
        while not self.frontier.empty():
            
            if len(self.frontier.queue) == self.max_frontier_size:
            #if False:
                
                shared_frontier = mp.Manager().list()
                pool = mp.Pool()
                chunks = np.array_split(np.array(self.frontier.queue), os.cpu_count())
                self.frontier.queue = []
                
                for chunk in chunks:
                    pool.apply_async(func=generate_succesors_parallel, args=(chunk, self.countries, shared_frontier, self.gamma, self.r_weights, self.state_reduction, ))
                
                pool.close()
                pool.join()
                
                total += len(shared_frontier)
                for sol in shared_frontier:
                    
                    if len(sol.path) > self.depth:
                        self.solutions.push(sol)
                    else:
                       # print("Shouldn't be in here")
                        self.frontier.push(sol)
            
            else:
                
                solution = self.frontier.pop()
                
                total += 1
                
                if len(solution.path) > self.depth:
                    self.solutions.push(solution)
                    continue
                
                self.generate_succesors(solution)
        
        print(total)
        

def calculate_reward_parallel(solution: Solution, countries: dict, gamma: int):
        
        new_state = solution.path[-1][1]
        curr_quality = new_state.state_value()
        og_quality = solution.path[0][1].state_value()
        
        other_country_probobility = []
        for step in solution.path:
            
            if type(step[0]) is Transfer:
                other_c_utility = countries[step[0].c_1_name].state_value()
                other_country_probobility.append(math.log(other_c_utility))
        
        if other_country_probobility:
            other_c_prob = sum(other_country_probobility) / len(other_country_probobility)
        else:
            other_c_prob = 1
            
        discounted_reward =  round(pow(gamma, len(solution.path)+1) * (curr_quality - og_quality), 3)        # Need to take into account other states when were trading
        expected_utility = (other_c_prob * discounted_reward) + ((1 - other_c_prob) * 0.1)      # 0.1 encourages it to take risks and not give too much weight to the probobility
        
        return expected_utility
                 
def generate_transfer_succesors_parallel(solution: Solution, r_weights: ResourceWeights, countries: dict, state_reduction: int, shared_frontier: list, gamma: int):
        
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
                
                other_elm_scale = 1 / r_weights[elm]               #0.2 - 5 are needed to be $1
                self_elm_scale = 1 / r_weights[curr_elm]    #0.5 - 2 are needed to be $1
                max_amount = min(int(countries_elms[c][elm]/other_elm_scale), int(curr_elms[curr_elm]/self_elm_scale))      # 500/5 = 100, 100 / 0.5 = 200, max swap is 100 (equivalent value)
                
                poss_trades = [i+1 for i in range(max_amount)]
                num_buckets = ceil(len(poss_trades) / state_reduction)
                
                if num_buckets < 1 or len(poss_trades) == 0:
                    continue
                
                amounts = []
                buckets = np.array_split(poss_trades, num_buckets)
                for bucket in buckets:
                    if len(bucket) > 0:                  # Takes care if state_reduction is larger than starting buckets
                        amounts.append(int(sum(bucket)/len(bucket)))
                
                for amount in amounts:
                    
                    other_elm_amount = ceil(amount / other_elm_scale)
                    self_elm_amount = ceil(amount / self_elm_scale)
                    
                    trade = Transfer(elm, curr_elm, other_elm_amount, self_elm_amount, c, curr_state.name)
                    new_curr_state = curr_state.make_trade(curr_elm, self_elm_amount)
                    new_countries = copy.deepcopy(curr_countries)
                    new_countries[c] = curr_countries[c].make_trade(elm, other_elm_amount)
                    new_solution = copy.deepcopy(solution)
                    new_solution.path += [[trade, new_curr_state, new_countries]]
                    new_solution.priority = calculate_reward_parallel(new_solution, countries, gamma)       
                    shared_frontier.append(new_solution)
                                          
def generate_transform_succesors_parallel(solution: Solution, countries: dict, shared_frontier: list, gamma: int):
        
        curr_state = solution.path[-1][1]
        
        housing_scalers = curr_state.can_housing_transform()
        alloy_scalers = curr_state.can_alloys_transform()
        electronics_scalers = curr_state.can_electronics_transform()
        
        for scaler in housing_scalers:
            
            trans = HousingTransform(curr_state, scaler)
            new_state = curr_state.housing_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, countries]]
            new_solution.priority = calculate_reward_parallel(new_solution, countries, gamma)       
            shared_frontier.append(new_solution)
            
        for scaler in alloy_scalers:
            
            trans = AlloyTransform(curr_state, scaler)
            new_state = curr_state.alloys_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, countries]]
            new_solution.priority = calculate_reward_parallel(new_solution, countries, gamma)     
            shared_frontier.append(new_solution)
        
        for scaler in electronics_scalers:
            
            trans = ElectronicTransform(curr_state, scaler)
            new_state = curr_state.electronics_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, countries]]
            new_solution.priority = calculate_reward_parallel(new_solution, countries, gamma)       
            shared_frontier.append(new_solution)
                      
def generate_succesors_parallel(chunk: Solution, countries: dict, shared_frontier: list, gamma: int, r_weights: ResourceWeights, state_reduction: int):
    
    for solution in chunk:
        generate_transform_succesors_parallel(solution, countries, shared_frontier, gamma)
        generate_transfer_succesors_parallel(solution, r_weights, countries, state_reduction, shared_frontier, gamma)

   
   
   
   
"""
    Paralization Metrics
        2 - 0.8 - 10 - 10000: 2.66s
        3 - 0.8 - 10 - 1000: 59.57s
            total: 176032 - 0.00034 per state
    
    Sequential Metrics
        2 - 0.8 - 10 - 10000: 1.82s
        3 - 0.8 - 10 - 1000: 11s
            total: 22418 - 0.00049 per state
"""

   
   
   
   
   
   
   
   
   
   
# With depth of 1, country 4 took 0.0017349720001220703 - 70 solutions
# With depth of 2, country 4 took 0.6941020488739014 - 4900 solutions
    # Time scale = 400x longer for each depth
    # State space scale = 70x more states
# Depth of 3, country 4 took 2756.87561917305 - 343000 solutions
    # Time scale = 3900x longer
    # State space scale = 70x


def main():
    
    s = Simulation('Example-Initial-Countries.xlsx', 'resources.csv', 'Erewhon', 3, 0.8, 10, 1000)
    
    start = time.time()
    s.search()
    end = time.time()
    
    print(f"Took: {end-start}")
    
    best_solution = s.solutions.queue[-1]
    
    best_solution.print()
 
    

if __name__ == '__main__':
    main()