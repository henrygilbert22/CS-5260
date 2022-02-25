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
from typing import List
import uuid

from priority_queue import PriorityQueue, Solution
from transforms import FarmTransform, FoodTransform, HousingTransform, AlloyTransform, ElectronicTransform
from country import Country, ResourceWeights
from transfer import Transfer

class Simulation:

    countries: List[Country]
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
    C: int

    def __init__(self, countries_file_name: str, weights_file_name: str, 
                 country: int, depth: int, gamma: float, state_reduction: int, 
                 max_frontier_size: int, solution_size: int, C: int) -> None:
        """Initialization function for Simulation class

        Args:
            countries_file_name (str): File name of countries file
            weights_file_name (str): File name of weights file
            country (int): Starting country
            depth (int): Depth of search
            gamma (float): Given gamma hyperparameter
            state_reduction (int): Given state reduction hyperparameter
            max_frontier_size (int): Given max frontier size hyperparameter
            
        Returns:
            None
        """
        self.countries_file_name = countries_file_name
        self.weights_file_name = weights_file_name
        self.depth = depth
        self.max_frontier_size = max_frontier_size
        self.gamma = gamma
        self.state_reduction = state_reduction
        self.countries = {}
        self.frontier = PriorityQueue(max_frontier_size)
        self.solutions = PriorityQueue(solution_size)
        self.C = C

        self.load()

        self.country = self.countries[country]
        del self.countries[country]

    def load(self):
        """Calls functions to load both countries and
        resource weights into class
        
        Arguements:
            None
            
        Returns:
            None
        """

        self.load_weights(self.weights_file_name)
        self.load_countries(self.countries_file_name)
        
    def load_weights(self, file_name: str):
        """Loads the weights of resources from the given file

        Args:
            file_name (str): File containing weights of resources
            
        Returns:
            None
        """

        df = pd.read_excel(file_name)
        args = pd.Series(df.Weight.values).to_list()
        self.r_weights = ResourceWeights(*args)

    def load_countries(self, file_name: str):
        """Loads the countries from the given csv file

        Parameters:
            file_name (str): File name for the csv holding the countries
            
        Returns:
            None
        """

        df = pd.read_excel(file_name)

        for index, row in df.iterrows():
            args = list(row.values) + [self.state_reduction] + [self.r_weights]
            self.countries[args[0]] = Country(*args)

    def calculate_reward(self, solution: Solution):
        """Given the current solution, calculate the state 
        quality, the undiscounted reward, the discounted reward,
        the probility a country accepts said transform (if applicable)
        The probility of success given self parameter c, and finally,
        the expected utility given the function defined in class.
        
        Args:
            solution (Solution): The current solution
            
        Returns:
            EU (float): Expected Utility of the state
        """
        
        new_state = solution.path[-1][1]   
        curr_quality = new_state.state_value()
        og_quality = solution.path[-2][1].state_value()

        other_country_probobility = []
        for step in solution.path:

            if type(step[0]) is Transfer:
                other_c_utility = self.countries[step[0].c_1_name].state_value()
                other_country_probobility.append(math.log(other_c_utility))

        if other_country_probobility:
            other_c_prob = sum(other_country_probobility) / len(other_country_probobility)
        
        else:
            other_c_prob = 1
        
        discounted_reward = round(pow(self.gamma, len(solution.path)+1) * (curr_quality - og_quality), 3)
        expected_utility = (other_c_prob * discounted_reward) + ((1 - other_c_prob) * self.C)

        solution.path[-1] += [expected_utility]
        solution.priority = expected_utility
        
    def generate_transform_succesors(self, solution: Solution):
        """Given the current solution, computes all possible transforms 
        and the resulting states. Stores new states and corresponding
        transforms in the frontier.

        Args:
            solution (Solution): Current solution
            
        Returns:
            None
        """

        curr_state = solution.path[-1][1]

        housing_scalers = curr_state.can_housing_transform()
        alloy_scalers = curr_state.can_alloys_transform()
        electronics_scalers = curr_state.can_electronics_transform()
        food_scalers = curr_state.can_food_transform()
        farm_scalers = curr_state.can_farm_transform()

        for scaler in housing_scalers:

            trans = HousingTransform(scaler)
            new_state = curr_state.housing_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, self.countries]]
            self.calculate_reward(new_solution)
            self.frontier.push(new_solution)

        for scaler in alloy_scalers:

            trans = AlloyTransform(scaler)
            new_state = curr_state.alloys_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, self.countries]]
            self.calculate_reward(new_solution)
            self.frontier.push(new_solution)

        for scaler in electronics_scalers:

            trans = ElectronicTransform(scaler)
            new_state = curr_state.electronics_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, self.countries]]
            self.calculate_reward(new_solution)
            self.frontier.push(new_solution)
        
        for scaler in food_scalers:

            trans = FoodTransform(scaler)
            new_state = curr_state.food_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, self.countries]]
            self.calculate_reward(new_solution)
            self.frontier.push(new_solution)
    
        for scaler in farm_scalers:

            trans = FarmTransform(scaler)
            new_state = curr_state.farm_transform(scaler)
            new_solution = copy.deepcopy(solution)
            new_solution.path += [[trans, new_state, self.countries]]
            self.calculate_reward(new_solution)
            self.frontier.push(new_solution)

    def generate_transfer_succesors(self, solution: Solution):
        """Given the current solution, computes all equal trades
        between all countries and all resources. Stores corresponding
        new states and transfer in the given frontier.

        Args:
            solution (Solution): Current solution
            
        Returns:
            None
        """

        curr_state = solution.path[-1][1]
        curr_countries = solution.path[-1][2]
        countries_elms = {}

        curr_elms = {
            'metalic_elm': curr_state.metalic_elm,
            'timber': curr_state.timber,
            'available_land': curr_state.available_land,
            'water': curr_state.water,
        }

        for c in curr_countries:
            countries_elms[c] = {
                'metalic_elm': curr_countries[c].metalic_elm,
                'timber': curr_countries[c].timber,
                'available_land': curr_state.available_land,
                'water': curr_state.water,
            }

        for c in countries_elms:
            for elm in countries_elms[c]:

                for curr_elm in curr_elms:
                    
                    if curr_elm == elm:     # Skipping to avoid redundant trades
                        continue

                    other_elm_scale = 1 / self.r_weights[curr_elm]
                    self_elm_scale = 1 / self.r_weights[elm] 
                    max_amount = min(int(countries_elms[c][elm]/other_elm_scale), int(curr_elms[curr_elm]/self_elm_scale))
                    
                    if max_amount <= 0:
                        continue
                    
                    if self.state_reduction == -1:       # Ultimate reduction
                        
                        other_elm_amount = ceil(max_amount / other_elm_scale)
                        self_elm_amount = ceil(max_amount / self_elm_scale)

                        trade = Transfer(elm, curr_elm, other_elm_amount, self_elm_amount, c, curr_state.name)
                        new_curr_state = curr_state.make_trade(curr_elm, self_elm_amount)
                        new_countries = copy.deepcopy(curr_countries)
                        new_countries[c] = curr_countries[c].make_trade(elm, other_elm_amount)
                        new_solution = copy.deepcopy(solution)
                        new_solution.path += [[trade, new_curr_state, new_countries]]
                        self.calculate_reward(new_solution)
                        self.frontier.push(new_solution)
                    
                    else:
    
                        poss_trades = [i+1 for i in range(max_amount)]
                        num_buckets = ceil(len(poss_trades) / self.state_reduction)

                        if num_buckets < 1 or len(poss_trades) == 0:
                            continue

                        amounts = []
                        buckets = np.array_split(poss_trades, num_buckets)
                        
                        for bucket in buckets:
                            if len(bucket) > 0:
                                amounts.append(int(sum(bucket)/len(bucket)))

                        for amount in amounts:

                            other_elm_amount = ceil(amount / other_elm_scale)
                            self_elm_amount = ceil(amount / self_elm_scale)

                            trade = Transfer(
                                elm, curr_elm, other_elm_amount, self_elm_amount, c, curr_state.name)
                            new_curr_state = curr_state.make_trade(
                                curr_elm, self_elm_amount)
                            new_countries = copy.deepcopy(curr_countries)
                            new_countries[c] = curr_countries[c].make_trade(elm, other_elm_amount)
                            new_solution = copy.deepcopy(solution)
                            new_solution.path += [[trade, new_curr_state, new_countries]]
                            self.calculate_reward(new_solution)
                            self.frontier.push(new_solution)

    def generate_succesors(self, solution: Solution):
        """Given the current solution, calls functions
        to generate next transform states and transfer states.

        Args:
            solution (Solution): Current solution
        
        Returns:
            None
        """

        self.generate_transform_succesors(solution)
        self.generate_transfer_succesors(solution)

    def search(self):
        """ This is the generic anytime, forward searching, depth-bound,
        generic utility driven scheduler as outlined in the slides. Given a new state,
        all possible next states are computed and then sorted based on EU. The highest
        state is poped from the queue until the queue is empty and only solutions remain.
        
        Arguements:
            None
            
        Returns:
            None
        """

        total = 1
        initial_solution = Solution(self.country.state_value(), [[None, self.country, self.countries, 0]])
        self.frontier.push(initial_solution)

        while not self.frontier.empty():

            total += 1
            solution = self.frontier.pop()

            if len(solution.path) > self.depth:
                self.solutions.push(solution)
                continue

            self.generate_succesors(solution)
        
        print(f'Total States: {total}')

    def search_parallel(self) -> None:
        """This function searches the possible state space in parallel,
        utilizing all possible cores on a given machine. Once the number
        of searchable states hits the infliction point, they are chunked,
        and parallized, where beam search is then used to search to the
        given depth. The results are returned and sorted in the priority queue.
        
        Arguements:
            None
            
        Returns:
            None
        """

        total = 1
        initial_solution = Solution(self.country.state_value(), [[None, self.country, self.countries, 0]])
        self.frontier.push(initial_solution)
        seg_num = int(self.max_frontier_size/os.cpu_count())

        while not self.frontier.empty():
            
            if len(self.frontier.queue) >= seg_num:      # Maybe we only need the top 10?
                
                print("starting parallel")
                
                shared_frontier = mp.Manager().list()
                pool = mp.Pool()
                chunks = np.array_split(np.array(self.frontier.queue), os.cpu_count())
                self.frontier.queue = []

                for chunk in chunks:
                    pool.apply_async(
                        func=generate_succesors_parallel, 
                        args=(chunk, self.countries, 
                              shared_frontier, self.gamma, 
                              self.r_weights, self.state_reduction, 
                              self.depth, seg_num, self.C)
                        )
                    
                pool.close()
                pool.join()
                
                print("out of parallel")

                total += len(shared_frontier)
                for sol in shared_frontier:

                    if len(sol.path) > self.depth:
                        self.solutions.push(sol)
                    else:
                        print("Shouldn't be in here")
                        self.frontier.push(sol)

            else:

                solution = self.frontier.pop()
                total += 1

                if len(solution.path) > self.depth:
                    self.solutions.push(solution)
                    continue

                self.generate_succesors(solution)
            
        print(f'Total States: {total}')


def calculate_reward_parallel(solution: Solution, countries: dict, gamma: float, C: float):
    """Function to calculate reward of a given state. Given the current solution, 
        calculate the state quality, the undiscounted reward, the discounted reward,
        the probility a country accepts said transform (if applicable)
        The probility of success given self parameter c, and finally,
        the expected utility given the function defined in class.

    Args:
        solution (Solution): Current solution
        countries (dict): Current countries of simulation
        gamma (int): Current gamma of simulation

    Returns:
        EU (float): Expected Utility of state
    """

    new_state = solution.path[-1][1]
    curr_quality = new_state.state_value()
    og_quality = solution.path[-2][1].state_value()
    
    other_country_probobility = []
    for step in solution.path:

        if type(step[0]) is Transfer:
            other_c_utility = countries[step[0].c_1_name].state_value()
            other_country_probobility.append(math.log(other_c_utility))

    if other_country_probobility:
        other_c_prob = sum(other_country_probobility) / len(other_country_probobility)
    
    else:
        other_c_prob = 1

    discounted_reward = round(
        pow(gamma, len(solution.path)+1) * (curr_quality - og_quality), 3)

    expected_utility = (other_c_prob * discounted_reward) + ((1 - other_c_prob) * C)

    solution.path[-1] += [expected_utility]
    solution.priority = expected_utility

def generate_transfer_succesors_parallel(solution: Solution, r_weights: ResourceWeights, countries: dict, 
                                         state_reduction: int, shared_frontier: PriorityQueue, gamma: int, C: float):
    """Give the current solution and associated information, the function calculates
    all next potential transfer and the resuling states. The function adds these transfers
    and resulting states to the given shared_frontier as they are calculated

    Parameters:
        solution (Solution): Current solution
        r_weights (ResourceWeights): Current resource weights in simulation
        countries (dict): Current state of countires in simulation
        state_reduction (int): Given state reduction of simulation
        shared_frontier (PriorityQueue): Given shared_frontier to place new states in
        gamma (int): Gamma of current simulation
        
    Returns:
        None
    """

    curr_state = solution.path[-1][1]
    curr_countries = solution.path[-1][2]
    countries_elms = {}

    curr_elms = {
        'metalic_elm': curr_state.metalic_elm,
        'timber': curr_state.timber,
        'available_land': curr_state.available_land,
        'water': curr_state.water
    }

    for c in curr_countries:
        countries_elms[c] = {
            'metalic_elm': curr_countries[c].metalic_elm,
            'timber': curr_countries[c].timber,
            'available_land': curr_state.available_land,
            'water': curr_state.water
        }

    for c in countries_elms:
        for elm in countries_elms[c]:

            for curr_elm in curr_elms:
                
                if curr_elm == elm:     # Skipping to avoid redundant trades
                    continue

                other_elm_scale = 1 / r_weights[curr_elm]
                self_elm_scale = 1 / r_weights[elm] 
                max_amount = min(int(countries_elms[c][elm]/other_elm_scale), int(curr_elms[curr_elm]/self_elm_scale))

                if max_amount <= 0:
                    continue
                
                if state_reduction == -1:
                    
                    other_elm_amount = ceil(max_amount / other_elm_scale)
                    self_elm_amount = ceil(max_amount / self_elm_scale)

                    trade = Transfer(elm, curr_elm, other_elm_amount, self_elm_amount, c, curr_state.name)
                    new_curr_state = curr_state.make_trade(curr_elm, self_elm_amount)
                    new_countries = copy.deepcopy(curr_countries)
                    new_countries[c] = curr_countries[c].make_trade(elm, other_elm_amount)
                    new_solution = copy.deepcopy(solution)
                    new_solution.path += [[trade, new_curr_state, new_countries]]
                    calculate_reward_parallel(new_solution, countries, gamma, C)
                    shared_frontier.push(new_solution)
                
                else: 
                    
                    poss_trades = [i+1 for i in range(max_amount)]
                    num_buckets = ceil(len(poss_trades) / state_reduction)

                    if num_buckets < 1 or len(poss_trades) == 0:
                        continue

                    amounts = []
                    buckets = np.array_split(poss_trades, num_buckets)
                    
                    for bucket in buckets:
                        if len(bucket) > 0:
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
                        calculate_reward_parallel(new_solution, countries, gamma, C)
                        shared_frontier.push(new_solution)

def generate_transform_succesors_parallel(solution: Solution, countries: dict, shared_frontier: PriorityQueue, 
                                          gamma: int, C: float) -> None:
    """ Given the current soltuion, and needed surrounding information, this function
    searches and finds all potential next transform steps the solution could take. The
    function then adds the next steps to the given shared_frontier.

    Parameters:
        solution (Solution): Current solution
        countries (dict): Current state of countries for the simulation
        shared_frontier (PriorityQueue): Shared frontier to add new states into to
        gamma (int): Given gamma to calculate EU of a new state
        
    Returns:
        None
    """

    curr_state = solution.path[-1][1]

    housing_scalers = curr_state.can_housing_transform()
    alloy_scalers = curr_state.can_alloys_transform()
    electronics_scalers = curr_state.can_electronics_transform()
    food_scalers = curr_state.can_food_transform()
    farm_scalers = curr_state.can_farm_transform()

    for scaler in housing_scalers:

        trans = HousingTransform(scaler)
        new_state = curr_state.housing_transform(scaler)
        new_solution = copy.deepcopy(solution)
        new_solution.path += [[trans, new_state, countries]]
        calculate_reward_parallel(new_solution, countries, gamma, C)
        shared_frontier.push(new_solution)

    for scaler in alloy_scalers:

        trans = AlloyTransform(scaler)
        new_state = curr_state.alloys_transform(scaler)
        new_solution = copy.deepcopy(solution)
        new_solution.path += [[trans, new_state, countries]]
        calculate_reward_parallel(new_solution, countries, gamma, C)
        shared_frontier.push(new_solution)

    for scaler in electronics_scalers:

        trans = ElectronicTransform(scaler)
        new_state = curr_state.electronics_transform(scaler)
        new_solution = copy.deepcopy(solution)
        new_solution.path += [[trans, new_state, countries]]
        calculate_reward_parallel(new_solution, countries, gamma, C)
        shared_frontier.push(new_solution)
    
    for scaler in food_scalers:

        trans = FoodTransform(scaler)
        new_state = curr_state.food_transform(scaler)
        new_solution = copy.deepcopy(solution)
        new_solution.path += [[trans, new_state, countries]]
        calculate_reward_parallel(new_solution, countries, gamma, C)
        shared_frontier.push(new_solution)
    
    for scaler in farm_scalers:

        trans = FarmTransform(scaler)
        new_state = curr_state.farm_transform(scaler)
        new_solution = copy.deepcopy(solution)
        new_solution.path += [[trans, new_state, countries]]
        calculate_reward_parallel(new_solution, countries, gamma, C)
        shared_frontier.push(new_solution)

def generate_succesors_parallel(chunk: List[Solution], countries: dict, shared_frontier: list, 
                                gamma: int, r_weights: ResourceWeights, state_reduction: int, 
                                depth: int, max_size: int, C: float) -> None:
    """ This function takes in a chunk of solutions and performs a pseudo beam search on the next steps in 
    the solution. It utilizes a temporty frontier object set at a max size relative to given max size
    and the number of cores in the cpu; this allows it to not over extend the search for given states. 
    The function returns once the beam search is complete with the final list of solutions.

    Parameters:
        chunks: (list[Solution]): List of solutions to beam search
        countries (dict): Dictionary of the given countries from the simulation oject
        shared_frontier (list): Shared list over all process to record final list of solutions
        gamma (int): The gamme from the simulation object
        r_weights (ResourceWeights): Weights of given resources for simulation
        state_reduction (int): Given state reduction for the simulation
        depth (int): The given depth of the simulation
        max_size (int): The size of the temp_frontier
        
    Returns:
        None
    """
    
    temp_frontier = PriorityQueue(max_size)     

    for solution in chunk:
        generate_transform_succesors_parallel(solution, countries, temp_frontier, gamma, C)
        generate_transfer_succesors_parallel(solution, r_weights, countries, state_reduction, temp_frontier, gamma, C)

    while not temp_frontier.empty():

        curr_solution = temp_frontier.pop()

        if len(curr_solution.path) > depth:
            shared_frontier.append(curr_solution)
            continue

        generate_transform_succesors_parallel(curr_solution, countries, temp_frontier, gamma, C)
        generate_transfer_succesors_parallel(curr_solution, r_weights, countries, state_reduction, temp_frontier, gamma, C)

"""
    Paralization Metrics
        2 - 0.8 - 10 - 10000: 2.66s
        3 - 0.8 - 10 - 1000: 59.57s
            total: 176032 - 0.00034 per state
        3 - 0.8 - 10 - 1000: 32.02s - less batching
            total: 99427 - 0.00032 per state
        
    
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


def test_case_1():
   
    s = Simulation(
        'tests/1/case_1_countries.xlsx',       # Countries file
        'tests/1/case_1_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        0.1                                     # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/1/output.txt')

def test_case_2():
   
    s = Simulation(
        'tests/2/case_2_countries.xlsx',       # Countries file
        'tests/2/case_2_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        -500                                     # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/2/output.txt')

def test_case_3():
   
    s = Simulation(
        'tests/3/case_3_countries.xlsx',       # Countries file
        'tests/3/case_3_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        1                                   # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/3/output.txt')
    
def test_case_4():
   
    s = Simulation(
        'tests/4/case_4_countries.xlsx',       # Countries file
        'tests/4/case_4_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        1                                   # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/4/output.txt')
    
def main():

    id = str(uuid.uuid4())
    os.system(f'cp Example-Initial-Countries.xlsx input/{id}.xlsx')
    
    s = Simulation(
        'Example-Initial-Countries.xlsx',       # Countries file
        'Example-Sample-Resources.xlsx',        # Resources file
        'Erewhon',                              # Self country
        3,                                      # Depth
        0.8,                                    # Gamma
        1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        0.1                                     # C        
    )

    start = time.time()
    s.search()
    #s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]

    best_solution.print(f'output/{id}.txt')
    print(id)


if __name__ == '__main__':
    #main()
    #test_case_1()
    #test_case_2()
    #test_case_3()
    test_case_4()
    