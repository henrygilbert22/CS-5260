from dataclasses import dataclass
import pandas as pd
from rx import empty
import copy
import time
import multiprocessing as mp
from more_itertools import chunked

# Schedule reward is differnce of inititial state value and final state value, this is the undiscounted reward
# Discounted schedule reward: gamma^N * (Q_end(c_i, s_j) – Q_start(c_i, s_j))
# We must use the expected utility of a scheudle which is the dicounted * the probility that it will actually hapen
    # This probobility is calculated by taking into account how the schedule affects the other countries, and how benificial it would be
    # for them
    
# Search algo can not be recursive


@dataclass
class Country:
    
    name: str
    population: int
    metalic_elm: int
    timber: int
    metalic_alloys: int
    electronics: int
    housing: int
    metalic_waste: int = 0     
    electronics_waste: int = 0     
    housing_waste: int = 0     
    
    def state_value(self) -> float:
        
        resource_score =  self.metalic_alloys + self.timber + self.metalic_alloys
        developement_score = self.metalic_alloys + self.electronics + self.housing
        waste_score = self.metalic_waste + self.electronics_waste + self.housing_waste
        
        return round(resource_score + 2*developement_score - waste_score, 2)
    
    def can_housing_transform(self):
        
        if (self.population >= 5 and self.metalic_elm >= 1 
            and self.timber >= 5 and self.metalic_alloys >= 3): 
            
            scalers = []
            scalers.append(int(self.population / 5))
            scalers.append(int(self.metalic_elm / 1))
            scalers.append(int(self.timber / 5))
            scalers.append(int(self.metalic_alloys / 3))
            
            return [i+1 for i in range(min(scalers))]
        
        else:
            return []
    
    def can_alloys_transform(self):
        
        if (self.population >= 1, self.metalic_elm >= 2):       
            
            scalers = []
            scalers.append(int(self.population / 1))
            scalers.append(int(self.metalic_elm / 2))
            
            return [i+1 for i in range(min(scalers))]
        
        else:
            return []
    
    def can_electronics_transform(self):
        
        if (self.population >= 1 and self.metalic_elm >= 3
            and self.metalic_alloys >= 2):
            
            scalers = []
            scalers.append(int(self.population / 1))
            scalers.append(int(self.metalic_elm / 3))
            scalers.append(int(self.metalic_alloys / 2))

            return [i+1 for i in range(min(scalers))]
        
        else:
            return []
        
    def housing_transform(self, scaler: int):
            
        new_state = copy.deepcopy(self)
        
        new_state.population -= 5*scaler
        new_state.metalic_elm -= 1*scaler
        new_state.timber -= 5*scaler
        new_state.metalic_alloys -= 3*scaler
        
        new_state.housing += 1*scaler
        new_state.housing_waste += 1*scaler
        new_state.population += 5*scaler
        
        return new_state
    
    def alloys_transform(self, scaler: int):
  
        new_state = copy.deepcopy(self)
        
        new_state.population -= 1*scaler
        new_state.metalic_elm -= 2*scaler
        
        new_state.population += 1*scaler
        new_state.metalic_alloys += 1*scaler
        new_state.metalic_waste += 1*scaler
        
        return new_state
            
    def electronics_transform(self, scaler: int):

        new_state = copy.deepcopy(self)
        
        new_state.population -= 1*scaler
        new_state.metalic_elm -= 3*scaler
        new_state.metalic_alloys -= 2*scaler
        
        new_state.population += 1*scaler
        new_state.electronics += 2*scaler
        new_state.electronics_waste += 1*scaler
        
        return new_state
            
                       
@dataclass
class ResourceWeights:
    
    population: float
    metalic_elm: float
    timber: float
    metalic_alloys: float
    electronics: float
    housing: int
    metalic_waste: float = 0.0     
    electronics_waste: float = 0.0     
    housing_waste: float = 0.0     

@dataclass
class HousingTransform:
    
    population_input: int
    metalic_elm_input: int
    timber_input: int
    metalic_alloys_input: int
    
    housing_output: int
    housing_waste__output: int
    population_output: int
    
    def __init__(self, state: Country, scaler: int) -> None:
                
        self.population_input = state.population
        self.metalic_elm_input = state.metalic_elm
        self.timber_input = state.timber
        self.metalic_alloys_input = state.metalic_alloys
        
        self.housing_output = 1 * scaler
        self.housing_waste__output = 1 * scaler
        self.population_output = 5 * scaler
        

@dataclass 
class AlloyTransform:
    
    population_input: int
    metalic_elm_input: int
    
    population_output: int
    metalic_alloy_output: int
    metalic_allow_waste_ouptut: int
    
    def __init__(self, state: Country, scaler: int) -> None:
        
        self.population_input = state.population
        self.metalic_elm_input = state.metalic_elm
        
        self.population_output = 1 * scaler
        self.metalic_alloy_output = 1 * scaler
        self.metalic_allow_waste_ouptut = 1 * scaler
  
        
@dataclass
class ElectronicTransform:
    
    population_input: int
    metalic_elm_input: int
    metalic_alloy_input: int
    
    population_output: int
    electronics_output: int
    electronics_waste_output: int
    
    def __init__(self, state: Country, scaler: int) -> None:
                
        self.population_input = state.population
        self.metalic_elm_input = state.metalic_elm
        self.metalic_alloy_input = state.metalic_alloys
        
        self.population_output = 1 * scaler
        self.electronics_output = 2 * scaler
        self.electronics_waste_output = 1 * scaler
   

@dataclass
class Solution:
    
    expected_utility: float
    path: list
    

class PriorityQueue:
    
    priority_queue: list[Solution]
    
    def __init__(self):
        
        self.priority_queue = []
  
    def push(self, data: tuple) -> None:
        
        self.priority_queue.append(data)
    
    def empty(self):
        
        return len(self.priority_queue) == 0
    
    def pop(self) -> object:
        
        max_value = 0
        
        for i in range(len(self.priority_queue)):
            
            if self.priority_queue[i].expected_utility > self.priority_queue[max_value].expected_utility:       # Comparing value
                max_value = i
        
        selected_item = self.priority_queue[max_value]
        del self.priority_queue[max_value]
        
        return selected_item
     
                          
class Simulation:
    
    countries: list[Country]
    r_weights: ResourceWeights
    
    countries_file_name: str
    weights_file_name: str
    
    country: Country
    frontier: PriorityQueue
    solutions: PriorityQueue
    depth: int
    
    def __init__(self, countries_file_name: str, weights_file_name: str, country: int, depth: int) -> None:
        
        self.countries_file_name = countries_file_name
        self.weights_file_name = weights_file_name
        self.depth = depth
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
           self.countries.append(Country(*row.values))
    
    def generate_succesors(self, state: Country, solution: Solution):           # Paralize this for extra credit
                
        housing_scalers = state.can_housing_transform()
        alloy_scalers = state.can_alloys_transform()
        electronics_scalers = state.can_electronics_transform()
        
        for scaler in housing_scalers:
            
            trans = HousingTransform(state, scaler)
            new_state = state.housing_transform(scaler)
            new_solution = Solution(new_state.state_value(), solution.path + [[trans, new_state]])        
            self.frontier.push(new_solution)
            
        for scaler in alloy_scalers:
            
            trans = AlloyTransform(state, scaler)
            new_state = state.alloys_transform(scaler)
            new_solution = Solution(new_state.state_value(), solution.path + [[trans, new_state]])        
            self.frontier.push(new_solution)
        
        for scaler in electronics_scalers:
            
            trans = ElectronicTransform(state, scaler)
            new_state = state.electronics_transform(scaler)
            new_solution = Solution(new_state.state_value(), solution.path + [[trans, new_state]])        
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
            
            self.generate_succesors(self.country, solution)
            

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

        
def main():
    
    s = Simulation('Example-Initial-Countries.xlsx', 'Example-Sample-Resources.xlsx', 4, 3)
    
    start = time.time()
    s.search()
    end = time.time()
    
    print(f"Took: {end-start}")
    
    print(len(s.solutions.priority_queue))
    best_solution = s.solutions.pop()
    print(best_solution)
    

if __name__ == '__main__':
    main()