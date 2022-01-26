from dataclasses import dataclass
import pandas as pd


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
        waste_score = self.waste1 + self.waste2 + self.waste3
        
        return round(resource_score + 2*developement_score - waste_score, 2)
    
    def housing_transform(self, scaler: int):
        
        if (self.population >= 5*scaler and self.metalic_elm >= 1*scaler 
            and self.timber >= 5*scaler and self.metalic_alloys >= 3*scaler): 
            
            self.population -= 5*scaler
            self.metalic_elm -= 1*scaler
            self.timber -= 5*scaler
            self.metalic_alloys -= 3*scaler
            
            self.housing += 1*scaler
            self.housing_waste += 1*scaler
            self.population += 5*scaler
    
    def alloys_transform(self, scaler: int):
        
        if (self.population >= 1*scaler, self.metalic_elm >= 2*scaler):
            
            self.population -= 1*scaler
            self.metalic_elm -= 2*scaler
            
            self.population += 1*scaler
            self.metalic_alloys += 1*scaler
            self.metalic_waste += 1*scaler
            
    def electronics_transform(self, scaler: int):
        
        if (self.population >= 1*scaler and self.metalic_elm >= 3*scaler
            and self.metalic_alloys >= 2*scaler):
            
            self.population -= 1*scaler
            self.metalic_elm -= 3*scaler
            self.metalic_alloys -= 2*scaler
            
            self.population += 1*scaler
            self.electronics += 2*scaler
            self.electronics_waste += 1*scaler
            
                       
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


class PriorityQueue:
    
    priority_queue: list
    
    def __init__(self):
        
        self.priority_queue = []
  
    def insert(self, data):
        
        self.priority_queue.append(data)
        
    def pop(self):
        
        max_value = 0
        
        for i in range(len(self.priority_queue)):
            
            if self.priority_queue[i] > self.priority_queue[max_value]:
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
    depth: int
    
    def __init__(self, countries_file_name: str, weights_file_name: str, country: int, depth: int) -> None:
        
        self.countries_file_name = countries_file_name
        self.weights_file_name = weights_file_name
        self.depth = depth
        self.countries = []
        self.frontier = PriorityQueue()
        
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
    
    def search(self):
        
        while True:
            pass
            
            
        
        
def main():
    
    s = Simulation('Example-Initial-Countries.xlsx', 'Example-Sample-Resources.xlsx', 0, 3)

if __name__ == '__main__':
    main()