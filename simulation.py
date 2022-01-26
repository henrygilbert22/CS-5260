from dataclasses import dataclass
import pandas as pd

@dataclass
class Country:
    
    name: str
    population: int
    metalic_elm: int
    timber: int
    metalic_alloys: int
    electronics: int
    housing: int
    waste1: int = 0     # Waste for metalic_alloys
    waste2: int = 0     # Waste for electronics
    waste3: int = 0     # Waste for housing
    
    def state_value(self) -> float:
        
        resource_score =  self.metalic_alloys + self.timber + self.metalic_alloys
        developement_score = self.metalic_alloys + self.electronics + self.housing
        waste_score = self.waste1 + self.waste2 + self.waste3
        
        return round(resource_score + 2*developement_score - waste_score, 2)
    
@dataclass
class ResourceWeights:
    
    population: float
    metalic_elm: float
    timber: float
    metalic_alloys: float
    electronics: float
    housing: int
    waste1: float = 0.0     # Waste for metalic_alloys
    waste2: float = 0.0     # Waste for electronics
    waste3: float = 0.0     # Waste for housing


class Simulation:
    
    countries: list[Country]
    r_weights: ResourceWeights
    
    countries_file_name: str
    weights_file_name: str
    
    def __init__(self, countries_file_name: str, weights_file_name: str) -> None:
        
        self.countries_file_name = countries_file_name
        self.weights_file_name = weights_file_name
        
        self.countries = []
        
        self.load()
    
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
        
        
def main():
    
    s = Simulation('Example-Initial-Countries.xlsx', 'Example-Sample-Resources.xlsx')

if __name__ == '__main__':
    main()