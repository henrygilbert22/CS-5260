from dataclasses import dataclass
import pandas as pd

@dataclass
class Country:
    
    name: str
    r1: int
    r2: int
    r3: int
    r21: int
    r22: int
    r23: int
    r21w: int       # Waste for r21
    r22w: int       # Waste for r22
    r22w: int       # Waste for r23
    
@dataclass
class ResourceWeights:
    
    r1: int
    r2: int
    r3: int
    r21: int
    r22: int
    r23: int
    r21w: int       # Waste for r21
    r22w: int       # Waste for r22
    r22w: int       # Waste for r23


class Simulation:
    
    countries: list[Country]
    r_weights: ResourceWeights
    
    def __init__(self) -> None:
        pass
    
    def load(self):
        
        self.load_countries('Example-Initial-Countries.xlsx')
        
        
    def load_countries(self, file_name: str):
    
        df = pd.read_excel(file_name)
        
        print(df)
        
        
        
def main():
    
    s = Simulation()

if __name__ == '__main__':
    main()