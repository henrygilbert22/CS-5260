from dataclasses import dataclass

from country import Country

@dataclass
class HousingTransform:
    
    scaler: int
    
    population_input: int
    metalic_elm_input: int
    timber_input: int
    metalic_alloys_input: int
    
    housing_output: int
    housing_waste__output: int
    population_output: int    
    
    def __init__(self, scaler: int) -> None:
        """Given the state and the scaler, captures origional
        value and sets the new resource values of the state

        Parameters:
            state (Country): Current state to transform
            scaler (int): Scaler for transformations
            
        Returns:
            None
        """
        
        self.scaler = scaler
          
        self.population_input = 5 * scaler
        self.metalic_elm_input = 1 * scaler
        self.timber_input = 5 * scaler
        self.metalic_alloys_input = 3 * scaler
        
        self.housing_output = 1 * scaler
        self.housing_waste__output = 1 * scaler
        self.population_output = 5 * scaler
    
    def print(self):
        """Pretty prints the transformation
        
        Parameters: 
            None
            
        Returns:
            None
        """
        
        print("HOUSING TRANSFORM:")
        print(f"     INPUTS:")
        print(f"        population: {self.population_input}")
        print(f"        metalic_elm: {self.metalic_elm_input}")
        print(f"        timber: {self.timber_input}")
        print(f"        metalic_alloy: {self.metalic_alloys_input}")
        print(f"     OUTPUTS:")
        print(f"        housing: {self.housing_output}")
        print(f"        housing_waste: {self.housing_waste__output}")
        print(f"        population: {self.population_output}\n")
    
@dataclass 
class AlloyTransform:
    
    scaler: int
    
    population_input: int
    metalic_elm_input: int
    
    population_output: int
    metalic_alloy_output: int
    metalic_alloy_waste_ouptut: int
    
    def __init__(self, scaler: int) -> None:
        """Given the state and the scaler, captures origional
        value and sets the new resource values of the state

        Parameters:
            state (Country): Current state to transform
            scaler (int): Scaler for transformations
            
        Returns:
            None
        """
        
        self.scaler = scaler
        
        self.population_input = 1 * scaler
        self.metalic_elm_input = 2 * scaler
        
        self.population_output = 1 * scaler
        self.metalic_alloy_output = 1 * scaler
        self.metalic_alloy_waste_ouptut = 1 * scaler
    
    def print(self):
        """Pretty prints the transformation
        
        Parameters: 
            None
            
        Returns:
            None
        """
        
        print("ALLOY TRANSFORM:")
        print(f"     INPUTS:")
        print(f"        population: {self.population_input}")
        print(f"        metalic_elm: {self.metalic_elm_input}")
        print(f"     OUTPUTS:")
        print(f"        metalic_alloy: {self.metalic_alloy_output}")
        print(f"        metalic_alloy_waste: {self.metalic_alloy_waste_ouptut}")
        print(f"        population: {self.population_output}\n")
        
@dataclass
class ElectronicTransform:
    
    scaler: int
    
    population_input: int
    metalic_elm_input: int
    metalic_alloy_input: int
    
    population_output: int
    electronics_output: int
    electronics_waste_output: int
    
    def __init__(self, scaler: int) -> None:
        """Given the state and the scaler, captures origional
        value and sets the new resource values of the state

        Parameters:
            state (Country): Current state to transform
            scaler (int): Scaler for transformations
            
        Returns:
            None
        """
        
        self.scaler = scaler
             
        self.population_input = 1 * scaler
        self.metalic_elm_input = 3 * scaler
        self.metalic_alloy_input = 2 * scaler
        
        self.population_output = 1 * scaler
        self.electronics_output = 2 * scaler
        self.electronics_waste_output = 1 * scaler
    
    def print(self):
        """Pretty prints the transformation
        
        Parameters: 
            None
            
        Returns:
            None
        """
        
        print("ELECTRONIC TRANSFORM:")
        print(f"     INPUTS:")
        print(f"        population: {self.population_input}")
        print(f"        metalic_elm: {self.metalic_elm_input}")
        print(f"        metalic_allot: {self.metalic_alloy_input}")
        print(f"     OUTPUTS:")
        print(f"        electronics: {self.electronics_output}")
        print(f"        electronics_waste: {self.electronics_waste_output}")
        print(f"        population: {self.population_output}\n")

@dataclass
class FoodTransform:
    
    scaler: int
    
    water_input: int
    farm_input: int
    
    food_output: int
    food_waste_output: int
    farm_output: int
    
    def __init__(self, scaler: int) -> None:
        """Given the state and the scaler, captures origional
        value and sets the new resource values of the state

        Parameters:
            state (Country): Current state to transform
            scaler (int): Scaler for transformations
            
        Returns:
            None
        """
        
        self.scaler = scaler
             
        self.water_input = 10 * scaler
        self.farm_input = 5 * scaler

        self.farm_output = 5 * scaler
        self.food_output = 1 * scaler
        self.food_waste_output = int(0.5 * scaler)
       
    def print(self):
        """Pretty prints the transformation
        
        Parameters: 
            None
            
        Returns:
            None
        """
        
        print("FOOD TRANSFORM:")
        print(f"     INPUTS:")
        print(f"        water: {self.water_input}")
        print(f"        farm: {self.farm_input}")
        print(f"     OUTPUTS:")
        print(f"        food: {self.food_output}")
        print(f"        farm: {self.farm_output}")
        print(f"        food_waste: {self.food_waste_output}\n")

@dataclass
class FarmTransform:
    
    scaler: int
    
    timber_input: int
    available_land_input: int
    water_input: int
    
    farm_output: int
    farm_waste_output: int
    
    def __init__(self, scaler: int) -> None:
        """Given the state and the scaler, captures origional
        value and sets the new resource values of the state

        Parameters:
            state (Country): Current state to transform
            scaler (int): Scaler for transformations
            
        Returns:
            None
        """
        
        self.scaler = scaler
             
        self.water_input = 10 * scaler
        self.timber_input = 5 * scaler
        self.available_land_input = 10 * scaler
        
        self.farm_output = 5 * scaler
        self.farm_waste_output = 1 * scaler
        
       
    def print(self):
        """Pretty prints the transformation
        
        Parameters: 
            None
            
        Returns:
            None
        """
        
        print("FARM TRANSFORM:")
        print(f"     INPUTS:")
        print(f"        water: {self.water_input}")
        print(f"        timber: {self.timber_input}")
        print(f"        available_land: {self.available_land_input}")
        print(f"     OUTPUTS:")
        print(f"        farm: {self.farm_output}")
        print(f"        farm_waste: {self.farm_waste_output}\n")
