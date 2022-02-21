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
    metalic_elm_output: int
    timber_output: int
    metalic_alloys_output: int
    
    
    def __init__(self, state: Country, scaler: int) -> None:
        """Given the state and the scaler, captures origional
        value and sets the new resource values of the state

        Parameters:
            state (Country): Current state to transform
            scaler (int): Scaler for transformations
            
        Returns:
            None
        """
        
        self.scaler = scaler
          
        self.population_input = state.population
        self.metalic_elm_input = state.metalic_elm
        self.timber_input = state.timber
        self.metalic_alloys_input = state.metalic_alloys
        
        self.housing_output = 1 * scaler
        self.housing_waste__output = 1 * scaler
        self.population_output = 5 * scaler
        self.metalic_elm_output = state.metalic_elm - (1*scaler)
        self.timber_output = state.timber - (5*scaler)
        self.metalic_alloys_output = state.metalic_alloys - (3*scaler)
    
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
        print(f"        population: {self.population_output}")
        print()
        

@dataclass 
class AlloyTransform:
    
    scaler: int
    
    population_input: int
    metalic_elm_input: int
    
    population_output: int
    metalic_alloy_output: int
    metalic_allow_waste_ouptut: int
    metalic_elm_output: int
    
    def __init__(self, state: Country, scaler: int) -> None:
        """Given the state and the scaler, captures origional
        value and sets the new resource values of the state

        Parameters:
            state (Country): Current state to transform
            scaler (int): Scaler for transformations
            
        Returns:
            None
        """
        
        self.scaler = scaler
        
        self.population_input = state.population
        self.metalic_elm_input = state.metalic_elm
        
        self.population_output = 1 * scaler
        self.metalic_alloy_output = 1 * scaler
        self.metalic_allow_waste_ouptut = 1 * scaler
        self.metalic_elm_output = state.metalic_elm - (2*scaler)
    
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
        print(f"        metalic_alloy_waste: {self.metalic_allow_waste_ouptut}")
        print(f"        population: {self.population_output}")
        print()
  
        
@dataclass
class ElectronicTransform:
    
    scaler: int
    
    population_input: int
    metalic_elm_input: int
    metalic_alloy_input: int
    
    population_output: int
    electronics_output: int
    electronics_waste_output: int
    metalic_elm_output: int
    metalic_alloy_output: int
    
    def __init__(self, state: Country, scaler: int) -> None:
        """Given the state and the scaler, captures origional
        value and sets the new resource values of the state

        Parameters:
            state (Country): Current state to transform
            scaler (int): Scaler for transformations
            
        Returns:
            None
        """
        
        self.scaler = scaler
             
        self.population_input = state.population
        self.metalic_elm_input = state.metalic_elm
        self.metalic_alloy_input = state.metalic_alloys
        
        self.population_output = 1 * scaler
        self.electronics_output = 2 * scaler
        self.electronics_waste_output = 1 * scaler
        self.metalic_elm_output = state.metalic_elm - (3*scaler)
        self.metalic_alloy_output = state.metalic_alloys - (2*scaler)
    
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
        print(f"        population: {self.population_output}")
        print()