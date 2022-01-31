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
        
        self.scaler = scaler
        
        self.population_input = state.population
        self.metalic_elm_input = state.metalic_elm
        
        self.population_output = 1 * scaler
        self.metalic_alloy_output = 1 * scaler
        self.metalic_allow_waste_ouptut = 1 * scaler
        self.metalic_elm_output = state.metalic_elm - (2*scaler)
  
        
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
        
        self.scaler = scaler
             
        self.population_input = state.population
        self.metalic_elm_input = state.metalic_elm
        self.metalic_alloy_input = state.metalic_alloys
        
        self.population_output = 1 * scaler
        self.electronics_output = 2 * scaler
        self.electronics_waste_output = 1 * scaler
        self.metalic_elm_output = state.metalic_elm - (3*scaler)
        self.metalic_alloy_output = state.metalic_alloys - (2*scaler)