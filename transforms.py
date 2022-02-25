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
        
        output_string = f"""
        HOUSING TRANSFORM:
             INPUTS:
                population: {self.population_input}
                metalic_elm: {self.metalic_elm_input}
                timber: {self.timber_input}
                metalic_alloy: {self.metalic_alloys_input}
             OUTPUTS:
                housing: {self.housing_output}
                housing_waste: {self.housing_waste__output}
                population: {self.population_output}\n
        """
        
        print(output_string)
        return output_string
        
    
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
        
        output_string = f"""
        ALLOY TRANSFORM:
             INPUTS:
                population: {self.population_input}
                metalic_elm: {self.metalic_elm_input}
             OUTPUTS:
                metalic_alloy: {self.metalic_alloy_output}
                metalic_alloy_waste: {self.metalic_alloy_waste_ouptut}
                population: {self.population_output}\n
        """
        
        print(output_string)
        return output_string
        
        
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
        
        output_string = f"""
        ELECTRONIC TRANSFORM:
             INPUTS:
                population: {self.population_input}
                metalic_elm: {self.metalic_elm_input}
                metalic_allot: {self.metalic_alloy_input}
             OUTPUTS:
                electronics: {self.electronics_output}
                electronics_waste: {self.electronics_waste_output}
                population: {self.population_output}\n
        """
        print(output_string)
        return output_string
        

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
        output_string = f"""
        FOOD TRANSFORM:
            INPUTS:
                water: {self.water_input}
                farm: {self.farm_input}
            OUTPUTS:
                food: {self.food_output}
                farm: {self.farm_output}
                food_waste: {self.food_waste_output}\n
        """
        print(output_string)
        return output_string

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
        
        output_string = f"""
        FARM TRANSFORM:
             INPUTS:
                water: {self.water_input}
                timber: {self.timber_input}
                available_land: {self.available_land_input}
             OUTPUTS:
                farm: {self.farm_output}
                farm_waste: {self.farm_waste_output}\n
        """
        
        print(output_string)
        return output_string
        