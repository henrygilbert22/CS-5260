from dataclasses import dataclass
import numpy as np
import copy

@dataclass
class ResourceWeights:
    
    population: float
    metalic_elm: float
    timber: float
    available_land: float
    water: float
    
    metalic_alloys: float
    housing: float
    electronics: float
    farm: float
    food: float
    
    metalic_waste: float   
    electronics_waste: float 
    housing_waste: float 
    farm_waste: int 
    food_waste: int 
      
    
    def __getitem__(self, item):
        """Base level function to treat
        class of resource weights like a dictionary.
        This helps make the access code much cleaner in the
        simulation class

        Args:
            item (_type_): Attribute to access

        Returns:
            _type_: Attribute
        """
        return getattr(self, item) 
    
@dataclass
class Country:
    
    name: str
    
    population: int
    metalic_elm: int
    timber: int
    available_land: int
    water: int
    
    weights: ResourceWeights
     
    metalic_alloys: int = 0
    electronics: int = 0
    housing: int = 0
    farm: int = 0
    food: int = 0
    
    metalic_waste: int = 0     
    electronics_waste: int = 0     
    housing_waste: int = 0
    farm_waste: int = 0
    food_waste: int = 0

    max_water: int = 0
    max_timber: int = 0
    
    def __init__(self, name: str, population: int, metalic_elm: int, timber: int, 
                available_land: int, water: int, weights: ResourceWeights) -> None:
        
        self.name = name
        self.population = population
        self.metalic_elm = metalic_elm
        self.timber = timber
        self.available_land = available_land
        self.water = water
        self.weights = weights
        
        self.max_water = water
        self.max_timber = timber

    def __getitem__(self, item):
        """Base level function to treat
        class of resource weights like a dictionary.
        This helps make the access code much cleaner in the
        simulation class

        Args:
            item (_type_): Attribute to access

        Returns:
            _type_: Attribute
        """
        return getattr(self, item) 

    def __setitem__(self, key, value):
        """Base level function to treat
        class of resource weights like a dictionary.
        This helps make the access code much cleaner in the
        simulation class

        Args:
            item (_type_): Attribute to access

        Returns:
            _type_: Attribute
        """
        setattr(self, key, value)

    def state(self):

        return [self.population, self.metalic_elm, self.timber, self.available_land, self.water,
        self.metalic_alloys, self.electronics, self.housing, self.farm, self.food,
        self.metalic_waste, self.electronics_waste, self.housing_waste, self.farm_waste, self.food_waste]

    def get_resource_dict(self):

        resources = {
            'metalic_elm': self.metalic_elm,
            'timber': self.timber,
            'available_land': self.available_land,
            'water': self.water,
            'metalic_alloys': self.metalic_alloys,
            'electronics': self.electronics,
            'housing': self.housing,
            'farm': self.farm,
            'food': self.food,
        }

        waste = {
            'metalic_waste': self.metalic_waste,
            'electronics_waste': self.electronics_waste,
            'housing_waste': self.housing_waste,
            'farm_waste': self.farm_waste,
            'food_waste': self.food_waste,
        }

        return resources, waste

    def state_value(self) -> float:
        """Returns the base value of a state. Calculated
        as a weighted sum of the resources, developement and waste

        Returns:
            float: Base state value
        """
        
        resource_score =  (
            (self.weights['metalic_elm'] * self.metalic_elm) + 
            (self.weights['timber'] * self.timber) + 
            (self.weights['available_land'] * self.available_land) + 
            (self.weights['water'] * self.water)
            )
        
        developement_score = (
            (self.weights['metalic_alloys'] * self.metalic_alloys) + 
            (self.weights['electronics'] * self.electronics) + 
            (self.weights['housing'] * self.housing) + 
            (self.weights['farm'] * self.farm) +
            (self.weights['food'] * self.food)
            )
        
        waste_score = (
            (self.weights['metalic_waste'] * self.metalic_waste) + 
            (self.weights['electronics_waste'] * self.electronics_waste) + 
            (self.weights['housing_waste'] * self.housing_waste) + 
            (self.weights['farm_waste'] * self.farm_waste) + 
            (self.weights['food_waste'] * self.food_waste)
            )

        return round((resource_score + 10*developement_score - waste_score)* self.food_availability_scaler(), 2)           # make this more complex at some point
    
    def food_availability_scaler(self):

        if self.food - self.population < 0:
            return 0.5
        
        return 1

    def make_trade(self, self_resource: str, self_amount: int, other_resource: str, other_amount: int):
        """Function to subtract resource for any
        given trade

        Args:
            resource (str): Resource to manipulate
            amount (int): Amount to subtract 

        Returns:
            Country: New country
        """
        
        self[self_resource] -= self_amount
        self[other_resource] += other_amount
        self.adjust_continuals()
            
    def transform(self, type: str):

        if type == "housing":
            self.housing_transform()  
        
        elif type == "food":
            self.food_transform()
        
        elif type == "alloys":
            self.alloys_transform()
        
        elif type == "electronics":
            self.electronics_transform()
        
        elif type == "farm":
            self.farm_transform()

        self.adjust_continuals()

    def adjust_continuals(self):
        """ Function to adjust the continuals of a given country
        
        Arguements:
            None
            
        Returns:
            None
        """

        if self.water + (self.max_water * 0.1) < self.max_water:     #increasing water by 10%
            self.water += (self.max_water * 0.1)
        else:
            self.water = self.max_water

        self.food -= int(0.1*self.population)        #  Each population eats .1 food
        if self.food < 0: self.food = 0

        if self.timber + (self.max_timber * 0.1) < self.max_timber:     #increasing timber by 10% of current timber Can't grow more timber than you gave away
            self.timber += (self.max_timber * 0.1)
        else:
            self.timber = self.max_timber

    def housing_transform(self):
        """ Performs the given housing transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """
            
        scalers = []
        scalers.append(int(self.population / 5))
        scalers.append(int(self.metalic_elm / 1))
        scalers.append(int(self.timber / 5))
        scalers.append(int(self.metalic_alloys / 3))
        max_scalers = min(scalers)

        self.population -= 5*max_scalers
        self.metalic_elm -= 1*max_scalers
        self.timber -= 5*max_scalers
        self.metalic_alloys -= 3*max_scalers
        
        self.housing += 1*max_scalers
        self.housing_waste += 1*max_scalers
        self.population += 5*max_scalers
            
    def alloys_transform(self):
        """ Performs the given alloys transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """

        scalers = []
        scalers.append(int(self.population / 1))
        scalers.append(int(self.metalic_elm / 2))
        max_scaler = min(scalers)
        
        self.population -= 1*max_scaler
        self.metalic_elm -= 2*max_scaler
        
        self.population += 1*max_scaler
        self.metalic_alloys += 1*max_scaler
        self.metalic_waste += 1*max_scaler
                
    def electronics_transform(self):
        """ Performs the given electronics transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """

        scalers = []
        scalers.append(int(self.population / 1))
        scalers.append(int(self.metalic_elm / 3))
        scalers.append(int(self.metalic_alloys / 2))

        max_scaler = min(scalers)
        
        self.population -= 1*max_scaler
        self.metalic_elm -= 3*max_scaler
        self.metalic_alloys -= 2*max_scaler
        
        self.population += 1*max_scaler
        self.electronics += 2*max_scaler
        self.electronics_waste += 1*max_scaler
        
    def food_transform(self):
        """ Performs the given electronics transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """

        scalers = []
        scalers.append(int(self.farm / 5))
        scalers.append(int(self.water / 10))
        max_scaler = min(scalers)
        
        self.water -= 10*max_scaler
        
        self.food += 1*max_scaler
        self.food_waste += int(0.5*max_scaler)
            
    def farm_transform(self):
        """ Performs the given electronics transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """

        scalers = []
        scalers.append(int(self.timber / 5))
        scalers.append(int(self.available_land / 10))
        scalers.append(int(self.water / 10))
        max_scaler = min(scalers)

        self.timber -= 5*max_scaler
        self.available_land -= 10*max_scaler
        self.water -= 10*max_scaler
        
        self.farm += 5*max_scaler
        self.farm_waste += 1*max_scaler
        
