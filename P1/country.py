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
    
    state_reduction: int
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
    
    def __init__(self, name: str, population: int, metalic_elm: int, timber: int, 
                available_land: int, water: int, state_reduction: int, weights: ResourceWeights) -> None:
        
        self.name = name
        self.population = population
        self.metalic_elm = metalic_elm
        self.timber = timber
        self.available_land = available_land
        self.water = water
        self.state_reduction = state_reduction
        self.weights = weights
        self.max_water = water

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
        
        new_state = copy.deepcopy(self)

        new_state[self_resource] = new_state[self_resource] - self_amount
        new_state[other_resource] = new_state[other_resource] + other_amount

        new_state.adjust_continuals()
        return new_state
    
    def can_farm_transform(self):
        """Function to calculate possible scalers for a
        housing transform given the current state of the country.

        Parameters:
            None
            
        Returns:
           list: List of potential scalers for a housing transform.
        """
                
        if (self.timber >= 5 and self.available_land >=10
            and self.water >= 10): 
            
            scalers = []
            scalers.append(int(self.timber / 5))
            scalers.append(int(self.available_land / 10))
            scalers.append(int(self.water / 10))
            
            if self.state_reduction == -1:
                return [min(scalers)]
                
            poss_scalers = [i+1 for i in range(min(scalers))]
            num_buckets = round(len(poss_scalers) / self.state_reduction)
            
            if num_buckets < 1 or len(poss_scalers) == 0:
                return poss_scalers
            
            buckets = np.array_split(poss_scalers, num_buckets)    
            final_scalers = []
            
            for bucket in buckets:
                if len(bucket) > 0:                  # Takes care if state_reduction is larger than starting buckets
                    final_scalers.append(int(sum(bucket)/len(bucket)))
                    
            return final_scalers
        
        else:
            return []   
    
    def can_food_transform(self):
        """Function to calculate possible scalers for a
        housing transform given the current state of the country.

        Parameters:
            None
            
        Returns:
           list: List of potential scalers for a housing transform.
        """
                
        if (self.farm >=5 and self.water >= 10): 
            
            scalers = []
            scalers.append(int(self.farm / 5))
            scalers.append(int(self.water / 10))
            
            if self.state_reduction == -1:
                return [min(scalers)]
            
            poss_scalers = [i+1 for i in range(min(scalers))]
            num_buckets = round(len(poss_scalers) / self.state_reduction)
            
            if num_buckets < 1 or len(poss_scalers) == 0:
                return poss_scalers
            
            buckets = np.array_split(poss_scalers, num_buckets)    
            final_scalers = []
            
            for bucket in buckets:
                if len(bucket) > 0:                  # Takes care if state_reduction is larger than starting buckets
                    final_scalers.append(int(sum(bucket)/len(bucket)))
                    
            return final_scalers
        
        else:
            return []          
        
    def can_housing_transform(self):
        """Function to calculate possible scalers for a
        housing transform given the current state of the country.

        Parameters:
            None
            
        Returns:
           list: List of potential scalers for a housing transform.
        """
                
        if (self.population >= 5 and self.metalic_elm >= 1 
            and self.timber >= 5 and self.metalic_alloys >= 3): 
            
            scalers = []
            scalers.append(int(self.population / 5))
            scalers.append(int(self.metalic_elm / 1))
            scalers.append(int(self.timber / 5))
            scalers.append(int(self.metalic_alloys / 3))
            
            if self.state_reduction == -1:
                return [min(scalers)]
            
            poss_scalers = [i+1 for i in range(min(scalers))]
            num_buckets = round(len(poss_scalers) / self.state_reduction)
            
            if num_buckets < 1 or len(poss_scalers) == 0:
                return poss_scalers
            
            buckets = np.array_split(poss_scalers, num_buckets)    
            final_scalers = []
            
            for bucket in buckets:
                if len(bucket) > 0:                  # Takes care if state_reduction is larger than starting buckets
                    final_scalers.append(int(sum(bucket)/len(bucket)))
                    
            return final_scalers
        
        else:
            return []
    
    def can_alloys_transform(self):
        """Function to calculate possible scalers for a
        alloy transform given the current state of the country.

        Parameters:
            None
            
        Returns:
           list: List of potential scalers for a alloy transform.
        """
        
        if (self.population >= 1, self.metalic_elm >= 2):       
            
            scalers = []
            scalers.append(int(self.population / 1))
            scalers.append(int(self.metalic_elm / 2))
            
            if self.state_reduction == -1:
                return [min(scalers)]
            
            poss_scalers = [i+1 for i in range(min(scalers))]
            num_buckets = round(len(poss_scalers) / self.state_reduction)
            
            if num_buckets < 1 or len(poss_scalers) == 0:
                return poss_scalers
            
            buckets = np.array_split(poss_scalers, num_buckets)
            final_scalers = []
            
            for bucket in buckets:
                if len(bucket) > 0:                  # Takes care if state_reduction is larger than starting buckets
                    final_scalers.append(int(sum(bucket)/len(bucket)))
                    
            return final_scalers
        
        else:
            return []
    
    def can_electronics_transform(self):
        """Function to calculate possible scalers for a
        electronics transform given the current state of the country.

        Parameters:
            None
            
        Returns:
           list: List of potential scalers for a electronics transform.
        """
        
        if (self.population >= 1 and self.metalic_elm >= 3
            and self.metalic_alloys >= 2):
            
            scalers = []
            scalers.append(int(self.population / 1))
            scalers.append(int(self.metalic_elm / 3))
            scalers.append(int(self.metalic_alloys / 2))

            if self.state_reduction == -1:
                return [min(scalers)]
            
            poss_scalers = [i+1 for i in range(min(scalers))]
            num_buckets = round(len(poss_scalers) / self.state_reduction)
            
            if num_buckets < 1 or len(poss_scalers) == 0:
                return poss_scalers
            
            buckets = np.array_split(poss_scalers, num_buckets)
            final_scalers = []
            
            for bucket in buckets:
                if len(bucket) > 0:                  # Takes care if state_reduction is larger than starting buckets
                    final_scalers.append(int(sum(bucket)/len(bucket)))
                    
            return final_scalers
        
        else:
            return []

    def transform(self, type: str, scaler: int):

        if type == "housing":
            new_state = self.housing_transform(scaler)  
        
        elif type == "food":
            new_state = self.food_transform(scaler)
        
        elif type == "alloys":
            new_state = self.alloys_transform(scaler)
        
        elif type == "electronics":
            new_state = self.electronics_transform(scaler)
        
        elif type == "farm":
            new_state = self.farm_transform(scaler)

        new_state.adjust_continuals()
        return new_state

    def adjust_continuals(self):

        if self.water + (self.max_water * 0.1) < self.max_water:     #increasing water by 10%
            self.water += (self.max_water * 0.1)
        else:
            self.water = self.max_water

        self.food -= self.population        #  Each population eats 1 food
        if self.food < 0: self.food = 0

    def housing_transform(self, scaler: int):
        """ Performs the given housing transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """
            
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
        """ Performs the given alloys transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """
  
        new_state = copy.deepcopy(self)
        
        new_state.population -= 1*scaler
        new_state.metalic_elm -= 2*scaler
        
        new_state.population += 1*scaler
        new_state.metalic_alloys += 1*scaler
        new_state.metalic_waste += 1*scaler
        
        return new_state
            
    def electronics_transform(self, scaler: int):
        """ Performs the given electronics transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """

        new_state = copy.deepcopy(self)
        
        new_state.population -= 1*scaler
        new_state.metalic_elm -= 3*scaler
        new_state.metalic_alloys -= 2*scaler
        
        new_state.population += 1*scaler
        new_state.electronics += 2*scaler
        new_state.electronics_waste += 1*scaler
        
        return new_state

    def food_transform(self, scaler: int):
        """ Performs the given electronics transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """

        new_state = copy.deepcopy(self)
        
        new_state.water -= 10*scaler
        
        new_state.food += 1*scaler
        new_state.food_waste += int(0.5*scaler)
        
        return new_state
    
    def farm_transform(self, scaler: int):
        """ Performs the given electronics transformation.
        The amount is dictated by the passed in scaler.

        Parameters:
            scaler (int): Scaler amount for the transformation

        Returns:
            Country: New country after given transformation
        """

        new_state = copy.deepcopy(self)
        
        new_state.timber -= 5*scaler
        new_state.available_land -= 10*scaler
        new_state.water -= 10*scaler
        
        new_state.farm += 5*scaler
        new_state.farm_waste += 1*scaler
        
        return new_state
