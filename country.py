from dataclasses import dataclass
import numpy as np
import copy

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
class Country:
    
    name: str
    population: int
    metalic_elm: int
    timber: int
    metalic_alloys: int
    electronics: int
    housing: int
    state_reduction: int

    metalic_waste: int = 0     
    electronics_waste: int = 0     
    housing_waste: int = 0
        
    def state_value(self) -> float:
        
        resource_score =  self.metalic_alloys + self.timber + self.metalic_alloys
        developement_score = self.metalic_alloys + self.electronics + self.housing
        waste_score = self.metalic_waste + self.electronics_waste + self.housing_waste
        
        return round(resource_score + 3*developement_score - waste_score, 2)            # make this more complex at some point
    
    def make_trade(self, resource: str, amount: int):
        
        new_state = copy.deepcopy(self)
        
        if resource == 'metalic_elm':
            new_state.metalic_elm -= amount
        elif resource == 'timber':
            new_state.timber -= amount
        elif resource == 'metalic_alloys':
            new_state.metalic_alloys -= amount
        elif resource == 'electronics':
            new_state.electronics -= amount
        elif resource == 'housing':
            new_state.housing -= amount
        
        return new_state
                
    def can_housing_transform(self):
                
        if (self.population >= 5 and self.metalic_elm >= 1 
            and self.timber >= 5 and self.metalic_alloys >= 3): 
            
            scalers = []
            scalers.append(int(self.population / 5))
            scalers.append(int(self.metalic_elm / 1))
            scalers.append(int(self.timber / 5))
            scalers.append(int(self.metalic_alloys / 3))
            
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
        
        if (self.population >= 1, self.metalic_elm >= 2):       
            
            scalers = []
            scalers.append(int(self.population / 1))
            scalers.append(int(self.metalic_elm / 2))
            
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
        
        if (self.population >= 1 and self.metalic_elm >= 3
            and self.metalic_alloys >= 2):
            
            scalers = []
            scalers.append(int(self.population / 1))
            scalers.append(int(self.metalic_elm / 3))
            scalers.append(int(self.metalic_alloys / 2))

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