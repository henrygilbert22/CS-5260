from country import Country
from country import ResourceWeights

from math import ceil
import pandas as pd
import random
from collections import deque
import numpy as np

class Environment:

    country: Country
    other_countries: list
    weights: ResourceWeights

    step_counter: int = 0
    state_window: deque = deque(maxlen=5)

    max_value: int

    def __init__(self) -> None:
        
        self.load_weights('Example-Sample-Resources.xlsx')
        self.load_countries('Example-Initial-Countries.xlsx')

        [self.state_window.append(self.country.state()) for _ in range(5)]

    def reset(self):
        
        self.load_weights('Example-Sample-Resources.xlsx')
        self.load_countries('Example-Initial-Countries.xlsx')
        self.step_counter = 0
        [self.state_window.append(self.country.state()) for _ in range(5)]

    def current_state(self):

        return np.array(self.state_window)/self.max_value

    def random_action(self):

        return random.randint(0, 6)

    def load_countries(self, file_name: str):
        """Loads the countries from the given csv file

        Parameters:
            file_name (str): File name for the csv holding the countries
            
        Returns:
            None
        """

        self.other_countries = []
        df = pd.read_excel(file_name)
        self.max_value = 0

        for index, row in df.iterrows():
            
            args = list(row.values) + [self.weights]
            self.other_countries.append(Country(*args))
            self.max_value = max(self.max_value, max(list(row.values)[1:]))

        self.country = self.other_countries[4]
        del self.other_countries[4] # Remove the country with the same name as the current country

    def load_weights(self, file_name: str):
        """Loads the weights of resources from the given file

        Args:
            file_name (str): File containing weights of resources
            
        Returns:
            None
        """

        df = pd.read_excel(file_name)
        args = pd.Series(df.Weight.values).to_list()
        self.weights = ResourceWeights(*args)

    def facilatate_trade(self):

        external_biggest_resource = lambda interal_r, external_rs: external_rs[0] if external_rs[0] != interal_r else external_rs[1]

        tradeable_resources = {
            'metalic_elm': self.country.metalic_elm,
            'timber': self.country.timber,
            'available_land': self.country.available_land,
            'water': self.country.water,
        }

        tradeable_resources = dict(sorted(tradeable_resources.items(), key=lambda item: item[1], reverse=True))
        internal_biggest_resource = list(tradeable_resources.keys())[0]

        countries_trade_value = {}
        countries_trade_resource = {}
        for i in range(len(self.other_countries)):

            tradeable_resources = {
            'metalic_elm': self.other_countries[i].metalic_elm,
            'timber': self.other_countries[i].timber,
            'available_land': self.other_countries[i].available_land,
            'water': self.other_countries[i].water,
            }

            tradeable_resources = dict(sorted(tradeable_resources.items(), key=lambda item: item[1], reverse=True))
            biggest_resource = external_biggest_resource(internal_biggest_resource, list(tradeable_resources.keys()))
            countries_trade_value[i] = self.weights[biggest_resource] * tradeable_resources[biggest_resource]
            countries_trade_resource[i] = biggest_resource
        
        countries_trade_value = dict(sorted(countries_trade_value.items(), key=lambda item: item[1], reverse=True))
        best_country_to_trade = list(countries_trade_value.keys())[0]
        
        countries_max_trade_amount = self.other_countries[best_country_to_trade][countries_trade_resource[best_country_to_trade]]
        self_max_amount = self.country[internal_biggest_resource]
        max_amount = max(countries_max_trade_amount, self_max_amount)

        other_elm_scale = 1 / self.weights[internal_biggest_resource] * 0.9     # Making the trade 10% more favorable for us
        self_elm_scale = 1 / self.weights[countries_trade_resource[best_country_to_trade]] 
        other_elm_max = ceil(max_amount / other_elm_scale)
        self_elm_max = ceil(max_amount / self_elm_scale)

        if np.random.rand() < 0.9:  # 90% chance of trading because it's 10% more profitable for us
            self.country.make_trade(internal_biggest_resource, self_elm_max, countries_trade_resource[best_country_to_trade], other_elm_max)
            self.other_countries[best_country_to_trade].make_trade(countries_trade_resource[best_country_to_trade], other_elm_max, biggest_resource, self_elm_max)

    def step(self, action: int):
        
        previous_value = self.curr_state_value()
        reward = lambda curr, previous: 1 if curr > previous else -1

        if action == 0:
            pass    

        elif action == 1:
            self.facilatate_trade()

        elif action == 2:
            self.country.housing_transform()

        elif action == 3:
            self.country.alloys_transform()
        
        elif action == 4:
            self.country.electronics_transform()
        
        elif action == 5:
            self.country.food_transform()
        
        elif action == 6:
            self.country.farm_transform()
        
        self.step_counter += 1
        self.state_window.append(self.country.state())
        r = reward(self.curr_state_value(), previous_value)

        if self.step_counter >= 100:
            return r, True
            
        return r, False
    
    def curr_state_value(self):

        return self.country.state_value()