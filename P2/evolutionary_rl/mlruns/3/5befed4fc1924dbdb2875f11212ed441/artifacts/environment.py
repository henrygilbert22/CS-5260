from typing import Final
from country import Country
from country import ResourceWeights

from math import ceil
import pandas as pd
import random
from collections import deque
import numpy as np

class Environment:

    countries: list = []
    weights: ResourceWeights

    step_counter: int = 0
    countries_state_window: list = []

    max_value: int
    countries_step_counter: list = []

    def __init__(self) -> None:
        
        self.load_weights('Example-Sample-Resources.xlsx')
        self.load_countries('Example-Initial-Countries.xlsx')

        for i in range(5):
            state_window = deque(maxlen=5)
            [state_window.append(self.countries[i].state()) for _ in range(5)]
            self.countries_state_window.append(state_window)

        self.countries_step_counter = [0 for _ in range(len(self.countries))]

    def reset(self):
        
        self.load_weights('Example-Sample-Resources.xlsx')
        self.load_countries('Example-Initial-Countries.xlsx')
        self.countries_step_counter = [0 for _ in range(len(self.countries))]

        for i in range(5):
            state_window = deque(maxlen=5)
            [state_window.append(self.countries[i].state()) for _ in range(5)]
            self.countries_state_window.append(state_window)

    def current_state(self, country_index: int):

        return np.array(self.countries_state_window[country_index])/self.max_value

    def load_countries(self, file_name: str):
        """Loads the countries from the given csv file

        Parameters:
            file_name (str): File name for the csv holding the countries
            
        Returns:
            None
        """

        self.countries = []
        df = pd.read_excel(file_name)
        self.max_value = 0

        for index, row in df.iterrows():
            
            args = list(row.values) + [self.weights]
            self.countries.append(Country(*args))
            self.max_value = max(self.max_value, max(list(row.values)[1:]))

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

    def facilatate_trade(self, country_index: int):

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

        for i in range(len(self.countries)):

            if i == country_index:
                continue

            tradeable_resources = {
            'metalic_elm': self.countries[i].metalic_elm,
            'timber': self.countries[i].timber,
            'available_land': self.countries[i].available_land,
            'water': self.countries[i].water,
            }

            tradeable_resources = dict(sorted(tradeable_resources.items(), key=lambda item: item[1], reverse=True))
            biggest_resource = external_biggest_resource(internal_biggest_resource, list(tradeable_resources.keys()))
            countries_trade_value[i] = self.weights[biggest_resource] * tradeable_resources[biggest_resource]
            countries_trade_resource[i] = biggest_resource

        countries_trade_value = dict(sorted(countries_trade_value.items(), key=lambda item: item[1], reverse=True))
        best_country_to_trade = list(countries_trade_value.keys())[0]
        
        countries_max_trade_amount = self.countries[best_country_to_trade][countries_trade_resource[best_country_to_trade]]
        self_max_amount = self.countries[country_index][internal_biggest_resource]
        max_amount = max(countries_max_trade_amount, self_max_amount)

        other_elm_scale = 1 / self.weights[internal_biggest_resource] * 0.9     # Making the trade 10% more favorable for us
        self_elm_scale = 1 / self.weights[countries_trade_resource[best_country_to_trade]] 
        other_elm_max = ceil(max_amount / other_elm_scale)
        self_elm_max = ceil(max_amount / self_elm_scale)

        if np.random.rand() < 0.9:  # 90% chance of trading because it's 10% more profitable for us
            self.countries[country_index].make_trade(internal_biggest_resource, self_elm_max, countries_trade_resource[best_country_to_trade], other_elm_max)
            self.countries[best_country_to_trade].make_trade(countries_trade_resource[best_country_to_trade], other_elm_max, internal_biggest_resource, self_elm_max)

    def step(self, action: int, country_index: int):
        
        previous_value = self.curr_state_value(country_index)
        reward = lambda curr, previous: round(curr/previous, 2) if curr > previous else round(0.9 - previous/curr, 2)

        if action == 0:
            pass    

        elif action == 1:
            self.facilatate_trade(country_index)

        elif action == 2:
            self.countries[country_index].housing_transform()

        elif action == 3:
            self.countries[country_index].alloys_transform()
        
        elif action == 4:
            self.countries[country_index].electronics_transform()
        
        elif action == 5:
            self.countries[country_index].food_transform()
        
        elif action == 6:
            self.countries[country_index].farm_transform()
        
        self.countries_step_counter[country_index] += 1
        self.countries_state_window[country_index].append(self.countries[country_index].state())

        r = reward(self.curr_state_value(country_index), previous_value)

        if self.countries_step_counter[country_index] >= 100:
            return r, True

        return r, False
    
    def curr_state_value(self, country_index: int):

        return self.countries[country_index].state_value()

    def all_countries_finished(self):

        for i in range(len(self.countries)):
            if self.countries_step_counter[i] <100:
                return False

        return True

    def country_finished(self, country_index: int):

        return self.countries_step_counter[country_index] >= 100
