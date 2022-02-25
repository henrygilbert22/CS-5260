from dataclasses import dataclass, field
import heapq
from typing import Any
from unittest.mock import NonCallableMagicMock


@dataclass(order=True)
class Solution:
    
    priority: float
    path: Any=field(compare=False)
    
    def __init__(self, p: int, path: list) -> None:
        """Initilization function to set the prioirty
        and the current path

        Parameters:
            p (int): Priority or EU
            path (list): List of states and transforms
            
        Returns:
            None
        """
        
        self.priority = p
        self.path = path
        
    def print(self, file_name: str):
        """Function to pretty print the solution
        
        Arguements:
            None
            
        Returns:
            None
        """
        
        with open(file_name, 'a') as f:
            
            print(f"Expected Utility for Total Solution: {round(self.priority, 2)}\n")
            f.write(f"Expected Utility for Total Solution: {round(self.priority, 2)}\n")
            for p in self.path: 
                
                if p[0] != None:
                    print(f'Expected Utility for This Action: {p[3]}')
                    f.write(f'Expected Utility for This Action: {p[3]}')
                    f.write(p[0].print())
            
            
            
            
        
    
class PriorityQueue:
    
    queue: list
    max_size: int
    
    def __init__(self, maxsize: int):
        """Initilization function to set the maxsize
        of the priority queue

        Parameters:
            maxsize (int): Size of the queue
        """
        
        self.max_size = maxsize
        self.queue = []
               
    def push(self, item: Solution):
        """Adds new item into the queue utilizing heapq
        pop and push

        Parameters:
            item (Solution): New item to be added
        
        Returns:
            None
        """
                
        if len(self.queue) > self.max_size:
            heapq.heappop(self.queue)
        
        heapq.heappush(self.queue, item)

    def pop(self):
        """Pops the highest EU solution from the queue

        Returns:
            Solution: Highest EU solution in given queue
        """
        
        return heapq.heappop(self.queue)

    def empty(self):
        """Returns bool if the queue is empty or not

        Returns:
            bool: If the queue is empty
        """
        
        return len(self.queue) == 0
     