from dataclasses import dataclass


@dataclass
class Solution:
    
    expected_utility: float
    path: list
    

class PriorityQueue:
    
    priority_queue: list[Solution]
    max_size: int
    
    def __init__(self, max_size: int):
        
        self.priority_queue = []
        self.max_size = max_size
  
    def push(self, data: tuple) -> None:
        
        self.priority_queue.append(data)
        
        if len(self.priority_queue) > self.max_size:
            self.priority_queue.sort(key=lambda x: x.expected_utility)
            del self.priority_queue[0]
    
    def empty(self):
        
        return len(self.priority_queue) == 0
    
    def pop(self) -> object:
        
        self.priority_queue.sort(key=lambda x: x.expected_utility)
        max_item = self.priority_queue[-1]
        del self.priority_queue[-1]
        
        return max_item
        
     