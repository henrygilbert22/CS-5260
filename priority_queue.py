from dataclasses import dataclass


@dataclass
class Solution:
    
    expected_utility: float
    path: list
    

class PriorityQueue:
    
    priority_queue: list[Solution]
    
    def __init__(self):
        
        self.priority_queue = []
  
    def push(self, data: tuple) -> None:
        
        self.priority_queue.append(data)
    
    def empty(self):
        
        return len(self.priority_queue) == 0
    
    def pop(self) -> object:
        
        max_value = 0
        
        for i in range(len(self.priority_queue)):
            
            if self.priority_queue[i].expected_utility > self.priority_queue[max_value].expected_utility:       # Comparing value
                max_value = i
        
        selected_item = self.priority_queue[max_value]
        del self.priority_queue[max_value]
        
        return selected_item
     