from dataclasses import dataclass, field
import heapq
from typing import Any


@dataclass(order=True)
class Solution:
    
    priority: float
    path: Any=field(compare=False)
    
    def __init__(self, p: int, path: list) -> None:
        
        self.priority = p
        self.path = path
        
    def print(self):
        
        print(f"Expected Utility for Solution: {self.priority}")
        for p in self.path:
            
            if p[0] != None:
                p[0].print()
            
            
        
    
class PriorityQueue:
    
    queue: list
    max_size: int
    
    def __init__(self, maxsize: int):
        
        self.max_size = maxsize
        self.queue = []
               
    def push(self, item: Solution):
        
        if len(self.queue) > self.max_size:
            heapq.heappop(self.queue)
        
        heapq.heappush(self.queue, item)
    
    def pop(self):
        
        return heapq.heappop(self.queue)

    def empty(self):
        
        return len(self.queue) == 0
     