from dataclasses import dataclass

@dataclass
class Country:
    
    name: str
    r1: int
    r2: int
    r3: int
    r21: int
    r22: int
    r23: int
    
@dataclass
class ResourceWeights:
    
    r1: int
    r2: int
    r3: int
    r21: int
    r22: int
    r23: int
    r21w: int       # Waste for r21
    r22w: int       # Waste for r22
    r22w: int       # Waste for r23
def main():
    return

if __name__ == '__main__':
    main()