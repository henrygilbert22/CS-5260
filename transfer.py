from dataclasses import dataclass

@dataclass
class Transfer:
    
    resource_1_name: str
    resource_2_name: str
    
    resource_1_amount: str
    resource_2_amount: str
    
    c_1_name: str
    c_2_name: str
    
    def print(self):
        """Prints the given transfer
        
        Parameters:
            None
            
        Returns:
            None
        """
        
        print("TRANSFER:")
        print(f"    {self.c_1_name} - {self.c_2_name}")
        print(f"    {self.resource_1_name} - {self.resource_2_name}")
        print(f"    {self.resource_1_amount} - {self.resource_2_amount}")
        print()
        