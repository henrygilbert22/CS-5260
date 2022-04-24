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
        
        output_string = f"""
        TRANSFER:
            OTHER   -   SELF
            {self.c_1_name} -   {self.c_2_name}
            {self.resource_1_name}  -   {self.resource_2_name}
            {self.resource_1_amount}  -   {self.resource_2_amount}\n
        """
        print(output_string)
        return output_string
        