from simulation import Simulation

import time

def test_case_1():
   
    s = Simulation(
        'tests/1/case_1_countries.xlsx',       # Countries file
        'tests/1/case_1_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        0.1                                     # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/1/output.txt')

def test_case_2():
   
    s = Simulation(
        'tests/2/case_2_countries.xlsx',       # Countries file
        'tests/2/case_2_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        -500                                     # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/2/output.txt')

def test_case_3():
   
    s = Simulation(
        'tests/3/case_3_countries.xlsx',       # Countries file
        'tests/3/case_3_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        1                                   # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/3/output.txt')
    
def test_case_4():
   
    s = Simulation(
        'tests/4/case_4_countries.xlsx',       # Countries file
        'tests/4/case_4_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        1                                   # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/4/output.txt')

def test_case_5():
   
    s = Simulation(
        'tests/5/case_5_countries.xlsx',       # Countries file
        'tests/5/case_5_weights.xlsx',        # Resources file
        'Erewhon',                              # Self country
        4,                                      # Depth
        0.8,                                    # Gamma
        -1,                                     # State Reduction (-1 for the most)
        1000,                                   # Frontier size
        3,                                      # Solution size
        1                                   # C        
    )

    start = time.time()
    s.search_parallel()     # need to reformat parallel as well
    end = time.time()

    print(f"Took: {end-start}")

    best_solution = s.solutions.queue[-1]
    best_solution.print(f'tests/5/output.txt')
    

def main():

    choice = input("Enter test case number [1,5]: ")

    if choice == '1':
        test_case_1()
    elif choice == '2':
        test_case_2()
    elif choice == '3':
        test_case_3()
    elif choice == '4':
        test_case_4()
    elif choice == '5':
        test_case_5()