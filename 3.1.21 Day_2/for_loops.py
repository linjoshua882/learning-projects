import numpy as np
import pandas as pd

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# for loop
for areas in areas:
    print(areas)

# enumerate()
for index, area in enumerate(areas) :
    print("room " + str(index + 1) + ": " + str(area))

# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# house for loop
for x in house :
    print("the " + x[0] + " is " + str(x[1]) + " sqm")

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for key, value in europe.items():
    print("the capital of " + key + " is " + str(value))

# For loop (2d Numpy Array)
# for value in np.nditer(np_baseball):
#    print(str(value))

