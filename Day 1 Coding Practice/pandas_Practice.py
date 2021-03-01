import pandas as pd 

names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

my_dict = {'country':names, 'drives_right':dr, 'cars_per_cap':cpc}

cars = pd.DataFrame(my_dict)

row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
cars.index = row_labels


# print(cars[0:3])
# print(cars[3:6])

# print(cars)