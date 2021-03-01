import numpy as np

True == False
-5 * 15 != 75
"pyscript" == "PyScript"
True == 1

# Comparison of integers
x = -3 * 6
print(x >= -10)

# Comparison of strings
y = "test"
print("test" <= y)

# Comparison of booleans
True > False

my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

print(my_house >= 18)
print(my_house < your_house)

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

print(my_kitchen > 10 and my_kitchen < 18)

print(my_kitchen < 14 or my_kitchen > 17)

print((my_kitchen * 2) < (your_kitchen * 3))

my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])


print(np.logical_or(my_house > 18.5,
                    my_house < 10))


print(np.logical_and(my_house < 11,
                    your_house < 11))
