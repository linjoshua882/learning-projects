import matplotlib.pyplot as plt
import numpy as np

year = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
pop = [2.53, 2.57, 2.62, 2.76, 2.71, 2.67, 2.86, 2.92, 2.97, 3.03, 3.08, 3.14, 3.2, 3.26, 3.33, 3.34, 3.44, 3.22, 3, 2.9, 3.4, 3.6]

# print(year[-1])
# print(pop[-1])

# plt.plot(year, pop)

# plt.xscale('log')
# plt.yscale('log')

# np_pop = np.array(pop)
# np_pop = np_pop / 2

# plt.scatter(year, pop, s = np_pop)

# plt.hist(pop, 20)

plt.scatter(x = year, y = pop, alpha = 0.8)

xlab = 'Year'
ylab = 'Population'
title = 'Chart Practice'

plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)
plt.grid(True)

# plt.xticks()
# plt.yticks()

plt.show()