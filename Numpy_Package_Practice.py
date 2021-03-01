import numpy as np 

height_in = [67, 84, 76, 54, 62]
weight_lb = [118, 160, 140, 130, 135]

# weight_lb_sorted = sorted(weight_lb, reverse = False)

#print(weight_lb_sorted)

np_height_in = np.array(height_in)

np_height_m = np_height_in * 0.0254
np_weight_kg = np.array(weight_lb) * 0.453592
bmi = np_weight_kg / (np_height_m ** 2)

light = np.array(bmi < 21)

# print(light)

# print(np_height_in[1:4])

baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

np_baseball = np.array(baseball)

#print(type(np_baseball))

#print(np_baseball.shape)

np_heightx = np.array(np_baseball[:,0])

print(np.mean(np_heightx))
print(np.median(np_heightx))
print(np.std(np_heightx)
print(np.corrcoef(np_baseball[:,0], np_baseball[:,1]))