hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# create list areas
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# house information as list of lists
house = [["hallway", hall], 
        ["kitchen", kit], 
        ["living room", liv],
        ["bedroom", bed],
        ["bathroom", bath]]

#print(house)
#print(type(house))

#print(areas[1])
#print(areas[-1])
#print(areas[5])

eat_sleep_area = areas[3] + areas[-3]
#print(eat_sleep_area)

downstairs = areas[0:6]
upstairs = areas[6:10]

#print(downstairs)
#print(upstairs)

areas[-1] = 10.50
areas[4] = "chill zone"

areas_1 = areas + ["poolhouse", 24.5]
areas_2 = areas_1 + ["garage", 15.45]

areas_copy = list(areas)
areas_copy[0] = 5.0

# print(areas)

# print(areas.index(20.0))
# print(areas.count(10.75))

areas.append("living room")
areas.append(24.5)
areas.append("bathroom")
areas.append(15.45)

areas.reverse()

print(areas)