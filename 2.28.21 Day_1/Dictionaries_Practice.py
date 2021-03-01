countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

ind_ger = countries.index('germany')

# print(capitals[ind_ger])

europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo'}

# print(europe.keys())
# print(europe['norway'])

europe['italy'] = 'rome'
print('italy' in europe)
europe['poland'] = 'warsaw'

# print(europe)