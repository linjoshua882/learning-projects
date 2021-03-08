import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_csv('Uber_Data/uber-raw-data-apr14.csv')
data['Date/Time'] = pd.to_datetime(data['Date/Time'])

def get_dom(dt):
    return dt.day

data['dom'] = data['Date/Time'].map(get_dom)

def get_weekday(dt):
    return dt.weekday()

data['weekday'] = data['Date/Time'].map(get_weekday)

def get_hour(dt):
    return dt.hour

data['hour'] = data['Date/Time'].map(get_hour)

plt.hist(data.dom, bins = 30, rwidth = .8, range = (0.5, 30.5))
plt.title('Frequency of Pickups by Day of Month, 2014 Uber Data NYC')
plt.xlabel('Date of Month')
plt.ylabel('Frequency')
plt.show()

plt.hist(data.hour, bins = 24, range = (.5, 24))
plt.title('Frequency of Pickups by Hour of Day, 2014 Uber Data NYC')
plt.xlabel('Hour of Day (Total)')
plt.ylabel('Frequency')
plt.show()

plt.hist(data.weekday, bins=7, range=(-.5, 6.5), rwidth=.8)
plt.title('Frequency of Pickups by Day of Week, 2014 Uber Data NYC')
plt.xlabel('Day of Week')
plt.xticks(range(7), 'Mon Tues Wed Thurs Fri Sat Sun'.split())
plt.ylabel('Frequency')
plt.show()

x = data['Lon']
y = data['Lat']
# plt.hist2d(x, y, bins=546517, normed=False, cmap='plasma')
# cb = plt.colorbar()
# cb.set_label('Heatmap of Freq')

plt.plot(x, y, ms=1, alpha=0.5)
plt.title('Frequency of Pickups by Location, 2014 Uber Data NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-74.2, -73.7)
plt.ylim(40.7, 41)
plt.show()
