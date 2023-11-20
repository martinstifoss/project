import cv2
import random
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

#Database
connection = sqlite3.connect('database.db')
cursor = connection.cursor()

#Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS database (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        air_temperature REAL,
        wind_speed REAL,
        longitude REAL,
        latitude REAL,
        top_road_count INTEGER,
        bottom_road_count INTEGER
    )
''')
connection.commit()

count = 0
count1 = 0
antall_bilder = 7

def to_database(date, temperature, wind_speed, longitude1, latitude1, top_road_count, bottom_road_count):
    #Remove milliseconds
    date1 = date.strftime('%Y-%m-%d %H:%M:%S')

    #Insert to database
    cursor.execute('''
        INSERT INTO database (date, air_temperature, wind_speed, longitude, latitude, top_road_count, bottom_road_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (date1, temperature, wind_speed, longitude1, latitude1, top_road_count, bottom_road_count))
    connection.commit()

def filter_coordinates(coordinates, threshold):
    filtered_coordinates = []
    while len(coordinates[0]) > 0:
        i = 0
        close_points = np.where(
            (np.abs(coordinates[0] - coordinates[0, i]) < threshold) &
            (np.abs(coordinates[1] - coordinates[1, i]) < threshold)
        )
        if len(close_points[0]) > 0:
            filtered_coordinates.append((
                int(np.mean(coordinates[1, close_points])),  #Fix x and y
                int(np.mean(coordinates[0, close_points]))
            ))
            coordinates = np.delete(coordinates, close_points, axis=1)
    return np.array(filtered_coordinates).T


def template_matching(image_match, template_match, threshold):
    global count
    global count1

    #Loading Images
    img = cv2.imread(image_match)
    template = cv2.imread(template_match)

    #Grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #Storing width and scale
    w, h = template_grey.shape[::-1]

    #Canny Edge Detection
    img_edge = cv2.Canny(img_grey, 50, 150)
    template_edge = cv2.Canny(template_grey, 50, 150)

    #Match operations
    result = cv2.matchTemplate(img_grey, template_grey, cv2.TM_CCOEFF_NORMED)

    #Store coordinates
    loc = np.where(result >= threshold)

    #For filtering
    loc_arrays = [arr.flatten() for arr in loc]
    #Check if no templatematch
    if len(loc_arrays) == 0 or len(loc_arrays[0]) == 0:
        #Show detected cars
        cv2.imshow('Detected', img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        print("Number of cars found:", 0)
        print("Top road count:", count)
        print("Bottom road count:", count1)
        date = datetime.now()
        to_database(date, air_temperature, wind_speed, longitude1, latitude1, count, count1)
        return
    filtered_coordinates = filter_coordinates(np.array(loc_arrays), threshold=100)
    number_of_instances = len(set(zip(filtered_coordinates[0], filtered_coordinates[1])))

    #Count cars on each road
    for coord in filtered_coordinates.T:
        x, y = coord
        if y < 180:
            count += 1  # Below the line
        else:
            count1 += 1  # Above the line

    #Draw rectangles
    for pt in filtered_coordinates.T:
        x, y = pt
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    #Show detected cars
    cv2.imshow('Drone Footage', img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    print("Number of cars found:", number_of_instances)
    print("Top road count:", count)
    print("Bottom road count:", count1)
    date = datetime.now()
    to_database(date, air_temperature, wind_speed, longitude1, latitude1, count, count1)


#All assets used
#GPS Coordinates
latitude1 = 59.914
longitude1 = 10.752

#Pictures
templateimg = "Template2.jpg"
bil7 = "bilene.jpg"
bil6 = "bilane.jpg"
bil9 = "multiple1.jpg"
bil8 = "multiple.jpg"
bil4 = "empty.jpg"
bil3 = "Biil.jpg"
bil2 = "Bil.jpg"
bil1 = "Bilerr.jpg"
bil0 = "Biler.jpg"
bil_liste = [bil0,bil1,bil2,bil3,bil4,bil6,bil7,bil8,bil9]

#Getting date for WeatherAPI
date2 = datetime.now()
date3 = date2 - timedelta(days=1)
date22 = date2.strftime('%Y-%m-%d')
date33 = date3.strftime('%Y-%m-%d')
dato = f"{date33}/{date22}"


#Weather API
client_id = '24c15c97-c1e2-4cb9-849f-27ab5e32024a'
endpoint = 'https://frost.met.no/observations/v0.jsonld'
parameters = {
    'sources': 'SN18700,SN90450',
    'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D),mean(wind_speed P1D)',
    'referencetime': dato,
}
req = requests.get(endpoint, parameters, auth=(client_id,''))
json = req.json()

#Check for errors
if req.status_code == 200:
    data1 = json['data']
    print('API Data retrieved')
else:
    print('Error! Returned status code %s' % req.status_code)
    print('Message: %s' % json['error']['message'])
    print('Reason: %s' % json['error']['reason'])

#Extract air temp and wind information
for entry in data1:
    air_temperature = entry['observations'][0]['value']
    wind_speed = entry['observations'][2]['value']
dfs = []
for i in range(len(data1)):
    row = pd.DataFrame(data1[i]['observations'])
    row['referenceTime'] = data1[i]['referenceTime']
    row['sourceId'] = data1[i]['sourceId']
    dfs.append(row)

df = pd.concat(dfs, ignore_index=True)
df.head()

threshold1 = 0.4

#60km/h road, 16,67m/s, 35m long road.
#A picture every 3 seconds will suffice since the speed limit is 60 and road is 35m.
for _ in range(antall_bilder):
    bil_bilde = random.choice(bil_liste)
    template_matching(bil_bilde, templateimg, threshold1)


cursor.execute('SELECT * FROM database')
rows = cursor.fetchall()

#Print results
print("Image, Date and time, Air Temp, Wind Speed, Longitude, Latitude, Top road count, Bottom road count")
for row in rows:
    print(row)

connection.close()

#Optimizing traffic
if(count == count1):
    print("Both roads are the same.")
if(count < count1):
    print("Top road has less traffic.")
if(count1 < count):
    print("Bottom road has less traffic.")