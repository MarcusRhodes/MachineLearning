# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import random as Random
import decimal
import numpy as np

class movie:
    def __init__(self, title = "Title", year = 2018, runtime = 60):
        self.title = title
        self.year = year if year >= 0 else 0
        self.runtime = runtime
        
    def __repr__(self):
        return "{} ({}) - {} min".format(self.title, self.year, self.runtime)
    
    def hours(self):
        hours = self.runtime / 60;
        minutes = self.runtime - (hours * 60)
        return "{} hours {} min".format(hours, minutes)
    
    
def create_movie_list():
    list = []
    jurassic = movie("Juraassic Park", 1993, 160)
    #print(jurassic)
    #print(jurassic.hours())

    jurassic2 = movie("Juraassic Park 2", 1997, 140)
    #print(jurassic2)
    #print(jurassic2.hours())

    jurassic3 = movie("Juraassic Park 3", 2001, 150)
    jurassicworld = movie("Juraassic World", 2015, 170)
    list.append(jurassic)
    list.append(jurassic2)
    list.append(jurassic3)
    list.append(jurassicworld)
    return list

movies = create_movie_list()
for film in movies:
    print(film)

print([film for film in movies if film.runtime >= 150])

stars = {}
for film in movies:
    stars[film.title] = float(decimal.Decimal(Random.randrange(0, 599))/100)
    
for film in stars:
    print("{}, rated {} stars!".format(film, stars[film]))

def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = Random.randint(100, 10000)
        stars = Random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

data = get_movie_data()

print(data)

rows = data.shape[0]
cols = data.shape[1]

print("There are {} rows and {} cols".format(rows, cols))

print(data[0:2])

print(data[:,-2:])

print(data[:,1])








