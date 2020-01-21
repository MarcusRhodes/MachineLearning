
import numpy as np
from statistics import mode

x = np.array([3, 6])
y = np.array([5, 2])
data = np.array([[2, 3], [3, 4], [5, 7], [2, 7], [3, 2], [1, 2], [9, 3], [4, 1]])
animals = ["dog", "cat", "bird", "fish", "fish", "dog", "cat", "dog"]


def distance(x, y):
    return np.linalg.norm(x-y)

list = []
for d in data:
    list.append(distance(x, d))

    
#returns the indexes of the values smallest first
list = np.argsort(list)

print(list)   
for i in range(2):
    print(animals[list[i]])

#print(mode(animals))