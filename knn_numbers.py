import numpy as np
from imageio import imread
from collections import Counter
from math import sqrt
from matplotlib import pyplot as plt
import pickle

img = imread(r'G:\Stuff\Work\Computer RA Task\Task Dataset\Train/N1000.jpg')
f = open("G:\Stuff\Work\Computer RA Task\Task Dataset\Train\Training Labels.txt", "r")

## labels of training data in a 1D array
labels = f.read().split('\n')

listofdists = []

## Calculate the euclidian distance between two images
def euclidian_distance(x1,x2):
    dist = 0.0
    sum = (x1-x2)**2
    dist = np.sum(sum)
    return sqrt(dist)

## vote for the class of test image
def vote(neighbours):
    classvotes = np.zeros((10,1))
    for x in neighbours:
        i = int(x[1])
        classvotes[i] += 1

    return np.argmax(classvotes)
'''
## calculate distances of each image with all other images
for i in range(1,2400):
    imga = imread(r'G:\Stuff\Work\Computer RA Task\Task Dataset\Train/N' + str(i) +'.jpg')
    distances = []
    for j in range(1,2400):
        imgb = imread(r'G:\Stuff\Work\Computer RA Task\Task Dataset\Train/N' + str(j) + '.jpg')
        l = labels[j-1]
        d = euclidian_distance(imga , imgb)
        distances.append((d,l))

    distances.sort()
    distances.pop(0)
    listofdists.append(distances)
    

pickle.dump(listofdists, open(r'G:\Stuff\Work\model.sav', 'wb'))
'''
listofdists = pickle.load(open(r'G:\Stuff\Work\model.sav', 'rb'))
list_of_error = []
## Finding the best K using LOOCV
for k in range(1,100):
    error = 0.0
    b = 1
    for x , val in enumerate(listofdists):
        actual = int(labels[x])
        neighbours = val[:k]
        prediction = vote(neighbours)
        if prediction != actual:
            error += 1


    list_of_error.append(error)
    plt.plot(k, list_of_error[k - 1],'ro')

print(np.argmin(list_of_error))
plt.ylabel('Error Value')
plt.xlabel('K Value')
plt.show()


























