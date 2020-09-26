import numpy as np
from imageio import imread
from collections import Counter
from math import sqrt
from sklearn import svm , metrics
from matplotlib import pyplot as plt
import pickle

##Training

# f = open("G:\Stuff\Work\Computer RA Task\Task Dataset\Train\Training Labels.txt", "r")
#
# ## labels of training data in a 1D array
# labels = f.read().split('\n')

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
    

pickle.dump(listofdists, open(r'G:\Stuff\Work\Computer RA Task\model.sav', 'wb'))

listofdists = pickle.load(open(r'G:\Stuff\Work\Computer RA Task\model.sav', 'rb'))
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

print(np.argmax(list_of_error))
plt.ylabel('Error Value')
plt.xlabel('K Value')
plt.show()
'''
######################################################################################
##Testing

f_test = open("G:\Stuff\Work\Computer RA Task\Task Dataset\Test\Test Labels.txt", "r")
test_labels = f_test.read().split('\n')

## calculate distances of each image with all other images
# for i in range(1,200):
#     imga = imread(r'G:\Stuff\Work\Computer RA Task\Task Dataset\Test/N' + str(i) +'.jpg')
#     distances = []
#     for j in range(1,200):
#         imgb = imread(r'G:\Stuff\Work\Computer RA Task\Task Dataset\Test/N' + str(j) + '.jpg')
#         l = test_labels[j-1]
#         d = euclidian_distance(imga , imgb)
#         distances.append((d,l))
#
#     distances.sort()
#     distances.pop(0)
#     listofdists.append(distances)
#
# pickle.dump(listofdists, open(r'G:\Stuff\Work\Computer RA Task\test_model.sav', 'wb'))

listofdists = pickle.load(open(r'G:\Stuff\Work\Computer RA Task\test_model.sav', 'rb'))

K_test = 7
actual_labels = np.zeros(199)
predicted_label = np.zeros(199)


for index in range(1,200):
    actual_labels[index-1] = int(test_labels[index-1])
    x = listofdists[index -1]
    neighbours = x[:K_test]
    predicted_label[index-1] = vote(neighbours)




print('Accuracy:' ,metrics.accuracy_score(actual_labels, predicted_label) * 100,'%')


























