#Self organizing map clustering algorithm
# The Academician
import csv
import numpy as np, numpy.random
from scipy.spatial import distance
np.set_printoptions(suppress=True) #Force-suppress all exponential notation

""" file = open('mostTraversedLatLong.csv')
type(file)
csvreader = csv.reader(file)

header = []
header = next(csvreader)
header

rows = []
for row in csvreader:
    rows.append(row)
    rows """
k = 2
p = 0
alpha = 0.7 # Initial learning rate

rows = np.array([
        [1,1,0,0], 
       [0,1,0,0], 
       [0,0,1,0], 
       [0,0,1,1]])

#X = np.fromfile


# Print the number of data and dimension 
n = len(rows)
d = len(rows[0])
addZeros = np.zeros((n, 1))
rows = np.append(rows, addZeros, axis=1)
print("The SOM algorithm: \n")
print("The training data: \n", rows)
print("\nTotal number of data: ",n)
print("Total number of features: ",d)
print("Total number of Clusters: ",k)

C = np.zeros((k,d+1))

weight = np.random.rand(n,k)
print("\nThe initial weight: \n", np.round(weight,2))

for it in range(100): # Total number of iterations
    for i in range(n):
        distMin = 99999999
        for j in range(k):
            dist = np.square(distance.euclidean(weight[:,j], rows[i,0:d]))
            if distMin>dist:
                distMin = dist
                jMin = j
        weight[:,jMin] = weight[:,jMin]*(1-alpha) + alpha*rows[i,0:d]   
    alpha = 0.5*alpha
    
print("\nThe final weight: \n",np.round(weight,4))

for i in range(n):    
    cNumber = np.where(weight[i] == np.amax(weight[i]))
    rows[i,d] = cNumber[0]
    
print("\nThe data with cluster number: \n", rows)

#file.close()