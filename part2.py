#Özge Günay 150114027

import random
import numpy as np
import math

#Creating target function
def target():
    point1x1 = random.uniform(-1, 1)
    point1x2 = random.uniform(-1, 1)
    point2x1 = random.uniform(-1, 1)
    point2x2 = random.uniform(-1, 1)
    a = abs(point1x1-point2x1)/abs(point1x2-point2x2)
    b = point1x2 - a*point1x1

    def f(x): return a*x+b

    return f

#hypothesis function
def g(weight, x):
    s = np.dot(x, weight)
    theta = math.exp(s)/(1+math.exp(s))
    if(theta > 0.5):
        return 1
    else:
        return -1

#Compares the classification with the label
def compareResult(y,expectedY):
    if(y > expectedY):
        return 1
    else:
        return -1

#Calculating out sample error
def calOutSampleError(weight, f):
    errorArray = []
    for i in range(10000):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        yn = g(weight, [1.0, x1, x2])
        xn = [1.0, x1, x2]
        errorArray.append(np.log(1+ math.exp(-1*yn*np.matmul(np.transpose(weight),xn))))
    return  sum(errorArray)/10000
       

epochArray = []
outSampleArray = []
#Repeating the experiment for 1000 times
for itr in range(100):
    previousWeight = np.zeros(3)
    w = np.zeros(3)
    f = target()
    #Creating training input 
    trainingInput = [[1.0, random.uniform(-1,1), random.uniform(-1,1)] for i in range(100)]
    #Classifying the inputs
    trainingClassification = [compareResult(f(x[1]), x[2]) for x in trainingInput]
    #Gethering the inputs and classification for the shuffling porcess
    trainingData = [[trainingInput[index][0], trainingInput[index][1], trainingInput[index][2], trainingClassification[index]] for index in range(len(trainingInput))]
    #Giving the values
    thrs = 0.01
    learn_rate = 0.01
    distance = 1.0
    epoch = 0

    while(distance >= thrs):
        #Shuffling
        random.shuffle(trainingData)
        #Seperating the inputs and classification
        trainingInput = [[trainingData[i][0], trainingData[i][1], trainingData[i][2] ] for i in range(len(trainingData))]
        trainingClassification = [trainingData[i][3] for i in range(len(trainingData))]
        #For all data points, updating the weight
        for item in range(len(trainingInput)):
            egrad = -1*(trainingClassification[item]*np.array(trainingInput[item]))
            egrad = egrad/(1+ math.exp(trainingClassification[item]*np.matmul(np.transpose(w),trainingInput[item])))
            w = np.subtract(w,learn_rate*egrad)
        distance = np.linalg.norm(previousWeight-w) 
        previousWeight = w
        epoch += 1
    #Calculating the out sample error and epoch value
    outSampleError = calOutSampleError(w, f)
    outSampleArray.append(outSampleError)
    epochArray.append(epoch)

#Finding the average values
print("The average out of sample error is " + str(sum(outSampleArray)/len(outSampleArray)*1.0) )
print("The average epoch is " + str(sum(epochArray)/len(epochArray)*1.0) )