#Özge Günay 150114027

import numpy as np
import nltk

#hypothesis function
def g(weight, x):
    return np.sign(np.dot(x, weight))

#Calculating in-sample error
def calInSampleError(weight, trainingSet, trainingClassification):
    error = 0
    for i in range(len(trainingSet)):
        res = g(weight, trainingSet[i])
        if(res != trainingClassification[i]):
            error += 1
    return error/len(trainingSet)*1.0

#Calculating out sample error
def calOutSampleError(weight, testingInput, testingClassification):
    error = 0
    for i in range(len(testingInput)):
        expected = testingClassification[i]
        hypothesisResult = g(weight, testingInput[i])
        if(expected != hypothesisResult):
            error += 1
    return error/len(testingInput)*1.0    

#Reading from file
trainingData = open("in.dta", "r")
testData = open("out.dta", "r")
lines = trainingData.readlines()
testLines = testData.readlines()

trainingInput = []
trainingClassification = []
testingInput = []
testingClassification = []

#Transforming the training data
for line in lines:
    data = nltk.word_tokenize(line)
    in1 = float(data[0])
    in2 = float(data[1])
    trainingInput.append([1, in1, in2, in1**2, in2**2, in1*in2, abs(in1-in2), abs(in1+in2)])
    trainingClassification.append(float(data[2]))

#Transforming the testing data
for row in testLines:
    testData = nltk.word_tokenize(row)
    testIn1 = float(testData[0])
    testIn2 = float(testData[1])
    testingInput.append([1, testIn1, testIn2, testIn1**2, testIn2**2, testIn1*testIn2, abs(testIn1-testIn2), abs(testIn1+testIn2)])
    testingClassification.append(float(testData[2]))

#Updating the weight and calculating the in sample, out sample errors
weight = np.matmul(np.linalg.pinv(trainingInput), trainingClassification)
inSampleError = calInSampleError(weight, trainingInput, trainingClassification)
outSampleError = calOutSampleError(weight, testingInput, testingClassification)

print("In Sample Error is " + str(inSampleError))
print("Out Sample Error is " + str(outSampleError))


