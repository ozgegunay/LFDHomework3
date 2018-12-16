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

#Creating identity matrix
identity = np.identity(8)

#Updating weights with the given k values
weight = np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(trainingInput), trainingInput), identity * 10**-3)), np.transpose(trainingInput)),trainingClassification)
weight2 = np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(trainingInput), trainingInput), identity * 10**3)), np.transpose(trainingInput)),trainingClassification) 
weight3 = np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(trainingInput), trainingInput), identity * 10**-2)), np.transpose(trainingInput)),trainingClassification)
weight4 = np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(trainingInput), trainingInput), identity * 10**-1)), np.transpose(trainingInput)),trainingClassification)
weight5 = np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(trainingInput), trainingInput), identity * 10**0)), np.transpose(trainingInput)),trainingClassification)
weight6 = np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(trainingInput), trainingInput), identity * 10**1)), np.transpose(trainingInput)),trainingClassification)
weight7 = np.matmul(np.matmul(np.linalg.inv(np.add(np.matmul(np.transpose(trainingInput), trainingInput), identity * 10**2)), np.transpose(trainingInput)),trainingClassification)

#Calculating the in sample and out sample errors
inSampleError = calInSampleError(weight, trainingInput, trainingClassification)
outSampleError = calOutSampleError(weight, testingInput, testingClassification)

inSampleError2 = calInSampleError(weight2, trainingInput, trainingClassification)
outSampleError2 = calOutSampleError(weight2, testingInput, testingClassification)

inSampleError3 = calInSampleError(weight3, trainingInput, trainingClassification)
outSampleError3 = calOutSampleError(weight3, testingInput, testingClassification)

inSampleError4 = calInSampleError(weight4, trainingInput, trainingClassification)
outSampleError4 = calOutSampleError(weight4, testingInput, testingClassification)

inSampleError5 = calInSampleError(weight5, trainingInput, trainingClassification)
outSampleError5 = calOutSampleError(weight5, testingInput, testingClassification)

inSampleError6 = calInSampleError(weight6, trainingInput, trainingClassification)
outSampleError6 = calOutSampleError(weight6, testingInput, testingClassification)

inSampleError7 = calInSampleError(weight7, trainingInput, trainingClassification)
outSampleError7 = calOutSampleError(weight7, testingInput, testingClassification)

print("In Sample Error of k = -3: " + str(inSampleError))
print("Out Sample Error of k = -3: " + str(outSampleError) + "\n")

print("In Sample Error of k = 3: " + str(inSampleError2))
print("Out Sample Error of k = 3: " + str(outSampleError2)+ "\n")

print("In Sample Error of k = -2: " + str(inSampleError3))
print("Out Sample Error of k = -2: " + str(outSampleError3)+ "\n")

print("In Sample Error of k = -1: " + str(inSampleError4))
print("Out Sample Error of k = -1: " + str(outSampleError4)+ "\n")

print("In Sample Error of k = 0: " + str(inSampleError5))
print("Out Sample Error of k = 0: " + str(outSampleError5)+ "\n")

print("In Sample Error of k = 1: " + str(inSampleError6))
print("Out Sample Error of k = 1: " + str(outSampleError6)+ "\n")

print("In Sample Error of k = 2: " + str(inSampleError7))
print("Out Sample Error of k = 2: " + str(outSampleError7)+ "\n")


