import math
import operator


def KNN(Xtrainset, Ytrainset, Ytestset, testset, k):
    prediction = []

    for x in range(len(testset)):
        neighbours = getNeighbors(Xtrainset, Ytrainset, testset[x], k)
        result = getResponse(neighbours)
        prediction.append(result)
        print('> predicted=' + repr(result) + ',actual=' + repr(Ytestset[x]))

    accuracy = getAccuracy(testset, Ytestset, prediction)
    return accuracy


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, trainresult, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], trainresult[x], dist))
    distances.sort(key=operator.itemgetter(2))
    neighbors = []
    for x in range(k):
        neighbors.append((distances[x][0], distances[x][1]))
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, testResult, predictions):
    print(testResult)
    print(predictions)
    correct = 0
    for x in range(len(testSet)):
        if testResult[x] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
