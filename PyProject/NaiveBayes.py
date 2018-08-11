import math

def separateByClass(X_Train,Y_Train): # to seperate train set based on class/labels using maps
    separated = {}
    for i in range(len(X_Train)):
        vector = X_Train[i]
        if (Y_Train[i] not in separated):
            separated[Y_Train[i]] = []
        separated[Y_Train[i]].append(vector)
    return separated


def mean(numbers):
    xi=sum(numbers)
    total=float(len(numbers))
    mew=xi /total
    return mew


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    return summaries


def summarizeByClass(X_Train,Y_Train):
    separated = separateByClass(X_Train,Y_Train)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def fit(X_Train,Y_Train):     # for preparing the training model
    summaries=summarizeByClass(X_Train,Y_Train)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def getAccuracy(Y_Test, predictions):
    correct = 0
    for i in range(len(Y_Test)):
        if Y_Test[i] == predictions[i]:
            correct += 1
    return (correct / float(len(Y_Test))) * 100.0