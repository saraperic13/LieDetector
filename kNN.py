import csv
import math
import operator
import random


def load_dataset(filename, training=False, trainingSet=[], testSet=[]):
    with open(filename, 'rt') as csvfile:
        next(csv.reader(csvfile))
        lines = csv.reader(csvfile)
        dataset = list(lines)

        if training:
            split = 0.7 * len(dataset)
            # random.shuffle(dataset)
        else:
            split = len(dataset)

        for x in range(len(dataset) - 1):
            for y in range(5):
                dataset[x][y] = float(dataset[x][y])
            if len(trainingSet) <= split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


"""
 Calculates Euclidean distance between two instances.
"""
def euclidean_distance(instance1, instance2, number_of_params):
    distance = 0
    for param in range(number_of_params):
        distance += pow((instance1[param] - instance2[param]), 2)
    return math.sqrt(distance)


"""
 Finds k nearest neighbours of an instance from training dataset.
"""
def get_neighbors(trainingSet, instance, k):
    distances = []
    length = len(instance) - 1
    for x in range(len(trainingSet)):
        dist = euclidean_distance(instance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


"""
 Return predicted class by getting the majority voted response from a number of neighbors,
 by allowing each neighbor to vote for their class attribute.
"""
def calculate_votes(neighbors):
    class_votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def calculate_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def predict(to_predict, data_set_path, k=3, training=False):
    training_set = []
    test_set = []

    predictions = []

    load_dataset(data_set_path, training, training_set, test_set)

    if training:
        for x in range(len(test_set)):
            neighbors = get_neighbors(training_set, test_set[x], k)
            result = calculate_votes(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result) + ', actual=' + repr(test_set[x][-1]))

        accuracy = calculate_accuracy(test_set, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
    else:
        for x in range(len(to_predict)):
            neighbors = get_neighbors(training_set, to_predict[x], k)
            result = calculate_votes(neighbors)
            predictions.append(result)
            print('> predicted=' + repr(result))

    return predictions


# predict([[0.6333, 0.8333, 0, 0, 0]], "../files/datasetExtracted.csv", k=3)
