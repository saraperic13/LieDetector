import csv
import math
import operator
from random import shuffle

training_set = []
test_set = []

def cross_validation():
    for k in range(1, len(training_set) - 1):
        acc = 0
        for i in range(len(training_set)):
            to_predict = training_set[i]
            training_set.remove(to_predict)
            predictions = fit([to_predict], training_set, k)
            acc += calculate_accuracy([to_predict], predictions)
            training_set.insert(i, to_predict)

        print("k = %2d" % (k), " Tačnost: ", acc / len(training_set))


def load_dataset(filename, training=False):
    with open(filename, 'rt') as csvfile:
        next(csv.reader(csvfile))
        lines = csv.reader(csvfile)
        dataset = list(lines)

        shuffle(dataset)

        if training:
            split = 0.8 * len(dataset)
        else:
            split = len(dataset)

        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if len(training_set) <= split:
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])


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


def fit(to_predict, dataset, k):
    predictions = []
    for x in range(len(to_predict)):
        neighbors = get_neighbors(dataset, to_predict[x], k)
        result = calculate_votes(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result))

    return predictions


def predict(to_predict, data_set_path, k=12, training=False):
    if len(training_set) == 0:
        load_dataset(data_set_path, training)

    # cross_validation()

    if training:
        predictions = fit(test_set, training_set, k)
        accuracy = calculate_accuracy(test_set, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')
    else:
        predictions = fit(to_predict, training_set, k)

    return predictions


# predict([[0.6333, 0.4333, 1, 0]], "files/datasetExtracted.csv", k=12, training=False)
