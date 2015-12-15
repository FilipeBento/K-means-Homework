
import pprint
import random
import copy

import itertools
from collections import Counter

import numpy as np
import genes


def parse_file_data_size(filename):
    with open(filename, 'r') as f:
        numrows = f.readline()
        numcols = f.readline()
        lines = f.readlines()
        matrix = []
        for line in lines:
            values = line.split(' ')
            lst = []
            for value in values[:-1]:
                lst.append(float(value))
            matrix.append(lst)
    return matrix


def print_precision(labels, partition_number):
    label, occurrences = max(Counter(labels).iteritems(), key=lambda x: x[1])
    print 'Partition ', partition_number, "is of type '" + label + "' with", occurrences, 'occurrences.'
    print 'Precision:', float(occurrences) / len(labels)


def main():
    matrix = copy.deepcopy(genes.get_patients())
    genes.remove_last_column(matrix)
    k = 2
    partitions, centroids = k_means(matrix, k)
    assert len(partitions) == k
    for i, partition in enumerate(partitions):
        print "PARTITION ", i
        pprint.pprint(len(partitions[i]))
        if not partitions[0]:
            print 'labels 1: empty partition'
        else:
            matrix_with_labels = genes.get_patients()
            labels = get_labels_per_partition(matrix_with_labels, partitions[i])
            print 'labels 1 =', labels
            partition_number = i
            print_precision(labels, partition_number)
        print ''


def get_labels_per_partition(matrix, partition):
    points_and_labels = {}
    labels = []
    for line in matrix:
        label = line[len(line) - 1]
        points_and_labels[tuple(line[:-1])] = label

    points = points_and_labels.keys()
    for point in partition:
        point = tuple(point)
        if point in points:
            labels.append(points_and_labels[point])
    return labels


def print_padded_list(lst, pad):
    print '[',
    for l in lst:
        print '%*s' % (pad, l),
    print ']'


def k_means(matrix, k):
    centroids = random_centroids(k, matrix)
    prev_partitions = []
    partitions = None
    while prev_partitions != partitions:
        prev_partitions = copy.deepcopy(partitions)
        partitions = [[] for _ in xrange(k)]
        for column in matrix:
            partition_index = get_closer_centroid_index(column, centroids)
            partitions[partition_index].append(column)
        centroids = update_centroids(partitions, matrix)
        assert len(centroids) == len(partitions) == k
    return partitions, centroids


def get_closer_centroid_index(column, centroids):
    min_distance = 9999999999999999999999L
    index = None
    for i, centroid in enumerate(centroids):
        distance = euclidean_distance(column, centroid)
        if distance < min_distance:
            min_distance = distance
            index = i
    return index


def euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def random_centroids(k, matrix):
    centroids = []
    for _ in xrange(k):
        min_values = []
        max_values = []
        for line in np.array(matrix).transpose():
            min_values.append(np.amin(line))
            max_values.append(np.amax(line))
        assert len(min_values) == len(max_values) == len(matrix[0])
        centroid = [random.uniform(min_values[j], max_values[j]) for j in xrange(len(matrix[0]))]
        centroids.append(centroid)
    return centroids


def update_centroids(partitions, matrix):
    centroids = []
    points = copy.deepcopy(matrix)
    # Assign random point to empty partitions
    for i, partition in enumerate(partitions):
        if len(partition) == 0:
            partition.append(points[i])

    # Calculate centroid (mean) for each partition
    for partition in partitions:
        dimensions_per_point = len(partition[0])
        centroid = [0] * dimensions_per_point
        for point in partition:
            assert len(centroid) == len(point)
            for i in xrange(len(point)):
                centroid[i] += point[i]
        centroid = map(lambda k: k / float(len(partition)), centroid)
        centroids.append(centroid)
    return centroids


if __name__ == '__main__':
    main()
