
import pprint
import random
import copy
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


def main():
    # matrix = parse_file_data_size('test.txt')
    matrix = genes.get_patients()
    partitions, centroids = k_means(matrix, k=2)
    assert len(partitions) == 2

    print "partition 1"
    pprint.pprint(len(partitions[0]))
    print "partition 2"
    pprint.pprint(len(partitions[1]))
    print '%13s' % 'genes =',
    print_padded_list(genes.get_genes(), 18)
    print
    print '%13s' % 'centroid 1 =',
    print_padded_list(centroids[0], 18)
    print
    print '%13s' % 'centroid 2 =',
    print_padded_list(centroids[1], 18)


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
    for partition in partitions:
        if len(partition) == 0:
            random_point = random.choice(points)
            partition.append(random_point)
            points.remove(random_point)
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
