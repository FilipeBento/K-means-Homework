from __future__ import division

import genes
import lab6


def mean(a):
    return sum(a) / len(a)


def mean_of_lists(lists):
    return map(mean, zip(*lists))


def test_update_centroids():
    # Initialization
    matrix = [
        [1, 2, 3],
        [4.2, 5, 6],
        [7.4, 8, 9],
        [5, 6, 7],
        [8, 9.5, 10],
        [11, 12, 13.2]
    ]
    partitions = [
        [[1, 2, 3], [4.2, 5, 6], [7.4, 8, 9]],
        [[5, 6, 7], [8, 9.5, 10], [11, 12, 13.2]],
    ]

    # Test
    centroids = lab6.update_centroids(partitions, matrix)

    # Assert result
    for i, partition in enumerate(partitions):
        assert centroids[i] == mean_of_lists(partition)


def test_mean_col_val():
    # Initialization
    matrix = [
        [1, 2, 3, 4.2, 5, 6, 7.4, 8, 9],
        [5, 6, 7, 8, 9.5, 10, 11, 12, 13.2],
        [-0.3, 34, 7.2, 14.7, 9, 12.1, 1, 3, 5.8],
    ]

    # Test
    for i in xrange(len(matrix[0])):
        mean_val = genes.mean_column_val(matrix, i)
        assert mean_val == mean(zip(*matrix)[i])


test_update_centroids()
test_mean_col_val()