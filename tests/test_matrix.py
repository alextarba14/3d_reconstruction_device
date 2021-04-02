import unittest
from unittest import TestCase
import numpy as np

from mathematics.matrix import get_matrix_average, get_indexes_of_valid_points


class Test(TestCase):
    def test_get_indexes_of_valid_points(self):
        array = np.zeros((10, 3))
        indexes = get_indexes_of_valid_points(array)
        # expecting an empty array of tuples since there are only zeros in the array
        self.assertTrue(not all(indexes))

        array[1] = [-1, -2, 3]
        array[3] = [-1, 2, 8]
        array[5] = [1, -2, 5]
        array[6] = [-1, 22, -3]
        indexes = get_indexes_of_valid_points(array)
        # expecting indexes 1, 3, 5 but not 6 since it has negative value
        self.assertTrue(np.array_equal([1, 3, 5], indexes[0]))

    def test_get_matrix_average(self):
        matrix = [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1]]
        expected = [3, 1, 1]
        result = get_matrix_average(matrix)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
