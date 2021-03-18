import unittest
from mathematics.matrix import get_matrix_average


class MatrixTests(unittest.TestCase):

    def test_get_matrix_average(self):
        matrix = [[1,1,1], [2,1,1], [3,1,1],[4,1,1],[5,1,1]]
        expected = [3,1,1]
        result = get_matrix_average(5,3,matrix)
        self.assertEqual(expected, result)

if __name__ == '__main__':
    unittest.main()
