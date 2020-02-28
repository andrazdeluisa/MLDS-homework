import unittest

import numpy as np

from hw1 import logreg_log_likelihood, logreg_mle, logreg_predict, polynomial_expansion


class HW1Tests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                    
                      [1, 1]])
        self.y = np.array([0, 1, 1, 1])

    def test_logreg_log_likelihood(self):
        ll = logreg_log_likelihood(np.array([0.5, 0.1]), self.X, self.y)
        self.assertIsInstance(ll, float)
        self.assertLess(ll, 0)

    def test_logreg_mle(self):
        mle = logreg_mle(self.X, self.y)
        self.assertIsInstance(mle, np.ndarray)
        self.assertEqual(mle.shape, (2,))

    def test_logreg_predict(self):
        pred = logreg_predict(np.array([10, 10]), self.X)
        np.testing.assert_equal(pred, self.y)

    def test_polynomial_expansion(self):
        """ Enforces lexical ordering per expansion degree."""
        X = np.array([[2, 3],
                      [5, 7]])
        r = polynomial_expansion(X, 3)
        np.testing.assert_equal(r, [[2, 3, 4, 6, 9, 8, 12, 18, 27],
                                    [5, 7, 25, 35, 49, 125, 175, 245, 343]])


if __name__ == "__main__":
    unittest.main()
