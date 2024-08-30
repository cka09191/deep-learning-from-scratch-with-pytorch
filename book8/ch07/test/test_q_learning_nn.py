import numpy as np
import unittest


if (__package__ is None) or (__package__ == 'test'):
        import sys
        from os import path
        print(path.dirname(path.dirname(path.abspath(__file__))))
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from src.q_learning_nn import one_hot




class q_learning_nn_test(unittest.TestCase):
    
    def test_one_hot(self):
        print(one_hot((1,2)))
        print(type(one_hot((1,2))))
        self.assertTrue(np.array_equal(one_hot((1,2)), np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])))

