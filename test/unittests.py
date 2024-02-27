"""
Module Description: Unit tests for the trainer module.

This module defines unit tests for the trainer module, including tests for both training
and evaluation functionalities. It contains test cases to ensure that the train and
evaluation functions behave as expected when provided with the appropriate configurations.

The module defines two test classes:
- TestTrainerTrain: Test case for training the model.
- TestTrainerEval: Test case for evaluating the model.

These test classes use the create_train_test() and create_eval_test() functions to set up
the test environment and execute the tests accordingly.

Usage:
    Run this module as a standalone script to execute the unit tests.
"""

import os
import sys
import unittest
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.trainer import train


def create_train_test():
    """
    Create a test case for training the model.
    """
    class TrainerTest(unittest.TestCase):
        """
        Test case for training the model.
        """

        @classmethod
        def setUpClass(cls):
            """
            Set up class for the test case.
            """
            cls.config = json.load(open('configs/test_train.json'))
            
        def test_train(self):
            """
            Test the train function.
            """
            train(self.config)

    return TrainerTest


def create_eval_test():
    """
    Create a test case for evaluating the model.
    """
    class TrainerTest(unittest.TestCase):
        """
        Test case for evaluating the model.
        """
        
        @classmethod
        def setUpClass(cls):
            """
            Set up class for the test case.
            """
            cls.config = json.load(open('configs/test_eval.json'))
            
        def test_eval(self):
            """
            Test the evaluation function.
            """
            train(self.config)

    return TrainerTest

class TestTrainer_train(create_train_test()):
    """
    Test case for training the model.
    """

class TestTrainer_eval(create_eval_test()):
    """
    Test case for evaluating the model.
    """

if __name__ == '__main__':
    unittest.main()
