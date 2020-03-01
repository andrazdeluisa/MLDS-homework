import numpy as np


class Tree:

    def __init__(self, rand, get_candidate_columns, min_samples):
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples

    def build(self, X, y):
        return


class Bagging:

    def __init__(self, rand, tree_builder, n):
        self.rand = rand
        self.tree_builder = tree_builder
        self.n = n

    def build(self, X, y):
        return


class RandomForest:

    def __init__(self, rand, n, min_samples):
        self.rand = rand
        self.n = n
        self.min_samples = min_samples

    def build(self, X, y):
        return


class Model:

    def __init__(self, method):
        self.method = method

    def predict(self, X):
        return


def hw_tree_full():
    return


def hw_cv_min_samples():
    return


def hw_bagging():
    return


def hw_randomforest():
    return
