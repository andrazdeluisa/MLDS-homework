import numpy as np
import pandas as pd
import time
import random


class Tree:

    def __init__(self, rand, get_candidate_columns, min_samples):
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples

    def build(self, X, y):
        if len(y) < self.min_samples:
            # split has less than min samples, create leaf
            if len(y) == 0:
                node = None
            else:
                node = [(sum(y) / len(y)) > 0.5]
            left, right = None, None
        elif (sum(y) == len(y)) or sum(y) == 0:
            # split is homogeneous, create leaf
            node = [sum(y) / len(y)]
            left, right = None, None
        else:
            # create tree with node and 2 children
            lowest_gini = 2
            col, treshold = 0, None
            y1, y2 = [], []
            split1, split2 = [], []
            # which columns should be searched for the best split
            idxs = self.get_candidate_columns(X, self.rand)
            for idx in idxs:
                # find best split for each column
                gini_tmp, treshold_tmp, split_tmp, y1_tmp, y2_tmp = split(X[:, idx], y)
                # compare with previous best and update if necessary
                if gini_tmp <= lowest_gini:
                    lowest_gini = gini_tmp
                    col, treshold = idx, treshold_tmp
                    y1, y2 = y1_tmp, y2_tmp
                    split1, split2 = X[split_tmp[0], :], X[split_tmp[1], :]
            # set node and build subtrees
            if len(y2) == 0:
                return TreeModel([round(np.mean(y1))], None, None)
            node = [col, treshold]
            right = self.build(split2, y2)
            left = self.build(split1, y1)
        return TreeModel(node, left, right)


class TreeModel:

    def __init__(self, node, left, right):
        self.node = node
        self.left = left
        self.right = right

    def predict_row(self, row):
        if len(self.node) == 1:
            return int(self.node[0])
        else:
            idx = self.node[0]
            treshold = self.node[1]
            if row[idx] <= treshold:
                return self.left.predict_row(row)
            else:
                return self.right.predict_row(row)
        return

    def predict(self, X):
        y = []
        for row in X:
            y.append(self.predict_row(row))
        return np.asarray(y)


class Bagging:

    def __init__(self, rand, tree_builder, n):
        self.rand = rand
        self.tree_builder = tree_builder
        self.n = n

    def build(self, X, y):
        trees = []
        for _ in range(self.n):
            idxs = np.asarray(self.rand.choices(population=list(range(len(y))), k=len(y)))
            X_b = X[idxs, :]
            y_b = y[idxs]
            trees.append(self.tree_builder.build(X_b, y_b))
        return MultiTreeModel(trees)


class MultiTreeModel:

    def __init__(self, trees):
        self.trees = trees

    def predict(self, X):
        preds = np.empty((np.shape(X)[0], 1))
        pred_class = []
        for i in range(len(self.trees)):
            tree = self.trees[i]
            preds = np.insert(preds, i, np.asarray(tree.predict(X)), axis=1)
        preds = preds[:, :-1]
        for row in preds:
            pred_class.append(int(round(np.mean(row))))
        return np.asarray(pred_class)


class RandomForest:

    def __init__(self, rand, n, min_samples):
        self.rand = rand
        self.n = n
        self.min_samples = min_samples

    def build(self, X, y):
        t = Tree(rand=self.rand, get_candidate_columns=d_features, min_samples=self.min_samples)
        b = Bagging(rand=self.rand, tree_builder=t, n=self.n)
        return b.build(X, y)


def hw_tree_full(train, test):
    t = Tree(rand=1, get_candidate_columns=all_features, min_samples=2)
    a = time.time()
    p = t.build(train[0], train[1])
    print('Tree building time: {}'.format(time.time() - a))
    pred_train = p.predict(train[0])
    pred_test = p.predict(test[0])
    mis_train = mis_rate(pred_train, train[1])
    mis_test = mis_rate(pred_test, test[1])
    return mis_train, mis_test


def hw_cv_min_samples(train, test):
    X, y = train
    best_min = 2
    best_mis = 1
    idxs = np.tile(np.asarray([0, 1, 2, 3, 4]), (len(y) // 5) + 1)
    idxs = idxs[:len(y)]
    a = time.time()
    for min_sample in range(2, 50):
        mis_rates = []
        for i in range(max(idxs + 1)):
            train_cv = X[idxs != i, :]
            test_cv = X[idxs == i, :]
            train_y = y[idxs != i]
            test_y = y[idxs == i]
            t_cv = Tree(rand=1, get_candidate_columns=all_features, min_samples=min_sample)
            p_cv = t_cv.build(train_cv, train_y)
            pred_test_cv = p_cv.predict(test_cv)
            mis_rates.append(mis_rate(test_y, pred_test_cv))
        if np.mean(mis_rates) < best_mis:
            best_mis = np.mean(mis_rates)
            best_min = min_sample
    print('Cross validation building time: {}'.format(time.time() - a))
    t = Tree(rand=1, get_candidate_columns=all_features, min_samples=best_min)
    p = t.build(X, y)
    pred_train = p.predict(X)
    pred_test = p.predict(test[0])
    mis_train = mis_rate(pred_train, y)
    mis_test = mis_rate(pred_test, test[1])
    return mis_train, mis_test, best_min


def hw_bagging(train, test):
    t = Tree(rand=1, get_candidate_columns=all_features, min_samples=2)
    b = Bagging(rand=random.Random(1), tree_builder=t, n=50)
    a = time.time()
    p = b.build(train[0], train[1])
    print('Bagging building time: {}'.format(time.time() - a))
    pred_train = p.predict(train[0])
    pred_test = p.predict(test[0])
    mis_train = mis_rate(pred_train, train[1])
    mis_test = mis_rate(pred_test, test[1])
    return mis_train, mis_test


def hw_randomforests(train, test):
    # for n=20 the results are much better, both for the training and test
    rf = RandomForest(rand=random.Random(1), n=50, min_samples=2)
    a = time.time()
    p = rf.build(train[0], train[1])
    print('Random forest building time: {}'.format(time.time() - a))
    pred_train = p.predict(train[0])
    pred_test = p.predict(test[0])
    mis_train = mis_rate(pred_train, train[1])
    mis_test = mis_rate(pred_test, test[1])
    return mis_train, mis_test


def prepare_data(filename):
    data = pd.read_csv(filename)
    label = np.asarray(data.iloc[:, -1])
    # set C1 as predicted class
    label = np.asarray(label == "C1", dtype=int)
    data_np = np.asarray(data.iloc[:, :-2])
    nrows = np.size(label)
    ntrain = int(nrows*0.8)
    train_l = label[:ntrain]
    test_l = label[ntrain:]
    train = data_np[:ntrain, :]
    test = data_np[ntrain:, :]
    return train, train_l, test, test_l


def mis_rate(preds, labels):
    r = sum(preds != labels)
    r /= np.size(preds)
    return r


def gini_idx(split1, split2):
    n1, n2 = len(split1), len(split2)
    gini1 = 1 - (sum(split1)/n1) ** 2 - (1 - sum(split1)/n1) ** 2
    gini2 = 1 - (sum(split2)/n2) ** 2 - (1 - sum(split2)/n2) ** 2
    return (gini1 * n1 + gini2 * n2) / (n1 + n2)


def all_features(X, rand):
    try:
        m = np.shape(X)[1]
    except:
        m = 1
    return list(range(m))


def d_features(X, rand):
    try:
        m = np.shape(X)[1]
    except:
        m = 1
    return rand.sample(population=list(range(m)), k=int(np.ceil(np.sqrt(m))))


def split(X, y):
    gini = 2
    treshold = None
    split = ([], [])
    y1, y2 = [], []
    sort = np.argsort(X)
    X = X[sort]
    y = y[sort]
    X_u = np.unique(X)
    # some corrections after received grade
    if len(X_u) == 1:
        return 2, X_u[0], (sort, []), y, y2
    for i in range(len(X_u) - 1):
        x = X_u[i]
        e = np.where(X == x)[-1][0]
        if e == len(y) - 1:
            continue
        y1_tmp = y[:e + 1]
        y2_tmp = y[e + 1:]
        gini_tmp = gini_idx(y1_tmp, y2_tmp)
        if gini_tmp <= gini:
            gini = gini_tmp
            treshold = x
            split = (sort[:e + 1], sort[e + 1:])
            y1 = y1_tmp
            y2 = y2_tmp
    return gini, treshold, split, y1, y2


if __name__ == '__main__':
    train, train_l, test, test_l = prepare_data('HW2/housing3.csv')
    print(hw_tree_full((train, train_l), (test, test_l)))
    print(hw_bagging((train, train_l), (test, test_l)))
    print(hw_cv_min_samples((train, train_l), (test, test_l)))
    print(hw_randomforests((train, train_l), (test, test_l)))
