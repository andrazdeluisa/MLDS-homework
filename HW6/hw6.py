import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
import matplotlib.pyplot as plt
import sys
from time import time


sys.path.append('HW1')
sys.path.append('HW5')


from hw1 import logreg_mle, polynomial_expansion
from hw5 import SVR, RBF


class ANNClassification:

    def __init__(self, units, lambda_, check_grad=True):
        self.units = units
        self.lambda_ = lambda_
        self.last_act = softmax
        self.loss = log_loss
        self.check_grad = check_grad

    def fit(self, X, y):
        m = X.shape[1]
        layer_size = [m] + self.units + [len(np.unique(y))]
        return fit_class_regr(X, y, layer_size, self.check_grad, self.last_act, self.loss, self.lambda_)


class ANNRegression:

    def __init__(self, units, lambda_, check_grad=True):
        self.units = units
        self.lambda_ = lambda_
        self.last_act = lambda x: x
        self.loss = mse
        self.check_grad = check_grad

    def fit(self, X, y):
        m = X.shape[1]
        layer_size = [m] + self.units + [1]
        return fit_class_regr(X, y, layer_size, self.check_grad, self.last_act, self.loss, self.lambda_)


class ANNModel:

    def __init__(self, weights_, layer_size, last_act, loss):
        self.weights_ = weights_
        self.last_act = last_act
        self.pred = lambda x, y: x
        self.layer_size = layer_size
        self.loss = loss

    def predict(self, X):
        res = feed_forw(self.weights_, X, None, self.layer_size, self.last_act, self.pred)
        if res.shape[1] == 1:
            res = res.flatten()
        return res

    def weights(self):
        return to_grid(self.weights_, self.layer_size)


def fit_class_regr(X, y, layer_size, check_grad, last_act, loss, lambda_):
    weights = [np.random.normal(loc=0, scale=0.5, size=(layer_size[i] + 1, layer_size[i + 1])) for i in range(len(layer_size) - 1)]
    weights = to_flat(weights)
    if check_grad:
        correct_grad = check_gradient(X, y, layer_size, last_act, loss, lambda_, len(weights))
        if correct_grad:
            print('Gradient is computed correctly: numerically verified')
        else:
            print('WARNING: numerical verification of gradient failed')
    weights_opt, _, _ = fmin_l_bfgs_b(feed_forw, weights, fprime=back_prop, args=(X, y, layer_size, last_act, loss, lambda_), maxfun=1e6, maxiter=1e3, factr=1e9)
    return ANNModel(weights_=weights_opt, layer_size=layer_size, last_act=last_act, loss=loss)


def feed_forw(weights, X, y, layer_size, last_act, loss, lambda_=0, info=False):
    weights = to_grid(weights, layer_size)
    a = [np.array([]) for _ in range(len(weights))]
    a[0] = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    for i in range(1, len(weights)):
        a[i] = inv_logit(a[i - 1].dot(weights[i - 1]))
        a[i] = np.append(a[i], np.ones((a[i].shape[0], 1)), axis=1)
    a.append(last_act(a[-1].dot(weights[-1])))
    if info:
        return a
    else:
        reg = lambda_ / 2 * np.sum(list(map(np.sum, list(map(lambda x: x[:-1, ] ** 2, weights)))))
        loss_ = loss(a[-1], y)
        return loss_ + reg


def back_prop(weights, X, y, layer_size, last_act, loss, lambda_=0):
    a = feed_forw(weights, X, y, layer_size, last_act, loss, lambda_=0, info=True)    
    weights = to_grid(weights, layer_size)
    # don't regularize bias
    grad_reg = [lambda_ * np.append(w[:-1, ], np.zeros((1, w.shape[1])), axis=0) for w in weights]
    # gradient of cost function: mse for regression
    grad = [np.zeros(w.shape) for w in weights]
    delta = [[] for _ in weights]
    if loss.__name__ == 'mse':
        delta[-1] = 2 * (a[-1] - np.reshape(y, a[-1].shape)) / len(y)
        grad[-1] = a[-2].T.dot(delta[-1])
    elif loss.__name__ == 'log_loss':
        delta[-1] = a[-1]
        delta[-1][range(len(y)), y] -= 1
        delta[-1] /= len(y)
        grad[-1] = a[-2].T.dot(delta[-1])
    for i in range(1, len(weights)):
        act_i = np.delete(a[-1 - i] * (1-a[-1 - i]), -1, axis=1)
        weights_i = np.delete(weights[-i], -1, axis=0).T
        delta[-1 - i] = act_i * delta[-i].dot(weights_i)
        grad[-1 - i] = a[-2 - i].T.dot(delta[-1 - i])
    return to_flat(grad) + to_flat(grad_reg)


def to_flat(weights):
    res = []
    for w in weights:
        res += list(w.flatten())
    return np.array(res)


def to_grid(weights, layer_size):
    res = []
    for i in range(len(layer_size) - 1):
        m, n = layer_size[i] + 1, layer_size[i + 1]
        tmp = np.reshape(weights[:int(m * n)], (m, n))
        weights = weights[m*n:]
        res += [tmp]
    return res


def inv_logit(x):
    pos = x > 0
    res = np.empty(x.shape, dtype=np.float)
    res[pos] = 1. / (1 + np.exp(-x[pos]))
    exp_t = np.exp(x[~pos])
    res[~pos] = exp_t / (1. + exp_t)
    return res


def log_loss(pred, y):
    loss = pred[range(pred.shape[0]), y]
    eps = 1e-20
    idx = loss < eps
    res = np.log(eps) * np.ones((len(loss)))
    res[~idx] = np.log(loss[~idx])
    return -np.mean(res)


def mse(pred, y):
    return np.mean((np.reshape(pred, (pred.shape[0],)) - y) ** 2)


def softmax(x):
    norm = np.sum(np.exp(x), axis=1, keepdims=True)
    return np.exp(x) / norm


def check_gradient(X, y, layer_size, last_act, loss, lambda_, n):
    def f(w): return feed_forw(w, X, y, layer_size, last_act, loss, lambda_)
    def df(w): return back_prop(w, X, y, layer_size, last_act, loss, lambda_)
    return numerical_gradient_test(f, df, n)


def numerical_gradient_test(f, df, n):
    eps = 1e-6
    tol = 1e-3
    for _ in range(10):
        x0 = np.random.normal(loc=0, scale=0.5, size=n)
        grad = df(x0)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1
            f_plus = f(x0 + eps*e_i)
            f_minus = f(x0 - eps*e_i)
            approx = (f_plus - f_minus) / 2 / eps
            if abs(approx - grad[i]) > tol:
                return False
    return True


def logistic_regression(X, y, test):
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    test = np.append(test, np.ones((test.shape[0], 1)), axis=1)
    beta = logreg_mle(X, y)
    tmp = np.reshape(inv_logit(test.dot(beta)), (test.shape[0], 1))
    pred = np.append(1 - tmp, tmp, axis=1)
    return pred


def support_vector_regression(X, y, test):
    # use rbf, sigma=1, constant lambda
    # already analyzed in hw5, not needed to use CV another time
    K = RBF(sigma=1)
    svr = SVR(kernel=K, lambda_=1, epsilon=2)
    model = svr.fit(X, y)
    pred = model.predict(test)
    return pred


def standardize(train, test):
    train_std = (train - np.mean(train, axis=0)) / np.std(train, axis=0)
    test_std = (test - np.mean(train, axis=0)) / np.std(train, axis=0)
    return train_std, test_std


def cv(train, train_l, units, lambdas, type_='class', k=5):
    idxs = np.tile(np.asarray(list(range(k))), (len(train_l) // k) + 1)
    idxs = idxs[:len(train_l)]
    errs = np.zeros((len(units), len(lambdas)))
    for i in range(len(units)):
        unit = units[i]
        for m in range(len(lambdas)):
            lambda_ = lambdas[m]
            if type_ == 'class':
                ann = ANNClassification(units=unit, lambda_=lambda_, check_grad=False)
            else:
                ann = ANNRegression(units=unit, lambda_=lambda_, check_grad=False)
            mses = []
            for j in range(k):
                train_cv = train[idxs != j, :]
                test_cv = train[idxs == j, :]
                train_y = train_l[idxs != j]
                test_y = train_l[idxs == j]
                model = ann.fit(train_cv, train_y)
                pred = model.predict(test_cv)
                if type_ == 'class':
                    mses.append(log_loss(pred, test_y))
                else:
                    mses.append(mse(pred, test_y))
            errs[i, m] = np.mean(mses)
    tmp = np.argmin(errs)
    best_unit, best_lambda = np.unravel_index(tmp, errs.shape)
    est_err = errs[best_unit, best_lambda]
    return units[best_unit], lambdas[best_lambda], est_err


def prepare_housing(filename, type_='regr', sep=','):
    data = pd.read_csv(filename, sep=sep)
    label = np.asarray(data.iloc[:, -1])
    if type_ == 'bin_class':
        label = np.asarray(label == "C1", dtype=int)
    data_np = np.asarray(data.iloc[:, :-1], dtype=np.float64)
    nrows = np.size(label)
    ntrain = int(nrows*0.8)
    train_l = label[:ntrain]
    test_l = label[ntrain:]
    train = data_np[:ntrain, :]
    test = data_np[ntrain:, :]
    train, test = standardize(train, test)
    return train, train_l, test, test_l


def compare2():
    train, train_l, test, test_l = prepare_housing('HW4/housing2r.csv', type_='regr')
    pred_svr = support_vector_regression(train, train_l, test)
    unit, lambda_, _ = cv(train, train_l, units=[[], [2], [5], [10], [5, 5]], lambdas=[0.0001, 0.001, 0.1, 1], type_='regr')
    ann = ANNRegression(units=unit, lambda_=lambda_)
    model_ann = ann.fit(train, train_l)
    pred_ann = model_ann.predict(test)
    loss_svr = mse(pred_svr, test_l)
    loss_ann = mse(pred_ann, test_l)
    return loss_svr, loss_ann


def compare3():
    train, train_l, test, test_l = prepare_housing('HW2/housing3.csv', type_='bin_class')
    pred_logreg = logistic_regression(train, train_l, test)
    unit, lambda_, _ = cv(train, train_l, units=[[], [2], [5], [10], [5, 5]], lambdas=[0.0001, 0.001, 0.1, 1], type_='class')
    ann = ANNClassification(units=unit, lambda_=lambda_)
    model_ann = ann.fit(train, train_l)
    pred_ann = model_ann.predict(test)
    loss_logreg = log_loss(pred_logreg, test_l)
    loss_ann = log_loss(pred_ann, test_l)
    return loss_logreg, loss_ann


def load_final_data():
    train_data = pd.read_csv('HW6/train.csv', sep=',')
    test_data = pd.read_csv('HW6/test.csv', sep=',')
    train_l = np.asarray(train_data.iloc[:, -1])
    train_l = np.array(list(map(lambda x: int(x.split('_')[-1]) - 1, train_l)))
    train = np.asarray(train_data.iloc[:, 1:-1])
    test = np.asarray(test_data.iloc[:, 1:])
    train, test = standardize(train, test)
    return train, train_l, test


def create_final_predictions():
    start = time()
    train, train_l, test = load_final_data()
    loading = time()
    print('loading ' + str(loading - start))
    unit, lambda_, est_err = cv(train, train_l, units=[[5], [10], [20], [10, 5]], lambdas=[0.0001, 0.001, 0.1, 1], type_='class')
    cv_time = time()
    print('cv ' + str(cv_time - loading))
    ann = ANNClassification(units=unit, lambda_=lambda_, check_grad=False)
    model = ann.fit(train, train_l)
    pred = model.predict(test)
    fitting = time()
    print('fitting ' + str(fitting - cv_time))
    pred_train = model.predict(train)
    with open('HW6/final.txt', 'w+') as file:
        file.write(','.join(['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']) + '\n')
        for i in range(pred.shape[0]):
            file.write(str(i + 1) + ',' + ','.join(list(map(str, pred[i, :]))) + '\n')
        file.close()
    return log_loss(pred_train, train_l), est_err


if __name__ == '__main__':
    np.random.seed(1)
    print('Classification (log reg, ann):')
    print(compare3())
    print('Regression (svr, ann):')
    print(compare2())
    print('Final predictions:')
    print(create_final_predictions())
