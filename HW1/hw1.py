from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from itertools import combinations_with_replacement
import pandas as pd
import matplotlib.pyplot as plt


def logreg_log_likelihood(beta, X, y):
    tmp = X @ np.transpose(beta)
    res = -(1-y) @ tmp
    idx = tmp < -50
    res += tmp[idx].sum()
    res -= np.log(1 + np.exp(-tmp[~idx])).sum()
    return res


def inv_logit(x):
    # added after response from assistant
    pos = x > 0
    res = np.empty(x.size, dtype=np.float)
    res[pos] = 1. / (1 + np.exp(-x[pos]))
    exp_t = np.exp(x[~pos])
    res[~pos] = exp_t / (1. + exp_t)
    return res


def grad(beta, X, y):
    # corrected after response from assistant
    tmp = y - inv_logit(X @ beta)
    res = - tmp @ X
    return res / np.linalg.norm(res)


def logreg_mle(X, y):

    def ll_tmp(beta):
        return - logreg_log_likelihood(beta, X, y)

    def grad_tmp(beta):
        return grad(beta, X, y)

    m = np.size(X, axis=1)
    beta0 = np.ones(m) / 2
    # without gradient of log_likelihood
    #res, _, info = fmin_l_bfgs_b(ll_tmp, beta0, approx_grad=True, maxfun=1e6, maxiter=1e6)
    # with gradient of log-likelihood
    res, _, info = fmin_l_bfgs_b(ll_tmp, beta0, fprime=grad_tmp, maxfun=1e6, maxiter=1e6)
    ncalls = info['funcalls']
    return res


def logreg_predict(beta, X):
    tmp = inv_logit(X @ beta)
    return tmp > 0.5


def polynomial_expansion(X, n):
    combs = []
    for i in range(2, n+1):
        combs += combinations_with_replacement(range(np.size(X, axis=1)), i)
    m = np.size(X, axis=0)
    for comb in combs:
        tmp = np.ones((1, m))
        for i in comb:
            tmp *= X[:, i]
        X = np.append(X, np.transpose(tmp), axis=1)
    return X


def normalize(X):
    X_norm = (X - X.min(axis=0)) / X.ptp(axis=0)
    return X_norm


def prepare_data(filename):
    data = pd.read_csv(filename)
    label = np.asarray(data.iloc[:, 5])
    # set C1 as predicted class
    label = np.asarray(label == "C1", dtype=int)
    data_np = np.asarray(data.iloc[:, 0:5])
    if (sum(sum(data_np > 1)) + sum(sum(data_np < 0))) >= 1:
        data_np = normalize(data_np)
    nrows = np.size(label)
    ntrain = int(nrows*0.8)
    # train = np.append(data_np[:ntrain, :], np.ones((ntrain, 1)), axis=1)
    # test = np.append(data_np[ntrain:, :], np.ones((nrows - ntrain, 1)), axis=1)
    # correction after response from assistant
    train = data_np[:ntrain, :]
    test = data_np[ntrain:, :]
    train_l = label[:ntrain]
    test_l = label[ntrain:]
    return train, test, train_l, test_l


def logistic_regression(filename):
    train, test, train_l, test_l = prepare_data(filename)
    # without gradient of log_likelihood d<=5, othw d<=10
    d = 10
    miss_train = []
    miss_test = []
    for i in range(d):
        train_tmp = polynomial_expansion(train, i)
        test_tmp = polynomial_expansion(test, i)
        # correction after response from assistant
        train_tmp = np.append(train_tmp, np.ones((np.shape(train_tmp)[0], 1)), axis=1)
        test_tmp = np.append(test_tmp, np.ones((np.shape(test_tmp)[0], 1)), axis=1)
        beta = logreg_mle(train_tmp, train_l)
        pred_train = logreg_predict(beta, train_tmp)
        pred_test = logreg_predict(beta, test_tmp)
        miss_train.append(sum(pred_train != train_l) / np.size(train_l))
        miss_test.append(sum(pred_test != test_l) / np.size(test_l))

    x = np.linspace(1, d, d)
    plt.plot(x, miss_train, 'b-o', label='Training set')
    plt.plot(x, miss_test, 'r-o', label='Validation set')
    plt.xlabel('Polynomial expansion degree')
    plt.ylabel('Misclassification rate')
    plt.xticks(ticks=x, labels=list(map(str, list(range(1, d+1)))))
    ymax = min([max([1.3 * max(miss_test), 1.3 * max(miss_train)]), 1]) 
    plt.ylim(0, ymax)
    plt.title("Misclassification rates of logistic regression on '{}' dataset".format(filename.split('2')[0]))
    plt.legend()
    plt.savefig('{}.png'.format(filename.split('.')[0]), dpi=600)
    plt.close()
    return


if __name__ == "__main__":
    logistic_regression('painted2.csv')
    logistic_regression('housing2.csv')
