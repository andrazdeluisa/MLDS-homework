import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt.solvers as cvx
from cvxopt import matrix


class SVR:

    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def fit(self, X, y):
        tol = 1e-3
        c = 1 / self.lambda_
        tmp1 = np.array([1, -1])
        tmp2 = np.eye(len(y))
        P = np.kron(tmp2, tmp1)
        K = self.kernel(X, X)
        n = P.shape[1]
        G = np.kron(np.array([[1], [-1]]), np.eye(n))
        h = np.append(c * np.ones(n), np.zeros(n))
        res = cvx.qp(matrix(P.T.dot(K).dot(P)), matrix(self.epsilon * np.ones(n) - y.T.dot(P)), matrix(G), matrix(h), matrix(np.ones((1, len(y))).dot(P)), matrix(np.zeros((1, 1))))
        alpha = np.reshape(np.array(res['x']), (len(y), 2))
        idx1 = alpha > tol
        idx2 = alpha < c - tol
        wx = K.dot(alpha.dot(tmp1))
        tmp3 = y - wx - self.epsilon * np.ones(len(y))
        tmp4 = y - wx + self.epsilon * np.ones(len(y))
        m = max(tmp3[np.logical_or(idx1[:, 1], idx2[:, 0])])
        M = min(tmp4[np.logical_or(idx1[:, 0], idx2[:, 1])])
        b = (m + M) / 2
        return SVRModel(X, alpha, b, self.kernel)


class SVRModel:

    def __init__(self, X, alpha, b, kernel):
        self.alpha = alpha
        self.X = X
        self.kernel = kernel
        self.b = b

    def predict(self, X_):
        K = self.kernel(self.X, X_)
        tmp1 = np.array([1, -1])
        wx = K.T.dot(self.alpha.dot(tmp1))
        return wx + self.b * np.ones(len(wx))

    def get_alpha(self):
        return self.alpha

    def get_b(self):
        return self.b


class Polynomial:

    def __init__(self, M):
        self.M = M

    def __call__(self, A, B):
        return (A.dot(B.T) + 1) ** self.M


class RBF:

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, A, B):
        if len(np.shape(A)) == 1:
            A = np.reshape(A, (1, len(A)))
            if len(np.shape(B)) == 1:
                B = np.reshape(B, (1, len(B)))
        if len(np.shape(B)) == 1:
            B = np.reshape(B, (1, len(B)))
        tmp = - 2 * A.dot(B.T)
        K = (np.add(np.add(tmp.T, np.diag(A.dot(A.T))).T, np.diag(B.dot(B.T))))
        if np.shape(K)[0] == 1 or np.shape(K)[1] == 1:
            K = np.reshape(K, (max(np.shape(K))))
        return np.exp(- K / (2 * self.sigma ** 2))


def standardize(X):
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_std, m, s


def sine_evaluation():
    data = pd.read_csv('HW4/sine.csv', sep=',')
    label = np.asarray(data.iloc[:, -1])
    train = np.asarray(data.iloc[:, :-1], dtype=np.float64)
    train, mean, std = standardize(train)
    K1 = Polynomial(M=11)
    K2 = RBF(sigma=0.15)
    T1 = SVR(kernel=K1, lambda_=0.01, epsilon=0.5)
    T2 = SVR(kernel=K2, lambda_=0.01, epsilon=0.5)
    model1 = T1.fit(train, label)
    model2 = T2.fit(train, label)
    alpha1 = model1.get_alpha()
    alpha2 = model2.get_alpha()
    tol = 1e-3
    alpha1[alpha1 < tol] = 0
    alpha2[alpha2 < tol] = 0
    idx1 = np.sum(alpha1 != 0, axis=1) == 1
    idx2 = np.sum(alpha2 != 0, axis=1) == 1
    tmp = np.empty((np.shape(train)[0], 3))
    tmp[:, 0] = label
    tmp[:, 1] = model1.predict(train)
    tmp[:, 2] = model2.predict(train)
    train = train * std + mean
    train = np.reshape(train, (np.shape(train)[0]))
    idx = np.argsort(train)
    train = train[idx]
    tmp = tmp[idx]
    idx1 = idx1[idx]
    idx2 = idx2[idx]
    plt.scatter(x=train, y=tmp[:, 0], s=5, c='b')
    plt.plot(train, tmp[:, 1], c='g')
    plt.plot(train, tmp[:, 2], c='r')
    plt.scatter(x=train[idx1], y=tmp[idx1, 0], s=16, c='g')
    plt.scatter(x=train[idx2], y=tmp[idx2, 0], s=4, c='r')
    plt.title('Support vector regression on sine dataset')
    plt.legend(('Polynomial kernel (M = 11)', 'RBF kernel (sigma = 0.15)', 'Data points'), loc='lower left')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('HW5/sine.png', dpi=300)
    plt.close()
    return


def prepare_data(filename):
    data = pd.read_csv(filename, sep=',')
    label = np.asarray(data.iloc[:, -1])
    data_np = np.asarray(data.iloc[:, :-1], dtype=np.float64)
    data_np, _, _ = standardize(data_np)
    nrows = np.size(label)
    ntrain = int(nrows*0.8)
    train_l = label[:ntrain]
    test_l = label[ntrain:]
    train = data_np[:ntrain, :]
    test = data_np[ntrain:, :]
    return train, train_l, test, test_l


def cv(train, train_l, kernel, iter, lambdas, k):
    best_lambda = []
    idxs = np.tile(np.asarray(list(range(k))), (len(train_l) // k) + 1)
    idxs = idxs[:len(train_l)]
    for i in iter:
        if kernel == 'pol':
            K1 = Polynomial(i)
        elif kernel == 'rbf':
            K1 = RBF(i)
        errs = []
        for lambda_ in lambdas:
            K = SVR(K1, lambda_, epsilon=2)
            rmse = []
            for j in range(k):
                train_cv = train[idxs != j, :]
                test_cv = train[idxs == j, :]
                train_y = train_l[idxs != j]
                test_y = train_l[idxs == j]
                t = K.fit(train_cv, train_y)
                p = t.predict(test_cv)
                rmse.append(np.sqrt(np.mean((p - test_y) ** 2)))
            errs.append(np.mean(rmse))
        best_lambda.append(lambdas[np.argmin(errs)])
    return best_lambda


def rmse(train, train_l, test, test_l, kernel, iter, lambdas):
    values = []
    for i in range(len(iter)):
        if kernel == 'pol':
            K1 = Polynomial(iter[i])
        elif kernel == 'rbf':
            K1 = RBF(iter[i])
        K = SVR(K1, lambdas[i], epsilon=2)
        t = K.fit(train, train_l)
        p = t.predict(test)
        values.append(np.sqrt(np.mean((p - test_l) ** 2)))
    return np.asarray(values)


def housing_plot(kernel, iter, lambdas, k=10):
    train, train_l, test, test_l = prepare_data('HW4/housing2r.csv')
    best_lambda = cv(train, train_l, kernel, iter, lambdas, k)
    const_lambda = np.ones(len(best_lambda))
    errors = rmse(train, train_l, test, test_l, kernel, iter, best_lambda)
    errors2 = rmse(train, train_l, test, test_l, kernel, iter, const_lambda)
    idx = errors2 < 15
    plt.plot(iter[idx], errors[idx], c='r')
    plt.plot(iter[idx], errors2[idx], c='b')
    if kernel == 'pol':
        plt.title('RMSE versus degree of polynomial kernel')
        plt.xlabel('M')
    elif kernel == 'rbf':
        plt.title('RMSE versus RBF kernel parameter sigma')
        plt.xlabel('Sigma')
    plt.ylabel('RMSE')
    plt.legend(('Best lambda', 'Constant lambda'), loc='upper center')
    plt.savefig('HW5/housing_{}'.format(kernel), dpi=250)
    plt.close()
    return


if __name__ == "__main__":
    sine_evaluation()
    M = np.arange(1, 11)
    sigmas = np.arange(0.2, 3, 0.2)
    lambdas_pol = np.array([0.1, 0.5, 1, 1.5, 5, 10, 20])
    lambdas_rbf = np.arange(0.7, 1.5, 0.1)
    housing_plot('pol', M, lambdas_pol, 5)
    housing_plot('rbf', sigmas, lambdas_rbf, 5)
