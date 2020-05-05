import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class KernelizedRidgeRegression:

    def __init__(self, kernel, lambda_):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        beta = np.linalg.inv(self.kernel(X, X) + self.lambda_ * np.eye(np.shape(X)[0])).dot(y)
        return KRRModel(X, beta, self.kernel)


class KRRModel:

    def __init__(self, X, beta, kernel):
        self.beta = beta
        self.X = X
        self.kernel = kernel

    def predict(self, X_):
        return self.beta.dot(self.kernel(self.X, X_))


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


def prepare_data(filename):
    data = pd.read_csv(filename, sep=',')
    label = np.asarray(data.iloc[:, -1])
    data_np = np.asarray(data.iloc[:, :-1], dtype=np.float64)
    data_np = standardize(data_np)
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
            K = KernelizedRidgeRegression(K1, lambda_)
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
        K = KernelizedRidgeRegression(K1, lambdas[i])
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
    idx = errors2 < 200
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
    plt.savefig('HW4/housing_{}'.format(kernel), dpi=600)
    plt.close()
    return


def sine_evaluation():
    data = pd.read_csv('HW4/sine.csv', sep=',')
    label = np.asarray(data.iloc[:, -1])
    train = np.asarray(data.iloc[:, :-1], dtype=np.float64)
    train = standardize(train)
    K1 = Polynomial(M=11)
    K2 = RBF(sigma=0.15)
    T1 = KernelizedRidgeRegression(kernel=K1, lambda_=1)
    T2 = KernelizedRidgeRegression(kernel=K2, lambda_=1)
    model1 = T1.fit(train, label)
    model2 = T2.fit(train, label)
    tmp = np.empty((np.shape(train)[0], 3))
    tmp[:, 0] = label
    tmp[:, 1] = model1.predict(train)
    tmp[:, 2] = model2.predict(train)
    train = np.reshape(train, (np.shape(train)[0]))
    idx = np.argsort(train)
    train = train[idx]
    tmp = tmp[idx]
    plt.scatter(x=train, y=tmp[:, 0], s=5, c='b')
    plt.plot(train, tmp[:, 1], c='g')
    plt.plot(train, tmp[:, 2], c='r')
    plt.title('Kernelized ridge regression on sine dataset')
    plt.legend(('Polynomial kernel (M = 11)', 'RBF kernel (sigma = 0.15)', 'Data points'), loc='lower left')
    plt.xlabel('X (standardized)')
    plt.ylabel('Y')
    plt.savefig('HW4/sine.png', dpi=600)
    plt.close()
    return


def standardize(X):
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_std


if __name__ == "__main__":
    sine_evaluation()
    M = np.arange(1, 11)
    sigmas = np.arange(0.5, 20, 1)
    lambdas_pol = np.arange(0.5, 30, 0.5)
    lambdas_rbf = np.arange(0.001, 1, 0.01)
    housing_plot('pol', M, lambdas_pol, 10)
    housing_plot('rbf', sigmas, lambdas_rbf, 10)
