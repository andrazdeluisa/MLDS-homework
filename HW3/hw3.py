import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from sklearn.ensemble import RandomForestClassifier


def ordreg_log_likelihood(X, y, beta, t):
    tmp = X @ beta
    tmp2 = inv_logit(t[y + 1] - tmp) - inv_logit(t[y] - tmp)
    eps = 1e-20
    idx = tmp2 < eps
    res = np.log(eps) * np.ones((len(tmp2)))
    res[~idx] = np.log(tmp2[~idx])
    return sum(res)


def inv_logit(x):
    pos = x > 0
    res = np.empty(x.size, dtype=np.float)
    res[pos] = 1. / (1 + np.exp(-x[pos]))
    exp_t = np.exp(x[~pos])
    res[~pos] = exp_t / (1. + exp_t)
    return res


def ordreg_mle(X, y):
    n = np.shape(X)[1]

    def ll_tmp(x):
        beta = x[:n]
        delta = x[n:]
        t = np.array([-np.inf, 0, np.inf])
        for i in range(np.size(delta)):
            t = np.insert(t, i + 2, t[i + 1] + delta[i])
        return - ordreg_log_likelihood(X, y, beta, t)

    beta0 = np.zeros(n)
    k = np.size(np.unique(y))
    d = 1
    delta0 = np.asarray([d] * (k - 2))
    my_bounds = [(None, None)] * n + [(0, None)] * (k - 2)
    x0 = np.append(beta0, delta0)
    # without gradient of log_likelihood
    res, _, _ = fmin_l_bfgs_b(ll_tmp, x0=x0, bounds=my_bounds, approx_grad=True, maxfun=1e6, maxiter=1e6)
    beta = res[:n]
    delta = res[n:]
    return beta, delta


def log_loss(pred, y):
    loss = pred[range(np.shape(pred)[0]), y]
    eps = 1e-20
    idx = loss < eps
    res = np.log(eps) * np.ones((len(loss)))
    res[~idx] = np.log(loss[~idx])
    return -np.mean(res)


def ordreg_predict(X, beta, t):
    probs = np.zeros((np.shape(X)[0], len(t) - 1))
    tmp = X.dot(beta)
    for j in range(len(t) - 1):
        tmp2 = inv_logit(t[j+1] - tmp) - inv_logit(t[j] - tmp)
        probs[:, j] = tmp2
    return probs


def naive_predict(X):
    pi = [0.15, 0.1, 0.05, 0.4, 0.3]
    return np.asarray([pi for i in range(np.shape(X)[0])])


def mis_rate(probs, labels):
    preds = np.argmax(probs, axis=1)
    r = np.mean(preds != labels)
    return r


def near_mis_rate(probs, labels):
    preds = np.argmax(probs, axis=1)
    r = sum(abs(preds - labels) > 1)
    r /= np.size(preds)
    return r


def prepare_data(filename):
    data = pd.read_csv(filename, sep=';')
    label = np.asarray(data.iloc[:, 0])
    label = np.asarray(list(map(response, label)), dtype=int)
    data_np = np.asarray(data.iloc[:, 1:])
    data_np[:, 1] = np.asarray(list(map(sex, data_np[:, 1])), dtype=int)
    data_np = np.array(data_np, dtype=float)
    idx = [True] * data_np.shape[1]
    idx[1:3] = False, False
    data_np[:, idx] = standardize(data_np[:, idx])
    data_np[:, 2] = data_np[:, 2] - 1
    data_np = np.append(data_np, np.ones((np.shape(data_np)[0], 1)), axis=1)
    nrows = np.size(label)
    ntrain = int(nrows*0.8)
    train_l = label[:ntrain]
    test_l = label[ntrain:]
    train = data_np[:ntrain, :]
    test = data_np[ntrain:, :]
    return train, train_l, test, test_l


def standardize(X):
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_std


def response(x):
    grades = {"very poor": 0, "poor": 1, "average": 2, "good": 3, "very good": 4}
    return grades[x]


def sex(x):
    return x == "M"


def ordinal_regression(filename):
    train, train_l, test, test_l = prepare_data(filename)
    beta, delta = ordreg_mle(train, train_l)
    t = np.array([-np.inf, 0, np.inf])
    for i in range(np.size(delta)):
        t = np.insert(t, i + 2, t[i + 1] + delta[i])
    preds_test = ordreg_predict(test, beta, t)
    preds_naive = naive_predict(test)
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=1, min_samples_split=2)
    rf.fit(train, train_l)
    preds_rf = rf.predict_proba(test)
    mis_rates = [mis_rate(preds_test, test_l), mis_rate(preds_naive, test_l), mis_rate(preds_rf, test_l)]
    log_losses = [log_loss(preds_test, test_l), log_loss(preds_naive, test_l), log_loss(preds_rf, test_l)]
    print('Misclassification rates & log-loss: single model')
    print('Ordinal regression: {}, {}'.format(mis_rates[0], round(log_losses[0], 3)))
    print('Baseline model: {}, {}'.format(mis_rates[1], round(log_losses[1], 3)))
    print('Random forest: {}, {}'.format(mis_rates[2], round(log_losses[2], 3)))
    return beta, mis_rates, log_losses


def ordinal_regression_cv(filename, k):
    train, train_l, test, test_l = prepare_data(filename)
    train = np.append(train, test, axis=0)
    train_l = np.append(train_l, test_l)
    # cross validation
    idxs = np.tile(np.asarray(list(range(k))), (len(train_l) // k) + 1)
    idxs = idxs[:len(train_l)]
    mis_rates, log_losses = [], []
    mis_rates_naive, log_losses_naive = [], []
    mis_rates_rf, log_losses_rf = [], []
    for i in range(k):
        train_cv = train[idxs != i, :]
        test_cv = train[idxs == i, :]
        train_y = train_l[idxs != i]
        test_y = train_l[idxs == i]
        beta, delta = ordreg_mle(train_cv, train_y)
        t = np.array([-np.inf, 0, np.inf])
        for i in range(np.size(delta)):
            t = np.insert(t, i + 2, t[i + 1] + delta[i])
        pred_test_cv = ordreg_predict(test_cv, beta, t)
        naive_pred_cv = naive_predict(test_cv)
        rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=1, min_samples_split=2)
        rf.fit(train_cv, train_y)
        pred_rf_cv = rf.predict_proba(test_cv)
        mis_rates.append(mis_rate(pred_test_cv, test_y))
        log_losses.append(log_loss(pred_test_cv, test_y))
        mis_rates_naive.append(mis_rate(naive_pred_cv, test_y))
        log_losses_naive.append(log_loss(naive_pred_cv, test_y))
        mis_rates_rf.append(mis_rate(pred_rf_cv, test_y))
        log_losses_rf.append(log_loss(pred_rf_cv, test_y))
    print('Misclassification rates & log-loss & log-loss std. error: 10-fold cross validation')
    print('Ordinal regression: {}, {}, {}'.format(round(np.mean(mis_rates), 3), round(np.mean(log_losses), 3), round(np.std(log_losses), 3)))
    print('Baseline model: {}, {}, {}'.format(round(np.mean(mis_rates_naive), 3), round(np.mean(log_losses_naive), 3), round(np.std(log_losses_naive), 3)))
    print('Random forest: {}, {}, {}'.format(round(np.mean(mis_rates_rf), 3), round(np.mean(log_losses_rf), 3), round(np.std(log_losses_rf), 3)))
    mis_rates = np.asarray([mis_rates, mis_rates_naive, mis_rates_rf])
    log_losses = np.asarray([log_losses, log_losses_naive, log_losses_rf])
    return mis_rates, log_losses


if __name__ == "__main__":
    FILENAME = 'HW3/dataset.csv'
    coeff, mr1, ll1 = ordinal_regression(FILENAME)
    print('Coefficients:')
    print(coeff)
    mr_cv, ll_cv = ordinal_regression_cv(FILENAME, 10)
