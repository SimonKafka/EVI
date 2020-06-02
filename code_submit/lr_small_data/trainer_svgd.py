import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
from svgd import SVGD
from evi import evi
from sklearn.preprocessing import scale
from scipy.special import expit


def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


class BayesianLR:
    def __init__(self, X, Y, alpha):
        self.X, self.Y = X, Y
        self.alpha = alpha
        self.N = X.shape[0]
        self.dim = X.shape[1]
        self.iter = 0

    def dlnprob(self, theta):

        Xs = self.X
        Ys = self.Y

        w = theta  # logistic weights
        coff = np.matmul(Xs, w.T)
        y_hat = expit(coff)
        diff = ((nm.repmat(np.vstack(Ys), 1, theta.shape[0]) + 1) / 2.0 - y_hat).T  # change from {-1,1} to {0,1}
        dw_data = np.matmul(diff, Xs)
        dw_prior = -w/self.alpha
        dw = dw_data + dw_prior  # re-scale
        '''
        grad_loglik_W = np.expand_dims(diff, 2) * np.expand_dims(Xs, 0)
        mean_dW = np.mean(grad_loglik_W, axis=1, keepdims=True)
        diff_dW = grad_loglik_W - mean_dW
        cov_dW = np.matmul(np.transpose(diff_dW, axes=[0, 2, 1]), diff_dW) / self.batchsize
        # H_inv = np.linalg.inv(cov_dW + 1e-2 * np.expand_dims(np.eye(self.dim), 0))
        '''
        return dw  # first order derivative

    def evaluation(self, theta, X_test, y_test):
        # theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
        for t in range(M):
            coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
            prob[:, t] = expit(-coff)

        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        prob = replaceZeroes(prob)
        llh = np.mean(np.log(prob))
        return [acc, llh]


if __name__ == '__main__':


    data = scipy.io.loadmat('benchmarks.mat')
    splitIdx = 19
    dataName = 'splice'  # 2991 1000 2175   60  20
    X_train = data[dataName]['x'][0, 0][data[dataName]['train'][0, 0][splitIdx - 1, :] - 1, :]
    y_train = data[dataName]['t'][0, 0][data[dataName]['train'][0, 0][splitIdx - 1, :] - 1, :]
    X_test = data[dataName]['x'][0, 0][data[dataName]['test'][0, 0][splitIdx - 1, :] - 1, :]
    y_test = data[dataName]['t'][0, 0][data[dataName]['test'][0, 0][splitIdx - 1, :] - 1, :]
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    X_train = np.hstack([X_train, np.ones([X_train.shape[0], 1])])
    X_test = np.hstack([X_test, np.ones([X_test.shape[0], 1])])
    D = X_train.shape[1]


    # initialization
    M = 100  # number of particles
    alpha = 1e-2
    inner_iteration = 25
    outer_iteration = 20
    n_iter = inner_iteration * outer_iteration
    repeat = 20
    results_SVGD = np.zeros(shape=(repeat, outer_iteration, 2))

    for rep in range(repeat):
        theta0 = np.random.normal(0, np.sqrt(alpha), (M, D))
        # SVGD

        model = BayesianLR(X_train, y_train, alpha)  # batchsize = 32
        results_SVGD[rep, :, :] = SVGD().update(x0=theta0, lnprob=model.dlnprob, inner_iteration=inner_iteration,
                                                outer_iteration=outer_iteration,
                                                stepsize=0.01, X_test=X_test, y_test=y_test, evaluation=model.evaluation,
                                                debug=True)

    np.savez('lr_svgd', results_mean=np.mean(results_SVGD, axis=0), results_var=np.std(results_SVGD, axis=0))
