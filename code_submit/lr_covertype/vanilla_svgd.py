import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
from svgd import SVGD
from evi import evi
from sklearn.preprocessing import scale
from scipy.special import expit
from sklearn.datasets import load_svmlight_file

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


class BayesianLR:
    def __init__(self, X, Y, batchsize, alpha):
        self.X, self.Y = X, Y
        self.batchsize = min(batchsize, X.shape[0])

        self.N = X.shape[0]
        self.dim = X.shape[1]
        self.permutation = np.random.permutation(self.N)
        self.iter = 0
        self.alpha = alpha

    def dlnprob(self, theta):

        if self.batchsize > 0:
            batch = [i % self.N for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize)]
            ridx = self.permutation[batch]
            self.iter += 1
        else:
            ridx = np.random.permutation(self.X.shape[0])

        Xs = self.X[ridx, :]
        Ys = self.Y[ridx]

        w = theta  # logistic weights
        coff = np.matmul(Xs, w.T)
        y_hat = expit(coff)
        diff = ((nm.repmat(np.vstack(Ys), 1, theta.shape[0]) + 1) / 2.0 - y_hat).T  # change from {-1,1} to {0,1}
        dw_data = np.matmul(diff, Xs)
        dw_prior = -w/self.alpha
        dw = dw_data * self.X.shape[0] / Xs.shape[0] + dw_prior  # re-scale

        '''
        grad_loglik_W = np.expand_dims(diff, 2) * np.expand_dims(Xs, 0)
        mean_dW = np.mean(grad_loglik_W, axis=1, keepdims=True)
        diff_dW = grad_loglik_W - mean_dW
        cov_dW = np.matmul(np.transpose(diff_dW, axes=[0, 2, 1]), diff_dW) / self.batchsize
        # H_inv = np.linalg.inv(cov_dW + 1e-2 * np.expand_dims(np.eye(self.dim), 0))
        '''
        return dw  # first order derivative


    def evaluation(self, theta, X_test, y_test):

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

    # covertype
    data = scipy.io.loadmat('covertype.mat')
    # X_input = scale(data['covtype'][:, 1:])
    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = -1

    N = X_input.shape[0]
    X_input = np.hstack([X_input, np.ones([N, 1])])
    d = X_input.shape[1]
    D = d

    # initialization
    M = 20  # number of particles
    alpha = 1
    inner_iteration = 100
    outer_iteration = 20
    n_iter = inner_iteration * outer_iteration
    repeat = 20
    results_vanilla_svgd = np.zeros(shape=(repeat, outer_iteration, 2))


    for rep in range(repeat):
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2)

        theta0 = np.random.normal(0, np.sqrt(alpha), (M, D))
        # EVI
        model = BayesianLR(X_train, y_train, 256, alpha)
        results_vanilla_svgd[rep, :, :] = SVGD().svgd_updates(x0=theta0, lnprob=model.dlnprob, inner_iteration=inner_iteration, outer_iteration=outer_iteration,
                                  stepsize=1, X_test=X_test, y_test=y_test, evaluation=model.evaluation)
        print('one trial ends')

    np.savez('lr_vanilla_svgd', results_mean=np.mean(results_vanilla_svgd, axis=0), results_var=np.std(results_vanilla_svgd, axis=0))
