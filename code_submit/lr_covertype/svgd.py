import numpy as np
from scipy.spatial.distance import pdist, squareform


class SVGD():

    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):


        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)

        return (Kxy, dxkxy)

    def svgd_updates(self, x0, lnprob, inner_iteration, outer_iteration, stepsize, X_test, y_test, evaluation, debug=False):

        n_iter = inner_iteration * outer_iteration

        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)
        adag = np.zeros(theta.shape)
        results = []
        # results.append(evaluation(theta, X_test, y_test))

        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print
                'iter ' + str(iter + 1)

            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=-1)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]
            adag += grad_theta ** 2  # update sum of gradient's square
            theta = theta + stepsize * grad_theta / np.sqrt(adag + 1e-12)
            if (iter+1) % inner_iteration == 0:

                 results.append(evaluation(theta, X_test, y_test))

        return np.asarray(results)






