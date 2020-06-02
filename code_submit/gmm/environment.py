import numpy as np

class GMM(object):

    def __init__(self, y, vars_star):
        self.prior_var = 1
        self.y = y  # observation: n x 1
        self.vars_star = vars_star

    def log_gradient(self, mu):
        mu_model = mu.copy()    # mu: p x c x d
        mu_model[:, 1, :] = mu_model[:, 0, :] + mu_model[:, 1, :]  # p x c x d
        diff = np.expand_dims(self.y, 0) - np.expand_dims(mu_model, 2)  # p x c x n x d
        diff_inv_vars = diff / self.vars_star
        sq_dist_inv_vars = np.square(diff) / self.vars_star   # p x c x n x d
        log_components = -0.5 * sq_dist_inv_vars  # p x c x n x d
        prob = np.exp(log_components)
        gradient_prob = prob / np.sum(prob, axis=1, keepdims=True)  # p x c x n x d
        gradient_prob = np.multiply(gradient_prob, diff_inv_vars)
        gradient_prob[:, 0, :, :] = gradient_prob[:, 0, :, :] + gradient_prob[:, -1, :, :]
        score = np.sum(gradient_prob, axis=2)   # p x c x d
        return score - mu/self.prior_var


    def log_prob(self, mu_plot):
        mu_plot_model = mu_plot.copy()  # mu: p x c x d
        mu_plot_model[:, 1, :] = mu_plot_model[:, 0, :] + mu_plot_model[:, 1, :]  # p x c x d
        diff = np.expand_dims(self.y, 0) - np.expand_dims(mu_plot_model, 2)  # p x c x n x d
        sq_dist_inv_vars = np.square(diff) / self.vars_star  # p x c x n x d
        log_components = -0.5 * sq_dist_inv_vars  # p x c x n x d
        prob = np.exp(log_components)  # / 2 / math.sqrt(2*np.pi*self.vars_star)   # p x c x n x d
        prob = np.sum(prob, axis=1)  # p x n x d
        log_likelihood = np.sum(np.log(prob), axis=1)   # p x d
        log_prior = np.sum(mu_plot**2, axis=1) / 2 / self.prior_var  # p x d
        return np.squeeze(log_likelihood - log_prior)

