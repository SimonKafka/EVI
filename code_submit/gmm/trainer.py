import numpy as np
from environment import GMM
from method import gradient, rbf_kernel
import matplotlib.pyplot as plt
# from load import generate_sample_data


class Trainer(object):

    def __init__(self, method, particles, tau, inner_iteration, outer_iteration):
        self.method = method
        self.n_particles = particles
        self.n_components = 2
        self.dim = 1
        self.tau = tau
        self.inner_iteration = inner_iteration
        self.outer_iteration = outer_iteration

    def train(self):

        # x_train, mus_star, w_star, vars_star = generate_sample_data(self.n_components)
        generate_sample_data = np.load('generate_data.npz')
        x_train = generate_sample_data['x_train']
        vars_star = generate_sample_data['vars_star']
        approx = GMM(x_train, vars_star)

        # initial mus
        mus = np.random.randn(self.n_particles, self.n_components, self.dim)   # p x c x d
        mus_initial = mus
        N = self.n_particles * self.n_components * self.dim
        dynamics = np.zeros([self.outer_iteration, self.n_particles, self.n_components, self.dim])
        # for adagrad
        adag = np.zeros([self.n_particles, self.n_components, self.dim])  # p x c x d
        lr = 1

        for i in range(self.outer_iteration):

            if self.method == 'evi':
                # Evi begin
                for j in range(self.inner_iteration):

                    mu_grad = approx.log_gradient(mus)
                    updates_svgd, updates_evi = gradient(self.method, mus, mus_initial, mu_grad, self.tau, kernel='rbf')

                    grad_now = np.reshape(updates_evi, (1, N))
                    if np.sqrt(np.inner(grad_now, grad_now)) < 1e-8:
                        print(j)
                        break

                    step_l = 1e-3
                    # BB Step - length
                    if j > 0:
                        y_k = grad_now - grad_old
                        s_k = np.reshape(mus, (1, N)) - np.reshape(mus_old, (1, N))
                        step_l = np.inner(s_k, s_k) / np.inner(s_k, y_k)

                    grad_old = grad_now
                    mus_old = mus
                    mus = mus - step_l * updates_evi

                mus_initial = mus
                dynamics[i, :, :, :] = mus
                dynamics[i+1, :, :] = mus
                plt.plot(mus[:, 0], mus[:, 1], 'bo', ms=1.3, alpha=0.6)
                plt.draw()
                plt.pause(0.1)
                plt.clf()

                # Env end

            elif self.method == 'svgd':
                # SVGD begin
                mu_grad = approx.log_gradient(mus)
                updates_svgd, updates_env = gradient(self.method, mus, mus_initial, mu_grad, self.tau, kernel='rbf')
                adag += updates_svgd ** 2  # update sum of gradient's square
                mus = mus + lr * updates_svgd / np.sqrt(adag + 1e-12)
                dynamics[i, :, :, :] = mus
                # SVGD end
            else:
                print('input method')
        return dynamics




def main():

    # all input parameters
    method_name = 'evi'
    # method_name = 'svgd'
    repeat = 1
    np.random.seed(0)



    if method_name == 'svgd':
        inner_iter = 1
        outer_iter = 1000
    else:
        inner_iter = 300
        outer_iter = 100
    for rep in range(repeat):
        trainer = Trainer(method=method_name, particles=100, tau=0.01, inner_iteration=inner_iter, outer_iteration=outer_iter)
        dynamics = trainer.train()

    np.savez(method_name, dynamics=dynamics)


if __name__ == '__main__':
    main()
