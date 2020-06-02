import numpy as np

class evi():

    def __init__(self):
        pass

    def rbf_kernel(self, theta, h=0.05):

        diff = theta[:, None, :] - theta[None, :, :]
        kxy = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * h ** 2))
        sumkxy = np.sum(kxy, axis=1, keepdims=True)
        gradK = -diff * kxy[:, :, None] / h ** 2
        dxkxy = np.sum(gradK, axis=0)
        obj = np.sum(np.transpose(gradK, axes=[1, 0, 2]) / sumkxy, axis=1)

        return dxkxy, sumkxy, obj

    def gradient(self, x, x_initial, grad, tau, kernel='rbf', **kernel_params):
        assert x.shape == grad.shape, 'illegal inputs and grads'
        p_shape = np.shape(x)
        x = np.reshape(x, (np.shape(x)[0], -1))
        x_initial = np.reshape(x_initial, (np.shape(x_initial)[0], -1))
        grad = np.reshape(grad, (np.shape(grad)[0], -1))

        if kernel == 'rbf':
            dxkxy, sumkxy, obj = self.rbf_kernel(x, **kernel_params)

        Akxy = (x - x_initial) / tau + (- dxkxy / sumkxy - obj - grad)
        Akxy = np.reshape(Akxy, p_shape)

        return Akxy

    def evi_updates(self, x0, lnprob, inner_iteration, outer_iteration, tau, X_test, y_test, evaluation):
        particles = x0.copy()
        particles_initial = x0.copy()
        results = []
        N = particles.shape[0] * particles.shape[1]
        lr = .1

        for i in range(outer_iteration):
            adag = np.zeros(particles.shape)

            for j in range(inner_iteration):

                lnpgrad = lnprob(particles)
                updates_evi = self.gradient(particles, particles_initial, lnpgrad, tau, kernel='rbf')

                '''
                grad_now = np.reshape(updates_evi, (1, N))
                if np.sqrt(np.inner(grad_now, grad_now)) < 1e-8:
                    print(j)
                    break

                step_l = 1e-7
                # BB Step - length
                if j > 0:
                    y_k = grad_now - grad_old
                    s_k = np.reshape(particles, (1, N)) - np.reshape(particles_old, (1, N))
                    step_l = np.inner(s_k, s_k) / np.inner(s_k, y_k)

                grad_old = grad_now
                particles_old = particles
                particles = particles - step_l * updates_evi
                '''

                adag += updates_evi ** 2  # update sum of gradient's square
                particles = particles - lr * updates_evi / np.sqrt(adag + 1e-12)

            particles_initial = particles
            results.append(evaluation(particles, X_test, y_test))

        return np.asarray(results)















