import numpy as np
from environment import double_banana, sine, star_gaussian, banana, wave, crab

def trainer(model, method, n_particles, outer_iter):

    np.random.seed(0)
    d = model.dimension
    tau = 1e-1
    x = np.random.randn(n_particles, d)  # set initial particles as standard normal
    N = n_particles * d

    # comparision metric
    cross_entropy = []
    cross_entropy.append(-np.mean(env.logp(x)))

    if method == 'evi':
        # Evi
        x_initial = x
        dynamics = np.zeros([outer_iter, n_particles, d])
        h = 0.1

        for i in range(outer_iter):

            for j in range(5000):

                Sqy = model.grad_log_p(x)
                diff = x[:, None, :] - x[None, :, :]
                kxy = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * h ** 2)) / np.power(np.pi * 2.0 * h * h, d / 2)
                sumkxy = np.sum(kxy, axis=1, keepdims=True)
                gradK = -diff * kxy[:, :, None] / h ** 2  # N * N * 2
                dxkxy = np.sum(gradK, axis=0)  # N * 2
                obj = np.sum(np.transpose(gradK, axes=[1, 0, 2]) / sumkxy, axis=1)  # N * 2
                grad = (x - x_initial) / tau + (- dxkxy / sumkxy - obj - Sqy)
                grad_now = np.reshape(grad, (1, N))

                if np.sqrt(np.inner(grad_now, grad_now)) < 1e-9:
                    print(j)
                    break

                step_l = 1e-7
                # BB Step - length
                if j > 0:
                    y_k = grad_now - grad_old
                    s_k = np.reshape(x, (1, N)) - np.reshape(x_old, (1, N))
                    step_l = np.inner(s_k, s_k) / np.inner(s_k, y_k)

                grad_old = grad_now
                x_old = x
                x = x - step_l * grad

            x_initial = x
            cross_entropy.append(-np.mean(env.logp(x)))
            dynamics[i, :, :] = x

    return dynamics, cross_entropy  # cross_entropy




if __name__ == '__main__':
    # Pick one toy environment to play
    env_name = 'star'
    # env_name = 'sine'
    # env_name = 'double_banana'
    # env_name = 'banana'
    # env_name = 'wave'
    # env_name = 'crab'



    if env_name == 'star':
        env = star_gaussian(100, 5)  # star gaussian mixture example
    elif env_name == 'sine':
        env = sine(1., 0.003)  # unimodal sine shape example
    elif env_name == 'double_banana':
        env = double_banana(0.0, 100.0, 1.0, 0.09, np.log(30))  # bimodal double banana example
    elif env_name == 'banana':
        env = banana()
    elif env_name == 'wave':
        env = wave()
    elif env_name == 'crab':
        env = crab()


    method = 'evi'
    n_particles = 100
    dim = 2
    repeat = 1


    outer_iter = 500


    cross_entropy_results = np.zeros(shape=(repeat, outer_iter + 1))
    evolves = np.zeros(shape=(repeat, outer_iter, n_particles, dim))

    for rep in range(repeat):
        evolves[rep, :, :, :], cross_entropy_results[rep, :] = trainer(env, method, n_particles, outer_iter)
        np.savez(method, evolves=evolves[0, :, :, :], cross_entropy_results=np.mean(cross_entropy_results, axis=0))

    ngrid = 100
    # set the line space carefully to the region of your figure
    x = np.linspace(-3, 3, ngrid)
    y = np.linspace(-3, 3, ngrid)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(env.logp(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T)).reshape(ngrid, ngrid)
    np.savez(env_name, X=X, Y=Y, Z=Z)







