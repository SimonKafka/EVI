import numpy as np
# def generate_sample_data(k = 2):
k = 2
n_c_d = 500
x = []
sigma = 2.5
x1 = 1
x2 = -2
mus_star = [x1, x1+x2]
vars_star = sigma**2

for i in mus_star:
    x.append(np.random.normal(i, sigma, size=(n_c_d)))


wx = np.concatenate(x, axis=0)
wx = np.expand_dims(wx, 1)
np.random.shuffle(wx)
x_train = wx
w_star = [1. / k] * k
np.savez('generate_data', x_train=x_train, vars_star=vars_star)

# return x_train, np.asarray(mus_star), w_star, vars_star
