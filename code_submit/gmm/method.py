import numpy as np
import math
from scipy.spatial import distance


def rbf_kernel(method, x, h=-1):

    if method == 'svgd':
        H = distance.cdist(x, x, 'sqeuclidean')
        h = np.maximum(1e-9, math.sqrt(0.5*np.mean(H)/np.log(x.shape[0]+1)))
    elif method == 'evi':
        h = 0.05

    diff = x[:, None, :] - x[None, :, :]
    kxy = np.exp(-np.sum(diff**2, axis=-1) / (2 * h ** 2))
    sumkxy = np.sum(kxy, axis=1, keepdims=True)
    gradK = -diff * kxy[:, :, None] / h ** 2
    dxkxy = np.sum(gradK, axis=0)
    obj = np.sum(np.transpose(gradK, axes=[1, 0, 2]) / sumkxy, axis=1)

    return kxy, dxkxy, sumkxy, obj


def gradient(method, x, x_initial, grad, tau, kernel='rbf', **kernel_params):
    assert x.shape == grad.shape, 'illegal inputs and grads'
    p_shape = np.shape(x)

    if x.ndim > 2:
        x = np.reshape(x, (np.shape(x)[0], -1))
        x_initial = np.reshape(x_initial, (np.shape(x_initial)[0], -1))
        grad = np.reshape(grad, (np.shape(grad)[0], -1))

    if kernel == 'rbf':
        kxy, dxkxy, sumkxy, obj = rbf_kernel(method, x, **kernel_params)

    svgd_grad = (np.matmul(kxy, grad) + dxkxy) / x.shape[0]
    evi_grad = (x - x_initial) / tau + (- dxkxy / sumkxy - obj - grad)

    svgd_grad = np.reshape(svgd_grad, p_shape)
    evi_grad = np.reshape(evi_grad, p_shape)

    return svgd_grad, evi_grad