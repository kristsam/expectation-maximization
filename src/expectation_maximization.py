import numpy as np

# Bishop page 439
from PIL import Image as Img


def error(N, x_t, x_r):
    return 1 / N * np.sum(x_t - x_r)


def gaussian(x, mi, sigma):
    prod = np.prod(1 / (sigma * np.sqrt(2 * np.pi)) * np.e ** (-1 / 2 * ((x - mi) / sigma) ** 2), axis=1)
    return prod.reshape(prod.shape[0], 1)


def save(x, mi, dimension):
    picked = np.empty(shape=x.shape)
    for n in range(0, x.shape[0]):
        for d in range(0, mi.shape[1]):
            picked[n, d] = np.min(np.abs(x[n, d] - mi[:, d]))
    picked = picked.reshape(dimension + [picked.shape[-1]])
    pil_img = Img.fromarray(np.uint8(picked)).convert('RGB')
    path = '../data/scale1.jpg'
    pil_img.save(path)

    print("Image saved in: " + path)


def em(x, dimensions, K):
    """

    :param x:
    :param dimensions: The output image dimensions
    :param K:
    :return:
    """
    # num of dimensions of each data
    D = x.shape[1]
    # num of data
    N = x.shape[0]

    pi = np.full(fill_value=1 / K, shape=(K, 1))
    # TODO remove nan values
    mi = np.random.rand(K, D) + np.mean(x, axis=0)
    sigma = np.random.rand(K, 1) * 255
    # sigma = np.full(shape=(K, 1), fill_value=1/K)
    # sigma = [np.cov(x.T)*np.identity(x.shape[1]) for z in range(K)]

    # aposteriori probability
    gamma = np.empty(shape=(N, K))

    for i in range(0, 6):
        # E step
        for n in range(0, N):
            p = np.multiply(pi, gaussian(x[n], mi, sigma))
            gamma[n, :] = (p / np.sum(p, axis=0)).T
        # M step
        for k in range(0, K):
            mi[k] = (np.dot(gamma[:, k], x)) / np.sum(gamma[:, k])
            sigma[k] = np.sqrt(np.sum(np.dot(np.transpose(gamma[:, k]),
                                             np.power(x - mi[k, :, np.newaxis].T, 2))) / D * np.sum(gamma[:, k]))
            pi[k] = np.sum(gamma[:, k]) / N
            print("Iteration = "+str(i+1)+", category = "+str(k+1)+", error = "+str(error(N, x, mi[k])))
        print()
    save(x, mi, dimensions)
