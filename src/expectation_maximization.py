import numpy as np
import almost_io

# Bishop page 439


def error(N, x_t, x_r):
    return 1 / N * np.sum(x_t - x_r)


def gaussian(x, mi, sigma):
    prod = np.prod(1 / (sigma * np.sqrt(2 * np.pi)) * np.e ** (-1 / 2 * ((x - mi) / sigma) ** 2), axis=1)
    return prod.reshape(prod.shape[0], 1)


def em(x, K, file_name, is_image=True):
    """

    :param x:
    :param K:
    :param file_name: File to save the document
    :param is_image:
    :return:
    """
    if is_image:
        # x = x.transpose(2, 0, 1).reshape(-1, 3)
        dims = [x.shape[0], x.shape[1]]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[-1]) / 255

    # num of dimensions of each data
    D = x.shape[1]
    # num of data
    N = x.shape[0]

    pi = np.full(fill_value=1/K, shape=(K, 1))
    # TODO remove nan values
    # mi = np.random.rand(K, D) + np.mean(x, axis=0)
    mi = np.full(fill_value=1/K, shape=(K, D))
    # sigma = np.random.rand(K, 1) * 255
    sigma = np.full(fill_value=1/K, shape=(K, 1))
    # sigma = np.full(shape=(K, 1), fill_value=1/K)
    # sigma = [np.cov(x.T)*np.identity(x.shape[1]) for z in range(K)]

    # aposteriori probability
    gamma = np.empty(shape=(N, K))

    for i in range(0, 5):
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
    img_to_save = mi[np.argmax(gamma, axis=1)] * 255
    if is_image:
        img_to_save = img_to_save.reshape(dims + [img_to_save.shape[-1]])
    almost_io.save(img_to_save, file_name)
