import numpy as np
import almost_io


# Bishop page 439


def error(x_t, x_r):
    return 1 / x_t.shape[0] * (np.linalg.norm(x_t - x_r)) ** 2


def gauss(x, mi, sigma):
    return np.prod(
        np.multiply(
            1 / np.sqrt(2 * np.pi * sigma),
            np.power(np.e, (np.multiply(-1 / (2 * sigma), np.power((x[:, None] - mi), 2))))
        )
        , axis=2)


def em(x, K):
    """

    :param x: input matrix to be given (e.g pictue)
    :param K: number of clusters
    :return: output matrix
    """
    # preprocess
    dims = [x.shape[0], x.shape[1]]
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[-1]) / 255

    # num of dimensions of each data
    D = x.shape[1]
    # num of data
    N = x.shape[0]

    # init
    pi = np.full(fill_value=1 / K, shape=(K, 1))
    mi = np.random.uniform(low=0, high=1, size=(K, D))
    sigma = np.random.uniform(low=0, high=1, size=(K, 1))

    log_lik_old = -10000000000
    log_lik_new = -5000000000

    iterations = 0
    print("\nStarting EM process for K = "+str(K)+"...")

    while np.abs(log_lik_new - log_lik_old) > 1e-5 and iterations < 2000:

        # E step
        p = gauss(x, mi, sigma)
        # a posteriori probability
        gamma = np.multiply(p, pi.T) / np.sum(np.multiply(p, pi.T), axis=1).reshape(-1, 1)

        # M step
        mi = np.dot(gamma.T, x) / np.sum(gamma, axis=0).reshape(-1, 1)
        for k in range(0, K):
            sigma[k] = np.sum(np.dot(np.transpose(gamma[:, k]), np.power(x - mi[k, :], 2))) / (D * np.sum(gamma[:, k]))
        pi = (np.sum(gamma, axis=0) / N).reshape(-1, 1)

        log_lik_old = log_lik_new
        log_lik_new = np.sum(np.log(np.sum(np.multiply(gauss(x, mi, sigma), pi.T), axis=1)))
        iterations = iterations + 1
        print("Iteration = "+str(iterations)+", log likelihood score = " + str(log_lik_new))

    print("\nFor K = " + str(K) + ", error = " + str(error(x * 255, mi[np.argmax(gamma, axis=1)] * 255)))
    img_to_save = mi[np.argmax(gamma, axis=1)] * 255
    img_to_save = img_to_save.reshape(dims + [img_to_save.shape[-1]])
    return img_to_save
