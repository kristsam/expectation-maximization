import numpy as np
import almost_io


# Bishop page 439


def error(x_t, x_r):
    return 1 / x_t.shape[0] * (np.linalg.norm(x_t - x_r))**2


def gaussian(x, mi, sigma):
    prod = np.prod(1 / (sigma * np.sqrt(2 * np.pi)) * np.e ** (-1 / 2 * ((x - mi) / sigma) ** 2), axis=1)
    return prod.reshape(prod.shape[0], 1)


def gauss(x, mi, sigma):
    return np.prod(
        np.multiply(
            1 / np.sqrt(2 * np.pi * sigma),
            np.power(np.e, (np.multiply(-1 / (2 * sigma), np.power((x[:, None] - mi), 2))))
        )
        , axis=2)


def e_step(x, mi, sigma, pi, K):
    p = gauss(x, mi, sigma)
    gamma = np.multiply(p, pi.T) / np.sum(np.multiply(p, pi.T), axis=1).reshape(-1, 1)
    return gamma

    # # NOTE safe road
    # gamma = np.empty(shape=(x.shape[0], K))
    # for n in range(0, gamma.shape[0]):
    #     par = 0
    #     for k in range(0, K):
    #         temp = 1
    #         for d in range(0, x.shape[1]):
    #             temp = temp * 1 / np.sqrt(2 * np.pi * sigma[k]) * np.e ** (
    #                     np.power(x[n, d] - mi[k, d], 2) * (-1 / (2 * sigma[k])))
    #         par = par + pi[k] * temp
    #     for k in range(0, K):
    #         temp = 1
    #         for d in range(0, x.shape[1]):
    #             temp = temp * 1 / np.sqrt(2 * np.pi * sigma[k]) * np.e ** (
    #                     np.power(x[n, d] - mi[k, d], 2) * (-1 / (2 * sigma[k])))
    #         gamma[n][k] = pi[k] * temp / par
    #
    # return gamma


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
    mi = np.random.uniform(low=0, high=1, size=(K, D))
    sigma = np.random.uniform(low=0, high=1, size=(K, 1))
    # sigma = [np.cov(x.T)*np.identity(x.shape[1]) for z in range(K)]

    # aposteriori probability
    gamma = np.empty(shape=(N, K))

    log_lik_old = -10000000000
    log_lik_new = -5000000000

    while np.abs(log_lik_new - log_lik_old) > 1e-8:
        # p = np.multiply(pi, guassian(x, mi, sigma))
        # E step
        gamma = e_step(x, mi, sigma, pi, K)
        # for n in range(0, N):
        #     p = np.multiply(pi, gaussian(x[n], mi, sigma))
        #     gamma[n, :] = (p / np.sum(p, axis=0)).T
        # M step
        mi = np.dot(gamma.T, x) / np.sum(gamma, axis=0).reshape(-1, 1)
        for k in range(0, K):
            sigma[k] = np.sum(np.dot(np.transpose(gamma[:, k]), np.power(x - mi[k, :], 2))) / D * np.sum(gamma[:, k])
            # print("Iteration = " + str(i + 1) + ", category = " + str(k + 1) + ", error = " + str(error(N, x, mi[k])))
        pi = (np.sum(gamma, axis=0) / N).reshape(-1, 1)

        log_lik_old = log_lik_new
        log_lik_new = np.sum(np.log(np.sum(np.multiply(gauss(x, mi, sigma), pi.T), axis=1)))
        print("Log likelihood score = "+str(log_lik_new))
    print("\nFor K = "+str(K)+", error = "+str(error(x * 255, mi[np.argmax(gamma, axis=1)] * 255)))
    img_to_save = mi[np.argmax(gamma, axis=1)] * 255
    if is_image:
        img_to_save = img_to_save.reshape(dims + [img_to_save.shape[-1]])
    almost_io.save(img_to_save, file_name)
