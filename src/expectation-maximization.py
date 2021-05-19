import numpy as np


def error(N, x_t, x_r):
    return 1 / N * np.sum(x_t - x_r)


def gaussian(x, mi, sigma):
    prod = np.prod(1/(sigma*np.sqrt(2*np.pi)) * np.e**(-1/2*((x-mi)/sigma)**2))
    return prod


K = 10
D = 100
N = 2000

pi = np.full(fill_value=1/K, shape=K)
mi = np.random.rand(K, D)
sigma = np.random.rand(K)

x = np.random.rand(N, D)

gamma = np.empty(shape=(N, K))

for i in range(0, 5):
    # E step
    for n in range(0, N):
        for k in range(0, K):
            temp = 0
            for j in range(0, K):
                temp = temp + pi[j] * gaussian(x[n], mi[j], sigma[j])
            gamma[n, k] = (pi[k] * gaussian(x[n], mi[k], sigma[k]))/temp

    # M step
    for k in range(0, K):
        mi[k] = (np.dot(gamma[:, k], x))/np.sum(gamma[:, k])
        sigma[k] = np.sum(np.dot(np.transpose(gamma[:, k]), np.power(x - np.reshape(mi[k, :], (1, mi[k, :].size)), 2)))/D*np.sum(gamma[:, k])
        pi[k] = np.sum(gamma[:, k])/N
        # TODO find x_true
        print(error(N, x, mi[k]))
