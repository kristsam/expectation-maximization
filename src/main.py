import os
import almost_io
import expectation_maximization as em

path_to_data = os.path.dirname(os.path.realpath(__file__)) + '\\..\\data\\'
img = 'im'

K = [2**i for i in range(0, 7)]

x = almost_io.read(path_to_data + img + ".jpg")
for k in K:
    em.em(x, k, file_name=path_to_data + img + "_k" + str(k) + ".jpg")
