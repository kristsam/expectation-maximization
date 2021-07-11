import os
import almost_io
import expectation_maximization as em

path_to_data = os.path.dirname(os.path.realpath(__file__)) + '\\..\\data\\'
img = 'im'

K = [2**i for i in range(0, 8)]

x = almost_io.read(path_to_data + img + ".jpg")
for k in K:
    image = em.em(x, k)
    almost_io.save(pic=image, filename=path_to_data + img + "_k" + str(k) + ".jpg")
