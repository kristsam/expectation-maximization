import os
import read
import expectation_maximization as em

path_to_data = os.path.dirname(os.path.realpath(__file__)) + '\\..\\data\\'

K = [2**i for i in range(0, 7)]

x, dimensions = read.read(path_to_data+"scale.jpg")
em.em(x, dimensions,K[0])
