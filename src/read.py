import numpy as np
from PIL import Image


def read(file):

    # load the image
    img = Image.open(file)
    # convert image to numpy array
    data = np.asarray(img)
    rows = data.shape[0]
    columns = data.shape[1]
    data = data.transpose(2, 0, 1).reshape(-1, 3)
    # summarize shape
    print(data.shape)
    print("Image read!")
    return data, [rows, columns]
