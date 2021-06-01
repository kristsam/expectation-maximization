import numpy as np
from PIL import Image


def read(file):
    # load the image
    img = Image.open(file)
    # convert image to numpy array
    data = np.asarray(img)
    # summarize shape
    print(data.shape)
    print("Image read!")
    return data


def save(pic, filename):
    pil_img = Image.fromarray(np.uint8(pic)).convert('RGB')
    pil_img.save(filename)
    print("Image saved in: " + filename)
