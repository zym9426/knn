from numpy import *
import numpy as np
import struct


def read_image(file_name):
    file_handle=open(file_name,"rb")
    file_content=file_handle.read()
    offset=0
    head = struct.unpack_from('>IIII', file_content, offset)
    offset += struct.calcsize('>IIII')
    imgNum = head[1]
    rows = head[2]
    cols = head[3]
    images = np.empty((imgNum, 784))
    image_size = rows*cols
    fmt = '>' + str(image_size) + 'B'

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        offset += struct.calcsize(fmt)
    return images


def read_label(file_name):
    file_handle = open(file_name, "rb")
    file_content = file_handle.read()

    head = struct.unpack_from('>II', file_content, 0)
    offset = struct.calcsize('>II')

    labelNum = head[1]

    bitsString = '>' + str(labelNum) + 'B'
    label = struct.unpack_from(bitsString, file_content, offset)
    return np.array(label)


def edistance(test_data, dataSet):
    dataSetSize = dataSet.shape[0]
    distance1 = tile(test_data, (dataSetSize)).reshape((60000,784))-dataSet
    distance2 = distance1**2
    distance3 = distance2.sum(axis=1)
    distances4 = distance3**0.5
    print('Euclidean Distance:', distances4)

def test_distance():
    train_image = "train-images-idx3-ubyte"
    test_image = "t10k-images-idx3-ubyte"
    train_x = read_image(train_image)
    test_x = read_image(test_image)
    testNum = 1000
    for i in range(testNum):
        result = edistance(test_x[i], train_x)

if __name__ == "__main__":
    test_distance()
