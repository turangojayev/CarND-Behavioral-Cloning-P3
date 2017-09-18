import numpy
import cv2
import pandas

uniform = numpy.random.uniform


def adjust_vertices(img):
    return numpy.array([
        [(uniform() * img.shape[1], uniform() * img.shape[0]),
         (uniform() * img.shape[1], uniform() * img.shape[0]),
         (uniform() * img.shape[1], uniform() * img.shape[0]),
         (uniform() * img.shape[1], uniform() * img.shape[0])]],
        dtype=numpy.int32)
    # return \
    #     numpy.array([
    #                 [(50, uniform() * 60),
    #                  (100, 100),
    #                  (150, 60),
    #                  (20, 120)]],
    #                 dtype=numpy.int32)


# def add_shadow(img, vertices):
#     mask = numpy.zeros_like(img)
#     print(mask)
#     mask = cv2.fillPoly(mask, vertices(img), (255, 255, 255))
#     print(mask)
#     alfa = uniform(0.2, 0.8)
#     return cv2.addWeighted(mask, alfa, img, 1 - alfa, 0)



def shadow2(image):
    rows = image.shape[0]
    cols = image.shape[1]

    hlsed = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    mask = numpy.zeros(shape=(rows, cols))

    vertices = numpy.array([
        [(cols * uniform(), rows * uniform()),
         (cols * uniform(), rows * uniform()),
         (cols * uniform(), rows * uniform()),
         (cols * uniform(), rows * uniform())]],
        dtype=numpy.int32)

    mask = cv2.fillPoly(mask, vertices, 1)
    print(mask)
    print(numpy.sum(mask))

    to_be_shadowed = mask == 1
    hlsed[:, :, 1][to_be_shadowed] = hlsed[:, :, 1][to_be_shadowed] * numpy.random.uniform(0.1, 0.6)
    return cv2.cvtColor(hlsed, cv2.COLOR_HLS2RGB)


def shadow(image):
    rows = image.shape[0]
    cols = image.shape[1]

    y1, y2 = cols * uniform(size=2)
    x1, x2 = rows * uniform(size=2)
    hlsed = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = numpy.zeros(shape=(rows, cols))
    x_mesh, y_mesh = numpy.mgrid[0:rows, 0:cols]

    shadow_mask[((x_mesh - x1) * (y2 - y1) - (x2 - x1) * (y_mesh - y1) >= 0)] = 1

    to_be_shadowed = shadow_mask == 1
    print(to_be_shadowed)
    hlsed[:, :, 1][to_be_shadowed] = hlsed[:, :, 1][to_be_shadowed] * numpy.random.uniform(0.4, 0.6)
    return cv2.cvtColor(hlsed, cv2.COLOR_HLS2RGB)


df = pandas.read_csv('track2/driving_log.csv', header=None)
img = cv2.cvtColor(cv2.imread(df[0].iloc[0]), cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt

plt.imshow(img)
plt.show()

plt.imshow(shadow2(img))
plt.show()
