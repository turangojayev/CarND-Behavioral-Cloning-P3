import os

import cv2
import numpy
import pandas

batch_size = 32
# directory = 'data'
directory = 'track2'
csv_file = os.path.join(directory, 'driving_log.csv')

columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
df = pandas.read_csv(csv_file, header=None, names=columns)

X = numpy.array([cv2.cvtColor(cv2.imread(os.path.join(directory, path)), cv2.COLOR_BGR2RGB) for path in df.center])
y = df.steering

print(X.shape)

print(numpy.max(y))

indices_of_sharp_angles = numpy.where(numpy.abs(y) > 0.7)[0]
indices_to_upsample = numpy.random.choice(indices_of_sharp_angles, 10000)
X = numpy.vstack((X, X[indices_to_upsample]))
y = numpy.hstack((y, y[indices_to_upsample]))

print(X.shape)
