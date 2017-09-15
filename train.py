import os
from functools import partial

import cv2
import numpy
import pandas
import scipy
from keras import optimizers, losses
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Input, Lambda, activations, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Reshape
from keras.layers.convolutional import Cropping2D
from keras.layers.pooling import AveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

batch_size = 32
directory = 'data'
# directory = '/home/turan/Downloads/data/data'
csv_file = os.path.join(directory, 'driving_log.csv')

columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
df = pandas.read_csv(csv_file, header=None, names=columns)
# df = pandas.read_csv(csv_file)

# for path in df.center:
#     if not os.path.exists(path):
#         print(path, 'doesn\'t exist')

X = numpy.array([cv2.cvtColor(cv2.imread(os.path.join(directory, path)), cv2.COLOR_BGR2RGB) for path in df.center])
y = df.steering

indices_of_sharp_angles = numpy.where(numpy.abs(y) > 0.02)[0]
indices_to_upsample = numpy.random.choice(indices_of_sharp_angles, 4000)
X = numpy.vstack((X, X[indices_to_upsample]))
y = numpy.hstack((y, y[indices_to_upsample]))

# X = numpy.concatenate((X, numpy.array(list(map(partial(cv2.flip, flipCode=1), X)))))
X = numpy.vstack((X, numpy.array(list(map(partial(cv2.flip, flipCode=1), X)))))
y = numpy.hstack((y, -y))

left = numpy.array([cv2.cvtColor(cv2.imread(os.path.join(directory, path.strip())), cv2.COLOR_BGR2RGB)
                    for path in df.left if os.path.getsize(os.path.join(directory, path.strip())) > 0])
right = numpy.array([cv2.cvtColor(cv2.imread(os.path.join(directory, path.strip())), cv2.COLOR_BGR2RGB)
                     for path in df.left if os.path.getsize(os.path.join(directory, path.strip())) > 0])

X = numpy.vstack((X, left))
y = numpy.hstack((y, df.steering + 0.15))

X = numpy.vstack((X, right))
y = numpy.hstack((y, df.steering - 0.15))

X = numpy.array(list(map(partial(cv2.resize, dsize=(int(X[0].shape[1] / 2), int(X[0].shape[0] / 2))), X)))
print(X.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=123)


def batch_generator(X, y, batch_size=32):
    while True:
        X, y = shuffle(X, y)
        for offset in range(0, len(X), batch_size):
            batch_x, batch_y = X[offset:offset + batch_size], y[offset:offset + batch_size]
            yield shuffle(batch_x, batch_y)


train_generator = batch_generator(X_train, y_train)
valid_generator = batch_generator(X_valid, y_valid)

inputs = Input(shape=X[0].shape)
normalized = Lambda(lambda x: (x - 127) / 128)(inputs)
cropped = Cropping2D(cropping=((30, 10), (0, 0)))(normalized)
conv1 = Conv2D(32, (3, 3), activation=activations.relu)(normalized)
maxpool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(48, (3, 3), activation=activations.relu)(maxpool1)
maxpool2 = MaxPooling2D((2, 2))(conv2)
conv3 = Conv2D(64, (3, 3), activation=activations.relu)(maxpool2)
maxpool3 = MaxPooling2D((2, 2))(conv3)
flattened = Flatten()(maxpool3)
dense1 = Dense(100, activation=activations.relu)(flattened)
dense2 = Dense(100, activation=activations.relu)(dense1)
dense2 = Dropout(0.85)(dense2)
output = Dense(1)(dense2)
model = Model(inputs, output)

model.compile(optimizer=optimizers.adam(), loss=losses.mse)
model.summary()

checkpoint = ModelCheckpoint('model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)
model.fit_generator(train_generator,
                    steps_per_epoch=X_train.shape[0] / batch_size,
                    epochs=10,
                    callbacks=[checkpoint],
                    validation_data=valid_generator,
                    validation_steps=X_valid.shape[0] / batch_size)


# model.save('model')
