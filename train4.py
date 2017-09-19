import os
from functools import partial
from operator import itemgetter

import cv2
import numpy
import pandas
from keras import optimizers, losses
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Input, Lambda, activations, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers.convolutional import Cropping2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

batch_size = 32
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
uniform = numpy.random.uniform


def combine_datasets(*csv_files):
    df = pandas.read_csv(csv_files[0][0], header=None, names=columns)
    df.center = df.center.apply(lambda x: os.path.join(csv_files[0][1], x))
    df.left = df.left.apply(lambda x: os.path.join(csv_files[0][1], x))
    df.right = df.right.apply(lambda x: os.path.join(csv_files[0][1], x))

    for i in range(1, len(csv_files)):
        df_new = pandas.read_csv(csv_files[i][0], header=None, names=columns)
        df_new.center = df_new.center.apply(lambda x: os.path.join(csv_files[i][1], x))
        df_new.left = df_new.left.apply(lambda x: os.path.join(csv_files[i][1], x))
        df_new.right = df_new.right.apply(lambda x: os.path.join(csv_files[i][1], x))
        df = pandas.concat([df, df_new])

    X = df.center.values
    y = df.steering.values

    X = numpy.hstack((X, df.left))
    y = numpy.hstack((y, df.steering + 0.25))

    X = numpy.hstack((X, df.right))
    y = numpy.hstack((y, df.steering - 0.25))

    return X, y


def upsample(X, y, bins):
    indices_of_sharp_angles = numpy.where(numpy.abs(y) > 0.40)[0]
    sharp_y = y[indices_of_sharp_angles]
    digitized = numpy.digitize(sharp_y, bins)
    classes, counts = numpy.unique(digitized, return_counts=True)
    print(classes, counts)
    probabilities = counts / counts.sum()
    print(probabilities)
    inverse = numpy.power(probabilities, -0.5)
    exponents = numpy.exp(inverse)
    new_probabilities = exponents / exponents.sum()
    print(new_probabilities)
    per_instance = new_probabilities / counts

    sample_probabilities = numpy.zeros(shape=len(sharp_y))
    for cls, probability in zip(classes, per_instance):
        indices = numpy.where(digitized == cls)
        sample_probabilities[indices] = probability

    indices_to_upsample = numpy.random.choice(indices_of_sharp_angles, 15000, p=sample_probabilities, replace=False)
    X = numpy.hstack((X, X[indices_to_upsample]))
    y = numpy.hstack((y, y[indices_to_upsample]))
    digitized = numpy.digitize(y, bins)
    return X, y, digitized


def translate(image, angle, max_horizontal_shift, correction=0.04):
    horizontal_shift = max_horizontal_shift * uniform(-1, 1)
    correction = numpy.abs(horizontal_shift / 100)
    new_angle = angle + correction * horizontal_shift / max_horizontal_shift
    vertical_shift = 20 * uniform(-1, 1)
    translation_matrix = numpy.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])
    new_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return new_image, new_angle


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
    hlsed[:, :, 1][to_be_shadowed] = hlsed[:, :, 1][to_be_shadowed] * numpy.random.uniform(0.4, 0.6)
    return cv2.cvtColor(hlsed, cv2.COLOR_HLS2RGB)


def if_yes():
    return uniform() > 0.5


def generate_from(filenames, steering, batch_size=32):
    half_batch_size = int(batch_size / 2)
    while True:
        filenames, steering = shuffle(filenames, steering)
        for offset in range(0, len(filenames), half_batch_size):
            batch_filenames, batch_steering = filenames[offset:offset + half_batch_size], \
                                              steering[offset:offset + half_batch_size]

            images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in batch_filenames]
            # image2angle = [translate(img, angle, uniform(20, 60)) for img, angle in zip(images, batch_steering)]
            image2angle = [translate(img, angle, 30) for img, angle in zip(images, batch_steering)]
            images = list(map(itemgetter(0), image2angle))
            batch_steering = numpy.array(list(map(itemgetter(1), image2angle)))
            images = [shadow(image) if if_yes() else image for image in images]
            images = numpy.vstack((images, numpy.array(list(map(partial(cv2.flip, flipCode=1), images)))))
            batch_steering = numpy.hstack((batch_steering, -batch_steering))

            # images = numpy.array(list(map(
            #     partial(cv2.resize, dsize=(int(images[0].shape[1] / 2), int(images[0].shape[0] / 2))), images)))

            yield shuffle(images, batch_steering)


pattern = '{}/driving_log.csv'

directories = ['track2', 'track3', 'track4', 'track5', 'backup3']
csv_files_images = [(pattern.format(image_folder), image_folder) for image_folder in directories]

X, y = combine_datasets(*csv_files_images)

angle_grid = [-1, -0.8, -0.6, -0.2, 0.2, 0.6, 0.8, 1]
bins = numpy.array(angle_grid)
digitized = numpy.digitize(y, bins)
# X, y, digitized = upsample(X, y, bins)
print(X.shape, y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=digitized, train_size=0.8)

train_generator = generate_from(X_train, y_train, batch_size)
valid_generator = generate_from(X_valid, y_valid, batch_size)

# inputs = Input(shape=(80, 160, 3))
inputs = Input(shape=(160, 320, 3))
output = Lambda(lambda x: (x - 127) / 128)(inputs)
cropped = Cropping2D(cropping=((20, 20), (0, 0)))(output)
# output = Conv2D(3, (1, 1), padding='same')(output)

output = Conv2D(32, (5, 5), activation=activations.relu, padding='same')(output)
output = MaxPooling2D((2, 2))(output)

output = Conv2D(48, (5, 5), activation=activations.relu, padding='same')(output)
output = MaxPooling2D((2, 2))(output)
print(output.shape)

output = Conv2D(64, (5, 5), activation=activations.relu)(output)
output = MaxPooling2D((2, 2))(output)

output = Conv2D(64, (5, 5), activation=activations.relu)(output)
output = MaxPooling2D((2, 2))(output)

output = Conv2D(64, (5, 5), activation=activations.relu)(output)
output = MaxPooling2D((2, 2))(output)

flattened = Flatten()(output)
dense1 = Dense(100, activation=activations.relu)(flattened)
dense2 = Dense(100, activation=activations.relu)(dense1)
dense2 = Dropout(0.8)(dense2)
output = Dense(1)(dense2)
model = Model(inputs, output)

model.compile(optimizer=optimizers.adam(), loss=losses.mse)
model.summary()

checkpoint = ModelCheckpoint('model7', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)
model.fit_generator(train_generator,
                    steps_per_epoch=X_train.shape[0] / batch_size,
                    epochs=5,
                    callbacks=[checkpoint],
                    validation_data=valid_generator,
                    validation_steps=X_valid.shape[0] / batch_size)
