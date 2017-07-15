import os
import csv

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

if __name__ == "__main__":
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # model architecture
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
    from keras.layers.convolutional import Convolution2D
    ch, row, col = 3, 160, 320 

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                input_shape=(ch, row, col),
                output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
                         validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

    print("Saving model weights and configuration file.")

    save_dir="./out"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save_weights(save_dir+"/model.h5", True)
    with open(save_dir+'/model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)