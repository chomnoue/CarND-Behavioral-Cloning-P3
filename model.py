import cv2
import numpy as np
import sklearn
import json
import os
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D
from keras.layers.convolutional import Convolution2D

def get_image(batch_sample, idx):
    name = 'data/IMG/'+batch_sample[idx].split('/')[-1]
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to BRG as it is the format use by PIL to read images in driver.py
    return image

def generator(samples, batch_size=32, correction=0.1):
    print("Generating data with correction: ", correction)
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        images = []
        angles = []

        def add_sample(image, angle):
            #add each image with a flipped version of it
            image_flipped = np.fliplr(image)
            angle_flipped = -angle
            images.extend((image, image_flipped))
            angles.extend((angle, angle_flipped))

        for idx, batch_sample in enumerate(samples):
            #use the three cameras             
            center_image = get_image(batch_sample,0)
            left_image = get_image(batch_sample,1)
            right_image = get_image(batch_sample,2)

            center_angle = float(batch_sample[3])
            left_angle = center_angle + correction
            right_angle = center_angle - correction
            add_sample(center_image, center_angle)
            add_sample(left_image, left_angle)
            add_sample(right_image, right_angle)

            #yield if we have batch_size images or we reach the end of the epoch
            if len(images) >= batch_size or idx==num_samples-1:
                X_train = np.array(images[:batch_size])
                y_train = np.array(angles[:batch_size])
                images = images[batch_size:]
                angles = angles[batch_size:]
                yield shuffle(X_train, y_train)

def get_model():
    print("using default achitecture")
    model = Sequential()
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(row, col, ch)))
    model.add(Lambda(lambda x: x/127.5 - 1.))
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
    return model

def get_nvdia_model():
    print("using nvidia achitecture")
    dropout=0.5
    model = Sequential()
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(row, col, ch)))
    model.add(Lambda(lambda x: x/127.5 - 1.))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    return model

if __name__ == "__main__":
    architecture = sys.argv[1]
    epochs = int(sys.argv[2])
    correction = float(sys.argv[3])
    print("architecture:",architecture, ", epochs:", epochs,", correction:", correction)

    samples = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32, correction = correction)
    validation_generator = generator(validation_samples, batch_size=32, correction = correction)

    # model architecture
    row, col, ch = 160, 320, 3

    model = get_nvdia_model() if architecture=="nvidia" else get_model()

    model.compile(optimizer="adam", loss="mse")

    #for each sample we have three camera images, each flipped. We thus have 3*2=6 images per sample
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6,
                         validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=epochs)

    print("Saving model weights and configuration file.")

    save_dir="./out"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_name = "/model_{}_{}_{}".format(architecture, epochs, str(correction).replace(".","_"))

    model.save(save_dir+model_name+".h5", True)
    with open(save_dir+model_name+'.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)