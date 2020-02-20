import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import requests
import sys
import random

from PIL import Image
from io import BytesIO

from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

print()
print()
print()


def read_data():
    path = './dataset/' # Path to directory which contains classes
    classes = os.listdir(path) # List of all classes
    print(f'Total number of categories: {len(classes)}')

    # A dictionary which contains class and number of images in that class
    counts = {}
    for c in classes:
        counts[c] = len(os.listdir(os.path.join(path, c)))

    print(f'Total number of images in dataset: {sum(list(counts.values()))}')

    train_x = []
    train_y = []
    liste = ['Mewtwo', 'Pikachu', 'Charmander']

    for c in classes:

        if c in liste :

            dir_path = os.path.join(path, c)

            for img in os.listdir(dir_path):

                image = cv.imread(os.path.join(dir_path, img))
                label = liste.index(c)

                try:
                    resize = cv.resize(image, (96,96))
                    train_x.append(resize)
                    train_y.append(label)
                except:
                    print(f'cant read file :{os.path.join(dir_path, img)}')

    train_x =  np.array(train_x).reshape(-1, 96, 96, 3)
    train_x = train_x/255.0
    train_y = to_categorical(train_y, num_classes = len(liste))

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.15, stratify=train_y, shuffle = True, random_state = 666)

    return [train_x, train_y], [test_x, test_y], liste


def train_data(train, test, imbalanced) :

    datagen = ImageDataGenerator(rotation_range = 45, # Degree range for random rotations
                            zoom_range = 0.2, # Range for random zoom 
                            horizontal_flip = True, # Randomly flip inputs horizontally
                            width_shift_range = 0.15, # Range for horizontal shift 
                            height_shift_range = 0.15, # Range for vertical shift 
                            shear_range = 0.2) # Shear Intensity

    datagen.fit(train[0])

    model = Sequential()
    model.add(Conv2D(32, 3, padding = 'same', activation = 'relu', input_shape =(96, 96, 3), kernel_initializer = 'he_normal'))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', activation = 'relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(len(imbalanced), activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    checkpoint = ModelCheckpoint('best_model.hdf5.png', verbose = 1, monitor = 'val_accuracy', save_best_only = True)

    history = model.fit_generator(datagen.flow(train[0], train[1], batch_size = 32), epochs = 4, validation_data = [test[0], test[1]],
                             steps_per_epoch=len(train[0]) // 32, callbacks = [checkpoint])

    return model


def try_predict(model, imbalanced):
    mewtwo = ['https://cdn.bulbagarden.net/upload/thumb/7/78/150Mewtwo.png/250px-150Mewtwo.png',
         'https://cdn.vox-cdn.com/thumbor/sZPPvUyKyF97UEU-nNtVnC3LpF8=/0x0:1750x941/1200x800/filters:focal(878x316:1158x596)/cdn.vox-cdn.com/uploads/chorus_image/image/63823444/original.0.jpg',
         'https://images-na.ssl-images-amazon.com/images/I/61j5ozFjJ0L._SL1024_.jpg']

    pikachu = ['https://lh3.googleusercontent.com/proxy/DrjDlKlu9YonKbj3iNCJNJ3DGqzy9GjeXXSUv-TcVV4UN9PMCAM5yIkGLPG7wYo3UeA4sq5OmUWM8M6K5hy2KOAhf8SOL3zPH3axb2Xo3HX2XTU8M2xW4X6lVg=w720-h405-rw',
          'https://giantbomb1.cbsistatic.com/uploads/scale_medium/0/6087/2437349-pikachu.png',
          'https://johnlewis.scene7.com/is/image/JohnLewis/237525467']

    rondoudou = ['https://www.pokepedia.fr/images/8/89/Salam%C3%A8che-RFVF.png']

    test_df = [mewtwo, pikachu, rondoudou]

    val_x = []
    val_y = []

    for i, urls in enumerate(test_df):
        for url in urls:        
            r = requests.get(url, stream = True).raw
            image = np.asarray(bytearray(r.read()), dtype="uint8")
            image = cv.imdecode(image, cv.IMREAD_COLOR)
            val_x.append(image)
            val_y.append(i)

    rows = len(test_df)
    cols = len(mewtwo)

    fig = plt.figure(figsize = (25, 25))

    for i, j in enumerate(zip(val_x, val_y)): # i - for subplots
        orig = j[0] # Original, not resized image
        label = j[1] # Label for that image

        image = cv.resize(orig, (96, 96)) # Resizing image to (96, 96)
        image = image.reshape(-1, 96, 96, 3) / 255.0 # Reshape and scale resized image
        preds = model.predict(image) # Predicting image
        pred_class = np.argmax(preds) # Defining predicted class

        true_label = f'True class: {imbalanced[label]}'
        pred_label = f'Predicted: {imbalanced[pred_class]} {round(preds[0][pred_class] * 100, 2)}%'

        fig.add_subplot(rows, cols, i+1)
        plt.imshow(orig[:, :, ::-1])
        plt.title(f'{true_label}\n{pred_label}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

train, test, liste = read_data()
model = train_data(train, test, liste)
try_predict(model, liste)