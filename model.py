#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:57:10 2017

@author: Edder

"""

'''Este scrip sigue lo planteado en el blog
"Building powerful image classification models using very little data"
que se puede encontrar en blog.keras.io.

Los datos de Dogs vs cats se reemplaza por el de plantas de porotos o habichuelas:
    0 es sin plaga y 1 es con plaga.

Para llegar a este punto hemos:
- creado el directorio vainitas
- creamos el directorio ./train y ./test dentro de /vainitas
- creamos ./1 y ./0 dentro de /train y dentro /test
- dentro de /train 7 tipo 0 y 24 tipo 1
- dentro de /test 3 tipo 0 y 5 tipo 1

Tenemos así solamente 31 casos para entrenamiento y 8 para validación

Esta es la estructura de los directorio, en resumen:
```
vainitas/
    train/
        0/
            0.jpg
            1.jpg
            ...
        1/
            0.jpg
            1.jpg
            ...
    test/
        0/
            0.jpg
            1.jpg
            ...
        1/
            0.jpg
            1.jpg
            ...
```
'''
#%%
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import backend as K
#%%
# dimensiones de las imágenes al procesarlas.
img_width, img_height = 150, 150
# dónde se va a guardar el modelo
top_model_weights_path = '/home/oscar/Documents/vainitas/vainitas_weigths.h5'
# dónde están los de entrenamiento
train_data_dir = '/home/oscar/Documents/vainitas/train'
# dónde están los de validación/test
validation_data_dir = '/home/oscar/Documents/vainitas/test'
## Esto toca cuadrarlo para que coincidan con los lotes
nb_train_samples = 248 
nb_validation_samples = 96
epochs = 10
batch_size = 8
#%%
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
#%%
def save_bottlebeck_features():
    datagen = ImageDataGenerator(
            rescale=1. / 255, # escalar los valores de pixel
            rotation_range = 180, #rotación permitida
            width_shift_range = 0.2, 
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            fill_mode = 'nearest' ,
            horizontal_flip=True)

    # Importamos la red VGG16 y sus pesos preentrenados
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('/home/oscar/Documents/vainitas/bottleneck_features_train',
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('/home/oscar/Documents/vainitas/bottleneck_features_validation',
            bottleneck_features_validation)
#%%
def train_top_model():
    train_data = np.load('/home/oscar/Documents/vainitas/bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load('/home/oscar/Documents/vainitas/bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
#%%
save_bottlebeck_features()
#%%
train_top_model()
#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%%
img=mpimg.imread('/home/oscar/Documents/billetes/reconocimiento_de_billetes/data/set_billetes/validation/bd50s/s.3.jpg')
img
#%%
imgplot = plt.imshow(img)
imgplot
#%%
from PIL import Image
img = Image.open('/home/oscar/Documents/billetes/reconocimiento_de_billetes/data/set_billetes/validation/bd50s/s.3.jpg')
img.thumbnail((150, 150), Image.ANTIALIAS) # resizes image in-place
imgplot = plt.imshow(img)
imgplot
#%%
def get_batches(dirname, gen=ImageDataGenerator(), shuffle=True, batch_size=batch_size, class_mode='categorical',
                target_size=(150,150)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
#%%
batches = get_batches('/home/oscar/Documents/billetes/reconocimiento_de_billetes/data/set_billetes/validation', batch_size=4)
#%%
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
#%%
imgs,labels = next(batches)
plots(imgs, titles=labels)