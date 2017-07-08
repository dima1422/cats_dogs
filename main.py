from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt


inc_model  = InceptionV3 (include_top= False,
                          weights = "imagenet",
                          imput_shape = (3, 150, 150))

gen = ImageDataGenerator(rescale= 1./255)

train_gen = gen.flow_from_directory("data/train",
                                    target_size= (150,150),
                                    batch_size= 32,
                                    class_mode= None,
                                    shuffle= false)

val_gen = gen.flow_from_directory("data/validation",
                                  target_size = (150,150),
                                  batch_size= 32,
                                  class_mode= None,
                                  shuffle= False)

features_train= inc_model.predict_generator(train_gen, 2000)

np.save(open("features/features_train.npy", "wb"),features_train)

features_validation=inc_model.predict_generator(val_gen, 2000)

np.save(open("features/features_validation.npy", "wb"),features_validation)
