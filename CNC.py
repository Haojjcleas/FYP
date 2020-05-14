
import keras
import h5py
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, MaxPool2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.applications import Xception
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score,
                             recall_score)
from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing import image
import pickle

from CNNclassifier import CNNclassifier
classifier = CNNclassifier()

classifier.train()
classifier.show()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
