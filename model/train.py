import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.layers import Input, Conv2D, Dropout, Activation, Concatenate, concatenate, MaxPool2D, Lambda, Flatten, Dense,  GlobalAveragePooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K
import pickle
from model import *
from scipy import spatial

data = pd.read_csv('../data/sentence_root.csv')
train_data = data
# train_data = data[0:1000]

train_triplet(train_data['sentence'].to_numpy(), train_data['positive_abstract'].to_numpy(), train_data['negative_abstract'].to_numpy(), epochs=30)
