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

Siamese_test = concept_encoder()
papers = pd.read_csv('../data/papers.csv')
test = Siamese_test.predict(papers['abstract'].to_numpy())

papers['embedding'] = np.nan
papers['embedding'] = papers['embedding'].astype(object)

for idx, row in papers.iterrows():
	papers.at[idx, 'embedding'] = test[idx]


papers.to_csv('triplet_enbedding_papers.csv', index = False)
