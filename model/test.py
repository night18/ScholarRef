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
papers = pd.read_csv('triplet_enbedding_papers.csv')

test_sentence = "Prior work has identified, either explicitly or implicitly via roles, that expertise is a defining attribute of interpretable ML stakeholders."
embedding = Siamese_test.predict([test_sentence])[0]
K_largest = []

for idx, row in papers.iterrows():
	abs_embedding = row['embedding']
	abs_embedding = abs_embedding[1:]
	abs_embedding = abs_embedding[:-1]
	tmp = abs_embedding.split()
	abs_embedding = []
	for t in tmp:
		abs_embedding.append(float(t))
	abs_embedding = np.array(abs_embedding)

	# Smaller is better: Smaller represent the two concept are more similar
	result = spatial.distance.cosine([embedding], [abs_embedding])
	
	if len(K_largest) < 5:
		K_largest.append(tuple((result, row['paper_id'])))
		K_largest = sorted(K_largest, key=lambda x: x[0], reverse=True)
	else:
		if result < K_largest[0][0]:
			K_largest[0] = tuple((result, row['paper_id']))
			K_largest = sorted(K_largest, key=lambda x: x[0], reverse=True)

K_largest = sorted(K_largest, key=lambda x: x[0])
K_title = []
for k, v in K_largest:
	paper_title = papers[papers['paper_id'] == v]['title'].values[0]
	K_title.append(paper_title)

print("Input: ", test_sentence)
print("OUtput: ")
# for title in K_title:
# 	print(title)
print(K_largest)