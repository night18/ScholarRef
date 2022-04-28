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
include_conclusion = False
topK = 10

if include_conclusion:
	paper = pd.read_csv('../data_asbtract_conclusion/papers.csv')
	sentence_root = pd.read_csv('../data_asbtract_conclusion/sentence_root.csv')
	links = pd.read_csv('../data_asbtract_conclusion/links.csv')
	embedding = pd.read_csv('./triplet_enbedding_papers.csv')
else:
	paper = pd.read_csv('../data/papers.csv')
	sentence_root = pd.read_csv('../data/sentence_root.csv')
	links = pd.read_csv('../data/links.csv')
	embedding = pd.read_csv('./triplet_enbedding_papers.csv')

paper = paper[['abstract','paper_id']]
paper = paper.rename(columns={'abstract':'content', 'paper_id': 'id'})
train_data = sentence_root[0:1000]
others = sentence_root[1000:]
others = others.reset_index(drop=True)
data = others[['sentence','sentence_id']]
data = data.rename(columns={'sentence':'content', 'sentence_id': 'id'})

Siamese_test = concept_encoder()

corect = 0
count = 0

for idx, row in data.iterrows():
	# print(row['id'])

	test_sentence = row['content']
	sentence_embedding = Siamese_test.predict([test_sentence])[0]

	K_largest = []

	for paper_idx, paper_row in embedding.iterrows():
		abs_embedding = paper_row['embedding']
		abs_embedding = abs_embedding[1:]
		abs_embedding = abs_embedding[:-1]
		tmp = abs_embedding.split()
		abs_embedding = []
		for t in tmp:
			abs_embedding.append(float(t))
		abs_embedding = np.array(abs_embedding)	
		# if paper_row['paper_id'] == 178:
		# 	print(sentence_embedding)
		# 	print(abs_embedding)
		# 	print(similarity)

		# Larger is greater
		similarity = 1 - spatial.distance.cosine([sentence_embedding], [abs_embedding])
		if len(K_largest) < topK:
			K_largest.append(tuple((similarity, paper_row['paper_id'])))
			K_largest = sorted(K_largest, key=lambda x: x[0])
		else:
			if similarity > K_largest[0][0]:
				K_largest[0] = tuple((similarity, paper_row['paper_id']))
				K_largest = sorted(K_largest, key=lambda x: x[0])

	
	reference_list = links[links['sentence_id'] == row['id']]['paper_id'].tolist()
	for top in K_largest:
		if top[1] in reference_list:
			corect += 1
			break
	# print(K_largest)

	# if count == 0:
	# 	break

print("accuracy")
print(corect / len(others.index))