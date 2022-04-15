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
from scipy import spatial
# from transformers import BertTokenizer, TFBertForSequenceClassification
# from transformers import InputExample, InputFeatures

# model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# model.summary()

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_sentence_embeding(sentences):
	preprocessed_text = bert_preprocess(sentences)
	return bert_encoder(preprocessed_text)['pooled_output']

# ===============================Example===============================
# print(get_sentence_embeding(['hello world!!', 'Hello world']))
# tf.Tensor(
# [[-0.9013801  -0.42470214 -0.8700524  ... -0.73245317 -0.691746
#    0.9264366 ]
#  [-0.9058484  -0.30958173 -0.615244   ... -0.30049452 -0.6393687
#    0.9164876 ]], shape=(2, 768), dtype=float32)
# ===============================Example===============================

def sentence_model():
	inputs = Input(shape=(768,))
	x = Dense(16, name='dense_32')(inputs)
	# x = Dropout(0.1, name='dropout_32')(x)
	x = Activation( 'relu', name='relu_32' )(x)
	x = Dense(100, name='dense_16')(x)
	x = Activation( 'relu', name='relu_16' )(x)

	model = Model(
		inputs = inputs,
		outputs = x
	)
	return model

def triplet():
	input_base = Input(shape=(), dtype=tf.string, name='base_text')
	input_pos = Input(shape=(), dtype=tf.string, name='pos_text')
	input_neg = Input(shape=(), dtype=tf.string, name='neg_text')

	basemodel = sentence_model()
	encode_base = basemodel(get_sentence_embeding(input_base))
	encode_pos = basemodel(get_sentence_embeding(input_pos))
	encode_neg = basemodel(get_sentence_embeding(input_neg))

	merged_vector = Concatenate(axis=-1)([encode_base, encode_pos, encode_neg])
	model = Model(
			inputs = [input_base, input_pos, input_neg],
			outputs = merged_vector
		)
	model.summary()
	return model

def siamese_loss(yTrue, yPred):
	# yTrue is label, and yPred is distance
	return K.mean( (1-yTrue) * K.square(yPred) + yTrue * K.square(K.maximum(1 - yPred, 0)))


def triplet_loss(y_true, y_pred, cosine=False, alpha=0.2):
	embedding_size = K.int_shape(y_pred)[-1] // 3
	ind = int(embedding_size * 2)
	a_pred = y_pred[:, :embedding_size]
	p_pred = y_pred[:, embedding_size:ind]
	n_pred = y_pred[:, ind:]

	if cosine:
		pos_distance = 1 - K.sum((a_pred * p_pred), axis = -1)
		neg_distance = 1 - K.sum((a_pred * n_pred), axis = -1)
	else:
		pos_distance = K.sqrt(K.sum(K.square(a_pred - p_pred), axis=-1))
		neg_distance = K.sqrt(K.sum(K.square(a_pred - n_pred), axis=-1))
	loss = K.maximum(0.0, pos_distance - neg_distance + alpha)
	return loss


def train_triplet(base, pos, neg, epochs=10, learning_rate=0.001):
	checkpoint_path = "checkpoint/triplet.hdf5"
	h5_storage_path = "h5/triplet.h5"
	hist_storage_path = "history/triplet"
	
	model = triplet()

	model.compile(
		loss = triplet_loss,
		optimizer = Adam(learning_rate = learning_rate)
	)

	checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
	callbacks_list = [checkpoint]

	hist = model.fit(
		[base, pos, neg],
		np.zeros(base.shape[0]),
		epochs = epochs,
		batch_size = 16,
		validation_split = 0.3,
		callbacks = callbacks_list,
		verbose = 1
	)

	model.layers[5].save_weights(
		h5_storage_path,
		overwrite = True
	)

	model.get_weights()
	with open(hist_storage_path, 'wb') as file_hist:
		pickle.dump(hist.history, file_hist)

	print("Successfully save the model at " + h5_storage_path)

	return model

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shapes[0], shapes[2])

def test_triplet(learning_rate=0.001):
	h5_storage_path = "h5/triplet.h5"
	input_base = Input(shape=(), dtype=tf.string, name='base_text')
	input_pair = Input(shape=(), dtype=tf.string, name='pair_text')

	basemodel = sentence_model()
	# For testing
	initial_weights = [layer.get_weights() for layer in basemodel.layers]


	basemodel.load_weights(
		h5_storage_path,
		by_name = True
	)

	basemodel.trainable = False

	encode_base = basemodel(get_sentence_embeding(input_base))
	encode_pair = basemodel(get_sentence_embeding(input_pair))

	L2_layer = Lambda( lambda tensor: K.sqrt(K.sum(K.square(tensor[0]-tensor[1]), axis=1, keepdims=True )),  output_shape=eucl_dist_output_shape)

	L2_distance = L2_layer([encode_base, encode_pair])
	output = Activation('sigmoid')(L2_distance)

	model = Model(
			inputs = [input_base, input_pair],
			outputs = L2_distance
		)

	model.compile(
		loss = siamese_loss,
		optimizer = SGD(learning_rate = learning_rate, momentum=0.9),
		metrics = ['acc']
	)
	return model

def concept_encoder(learning_rate=0.001):
	h5_storage_path = "h5/triplet.h5"

	input_base = Input(shape=(), dtype=tf.string, name='base_text')

	basemodel = sentence_model()
	# For testing
	initial_weights = [layer.get_weights() for layer in basemodel.layers]


	basemodel.load_weights(
		h5_storage_path,
		by_name = True
	)

	basemodel.trainable = False

	encode_base = basemodel(get_sentence_embeding(input_base))

	model = Model(
			inputs = input_base,
			outputs = encode_base
		)

	model.compile(
		loss = 'mean_squared_error',
		optimizer = SGD(learning_rate = learning_rate, momentum=0.9),
		metrics = ['acc']
	)
	return model
	

data = pd.read_csv('../data/sentence_root.csv')
train_data = data[0:1000]
others = data[1000:]

test_data = pd.DataFrame(columns={'base', 'pair', 'label'})
for i, row in others.iterrows():
	test_data = test_data.append({
		'base':row['sentence'],
		'pair': row['positive_abstract'],
		'label': 0.0
		}, ignore_index=True)

	test_data = test_data.append({
		'base':row['sentence'],
		'pair': row['negative_abstract'],
		'label': 1.0
		}, ignore_index=True)

try:
	# Siamese_test = test_triplet()
	Siamese_test = concept_encoder()
except Exception as e:
	print(e)
	train_triplet(train_data['sentence'].to_numpy(), train_data['positive_abstract'].to_numpy(), train_data['negative_abstract'].to_numpy(), epochs=30)
	Siamese_test = concept_encoder()
	# Siamese_test = test_triplet()

	
# Siamese_test = concept_encoder()
# predictions = Siamese_test.predict(["Model transparency might hinder users' ability to recognize the model’s serious error."])
# print(predictions)

# abs_predictions = Siamese_test.predict(["With machine learning models being increasingly used to aid decision making even in high-stakes domains, there has been a growing interest in developing interpretable models. Although many supposedly interpretable models have been proposed, there have been relatively few experimental studies investigating whether these models achieve their intended effects, such as making people more closely follow a model’s predictions when it is beneficial for them to do so or enabling them to detect when a model has made a mistake. We present a sequence of pre-registered experiments (N = 3, 800) in which we showed participants functionally identical models that varied only in two factors commonly thought to make machine learning models more or less interpretable: the number of features and the transparency of the model (i.e., whether the model internals are clear or black box). Predictably, participants who saw a clear model with few features could better simulate the model’s predictions. However, we did not find that participants more closely followed its predictions. Furthermore, showing participants a clear model meant that they were less able to detect and correct for the model’s sizable mistakes, seemingly due to information overload. These counterintuitive findings emphasize the importance of testing over intuition when developing interpretable models."])
# print(abs_predictions)

# result = 1 - spatial.distance.cosine(predictions, abs_predictions)
# print(result)

# Test accuracy 66.6%
# results = Siamese_test.evaluate(
# 	[test_data['base'].to_numpy(),test_data['pair'].to_numpy()],
# 	test_data['label'].to_numpy())
# print(results)

# papers = pd.read_csv('../data/papers.csv')
# test = Siamese_test.predict(papers['abstract'].to_numpy())

# papers['embedding'] = np.nan
# papers['embedding'] = papers['embedding'].astype(object)

# for idx, row in papers.iterrows():
# 	papers.at[idx, 'embedding'] = test[idx]


# papers.to_csv('../data/papers.csv', index = False)

# Test the file
papers = pd.read_csv('data/papers.csv')
papers['embedding'] = papers['embedding'].astype(object)


test_sentence = "Model transparency might hinder users' ability to recognize the model’s serious error."
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

	result = 1 - spatial.distance.cosine([embedding], [abs_embedding])
	
	if len(K_largest) < 5:
		K_largest.append(tuple((result, row['paper_id'])))
		K_largest = sorted(K_largest, key=lambda x: x[0], reverse=True)
	else:
		if result > K_largest[4][0]:
			K_largest[4] = tuple((result, row['paper_id']))
			K_largest = sorted(K_largest, key=lambda x: x[0], reverse=True)
K_title = []
for k, v in K_largest:
	paper_title = papers[papers['paper_id'] == v]['title'].values[0]	
	K_title.append(paper_title)

print("Input: ", test_sentence)
print("OUtput: ")
for title in K_title:
	print(title)