import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import spatial
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model.max_seq_length = 500
topK = 10

abstract_sbert = pd.read_csv('./abstract_sbert.csv')
sentence_root = pd.read_csv('../data/sentence_root.csv')

def text2array(text):
	
	text = text[1:]
	text = text[:-1]
	tmp = text.split()
	abs_embedding = []
	for t in tmp:
		abs_embedding.append(float(t))
	abs_embedding = np.array(abs_embedding)	
	return abs_embedding

store = pd.DataFrame(columns=['similarity','label', 'dataset'])
for option in range(1,4):
	if option == 1:
		sentence_root = pd.read_csv('../data_asbtract_conclusion/sentence_root.csv')
		title = 'title + abstract + conclusion'
		abstract_sbert = pd.read_csv('./abstract_conclusion_sbert.csv')
	elif option == 2:
		sentence_root = pd.read_csv('../data/sentence_root.csv')
		abstract_sbert = pd.read_csv('./abstract_sbert.csv')
		title = 'title + abstract'
	elif option == 3:
		sentence_root = pd.read_csv('../data_title/sentence_root.csv')
		abstract_sbert = pd.read_csv('./title_sbert.csv')
		title = 'title'

	sentences = sentence_root['sentence'].to_numpy()
	sentence_embeddings = model.encode(sentences)

	for idx, s_ebd in enumerate(sentence_embeddings):
		current_sentence_row = sentence_root.iloc[idx]

		positive_embedding = text2array(abstract_sbert[abstract_sbert['paper_id'] == current_sentence_row['positive_id']].iloc[0]['embedding'])
		positive_similarity =  1 - spatial.distance.cosine([s_ebd], [positive_embedding])

		negative_embedding = text2array(abstract_sbert[abstract_sbert['paper_id'] == current_sentence_row['negative_id']].iloc[0]['embedding'])
		negative_similarity =  1 - spatial.distance.cosine([s_ebd], [negative_embedding])

		store = pd.concat([store, pd.DataFrame({'similarity': [positive_similarity], 'label': ['matched'], 'dataset': [title]})], ignore_index = True)
		store = pd.concat([store, pd.DataFrame({'similarity': [negative_similarity], 'label': ['not matched'], 'dataset': [title]})], ignore_index = True)


sns.displot(store, x='similarity', hue='label', col='dataset', fill=True)

plt.show()


