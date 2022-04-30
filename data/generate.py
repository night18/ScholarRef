import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Load data
links = pd.read_csv('links.csv')
papers = pd.read_csv('papers.csv')
sentences = pd.read_csv('sentence.csv')
sample_size = 10000

# Creat Training dataset
sentence_root_data = pd.DataFrame(columns={'sentence', 'positive_abstract', 'negative_abstract'})
sample_pool = links.sample(n = sample_size, replace=True, random_state=1)


for _, link in sample_pool.iterrows():
	sentence_row_index = sentences[sentences['sentence_id'] == link['sentence_id']].first_valid_index()
	sentence = sentences.loc[sentence_row_index]['sentence']
	
	paper_row_index = papers[papers['paper_id'] == link['paper_id']].first_valid_index()
	positive = papers.loc[paper_row_index]['abstract']
	negative_id = papers[papers['paper_id'] != link['paper_id'] ].sample(n=1).first_valid_index()
	negative = papers.loc[negative_id]['abstract']

	sentence_root_data = sentence_root_data.append({
		'sentence':sentence,
		'positive_abstract': positive,
		'negative_abstract': negative,
		'sentence_id': link['sentence_id'],
		'positive_id': link['paper_id'],
		'negative_id': negative_id
		}, ignore_index=True)
	
sentence_root_data = shuffle(sentence_root_data)
sentence_root_data.to_csv('sentence_root.csv', index = False)