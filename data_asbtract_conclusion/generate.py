import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Load data
links = pd.read_csv('links.csv')
papers = pd.read_csv('papers.csv')
sentences = pd.read_csv('sentence.csv')

# Creat Training dataset
sentence_root_data = pd.DataFrame(columns={'sentence', 'positive_abstract', 'negative_abstract'})

for _, link in links.iterrows():
	sentence = sentences.loc[link['sentence_id']]['sentence']
	positive = papers.loc[link['paper_id']]['abstract']
	negative_id = papers[papers['paper_id'] != link['paper_id'] ].sample(n=1).first_valid_index()
	negative = papers.loc[negative_id]['abstract']

	sentence_root_data = sentence_root_data.append({
		'sentence':sentence,
		'positive_abstract': positive,
		'negative_abstract': negative
		}, ignore_index=True)
	
sentence_root_data = shuffle(sentence_root_data)
sentence_root_data.to_csv('sentence_root.csv', index = False)