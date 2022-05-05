import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
option = 1
include_conclusion = True
topK = 10

if option == 1:
	sentence_root = pd.read_csv('../data_asbtract_conclusion/sentence_root.csv')
	links = pd.read_csv('../data_asbtract_conclusion/links.csv')
	paper = pd.read_csv('../data_asbtract_conclusion/papers.csv')
elif option == 2:
	sentence_root = pd.read_csv('../data/sentence_root.csv')
	links = pd.read_csv('../data/links.csv')
	paper = pd.read_csv('../data/papers.csv')
	
elif option == 3:
	sentence_root = pd.read_csv('../data_title/sentence_root.csv')
	links = pd.read_csv('../data_title/links.csv')
	paper = pd.read_csv('../data_title/papers.csv')
	


paper = paper[['abstract','paper_id']]
paper = paper.rename(columns={'abstract':'content', 'paper_id': 'id'})
others = sentence_root
others = others.reset_index(drop=True)
data = others[['sentence','sentence_id']]
data = data.rename(columns={'sentence':'content', 'sentence_id': 'id'})
print(len(data))
# combine sentences with paper together
combined = pd.concat([data, paper], ignore_index=True, sort=False)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(combined['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

corect = 0
# Top 655 are sentences
for idx, row in data.iterrows():
	is_found = False
	simi = cosine_sim[idx][1655:].copy()
	top5 = []

	for simi_idx in range(len(simi)):
		if len(top5) <= 5:
			top5.append((simi[simi_idx], simi_idx))
			top5 = sorted(top5, key=lambda x: x[0])
		else:
			if simi[simi_idx] > top5[0][0]:
				top5[0] = (simi[simi_idx], simi_idx)
				top5 = sorted(top5, key=lambda x: x[0])
	
	reference_list = links[links['sentence_id'] == row['id']]['paper_id'].tolist()
	for top in top5:
		if top[1] in reference_list:
			corect += 1
			break

	data.loc[idx, 'is_found'] = is_found

print("accuracy")
print(corect / len(others.index))
data.to_csv('tfidf_found.csv', index=False)
