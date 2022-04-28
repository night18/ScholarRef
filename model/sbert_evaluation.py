import pandas as pd
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

include_conclusion = True
topK = 5

if include_conclusion:
	sentence_root = pd.read_csv('../data_asbtract_conclusion/sentence_root.csv')
	links = pd.read_csv('../data_asbtract_conclusion/links.csv')
	abstract_sbert = pd.read_csv('./abstract_conclusion_sbert.csv')
else:
	sentence_root = pd.read_csv('../data/sentence_root.csv')
	links = pd.read_csv('../data/links.csv')
	abstract_sbert = pd.read_csv('./abstract_sbert.csv')


others = sentence_root[1000:]
others = others.reset_index(drop=True)

sentences = others['sentence'].to_numpy()
sentence_embeddings = model.encode(sentences)
corect = 0

for idx, s_ebd in enumerate(sentence_embeddings):
	K_largest = []

	for paper_idx, paper_row in abstract_sbert.iterrows():
		abs_embedding = paper_row['embedding']
		abs_embedding = abs_embedding[1:]
		abs_embedding = abs_embedding[:-1]
		tmp = abs_embedding.split()
		abs_embedding = []
		for t in tmp:
			abs_embedding.append(float(t))
		abs_embedding = np.array(abs_embedding)	

		similarity = 1 - spatial.distance.cosine([s_ebd], [abs_embedding])
		if len(K_largest) < topK:
			K_largest.append(tuple((similarity, paper_row['paper_id'])))
			K_largest = sorted(K_largest, key=lambda x: x[0])
		else:
			if similarity > K_largest[0][0]:
				K_largest[0] = tuple((similarity, paper_row['paper_id']))
				K_largest = sorted(K_largest, key=lambda x: x[0])

	sentence_id = others.loc[idx, 'sentence_id']
	reference_list = links[links['sentence_id'] == sentence_id]['paper_id'].tolist()

	for top in K_largest:
		if top[1] in reference_list:
			corect += 1
			break
print("accuracy")
# 54.19% without conclusion
# 54.96% with conclusion
print(corect / len(others.index))
