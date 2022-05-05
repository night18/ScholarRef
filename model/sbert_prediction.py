import pandas as pd
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model.max_seq_length = 500
topK = 10

abstract_sbert = pd.read_csv('./abstract_sbert.csv')


test_sentence = "built upon the flexible composition of visual encoding for expressive visualizations"


s_ebd = model.encode([test_sentence])[0]

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

K_largest = sorted(K_largest, key=lambda x: x[0], reverse=True)
print("input sentence: ", test_sentence)
print("")
print("The most related papers:")
for pred in K_largest:
	print(abstract_sbert[abstract_sbert['paper_id'] == pred[1]].iloc[0]['title'])
