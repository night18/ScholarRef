import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

include_conclusion = True
if include_conclusion:
    papers = pd.read_csv('../data_asbtract_conclusion/papers.csv')
else:
    papers = pd.read_csv('../data/papers.csv')
abstracts = papers['abstract'].to_numpy()
print(len(abstracts))

#Sentences are encoded by calling model.encode()
embeddings = model.encode(abstracts)

papers['embedding'] = np.nan
papers['embedding'] = papers['embedding'].astype(object)

for idx, row in papers.iterrows():
    papers.at[idx, 'embedding'] = embeddings[idx]

if include_conclusion:
    papers.to_csv('abstract_conclusion_sbert.csv', index = False)
else:
    papers.to_csv('abstract_sbert.csv', index = False)