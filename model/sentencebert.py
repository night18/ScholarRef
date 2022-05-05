import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model.max_seq_length = 500

option = 2

if option == 1:
    papers = pd.read_csv('../data_asbtract_conclusion/papers.csv')
elif option == 2:
    papers = pd.read_csv('../data/papers.csv')
elif option == 3:
    papers = pd.read_csv('../data_title/papers.csv')
abstracts = papers['abstract'].to_numpy()
print(len(abstracts))

#Sentences are encoded by calling model.encode()
embeddings = model.encode(abstracts)

papers['embedding'] = np.nan
papers['embedding'] = papers['embedding'].astype(object)

for idx, row in papers.iterrows():
    papers.at[idx, 'embedding'] = embeddings[idx]

if option == 1:
    papers.to_csv('abstract_conclusion_sbert.csv', index = False)
elif option == 2:
    papers.to_csv('abstract_sbert.csv', index = False)
elif option == 3:
    papers.to_csv('title_sbert.csv', index = False)