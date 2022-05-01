import pandas as pd
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# modify the transformer as needed
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
sim_lst = []
real_pair_sim_lst = []
for idx, s_ebd in enumerate(sentence_embeddings):
    K_largest = []
    sentence_id = others.loc[idx, 'sentence_id']
    reference_list = links[links['sentence_id'] == sentence_id]['paper_id'].tolist()

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
        sim_lst.append(similarity)
        if paper_idx == reference_list[0]:
            real_pair_sim_lst.append(similarity)

        if len(K_largest) < topK:
            K_largest.append(tuple((similarity, paper_row['paper_id'])))
            K_largest = sorted(K_largest, key=lambda x: x[0])
        else:
            if similarity > K_largest[0][0]:
                K_largest[0] = tuple((similarity, paper_row['paper_id']))
                K_largest = sorted(K_largest, key=lambda x: x[0])

    for top in K_largest:
        if top[1] in reference_list:
            corect += 1
            break
print("accuracy")
# 54.19% without conclusion
# 54.96% with conclusion
print(corect / len(others.index))

# figure
# sim_lst is a list of all similarities
# real_pair_sim_lst is a list of the similarities between sentences and corresponding positive abstracts
plt.hist(sim_lst, bins=50, density=True, label='all_similarity', alpha=0.5)
plt.hist(real_pair_sim_lst, bins=50, density=True, label='true_pair_similarity', alpha=0.5)
plt.legend()
plt.xlabel('similarity')
plt.ylabel('density')