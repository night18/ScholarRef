import numpy as np
import pandas as pd

papers = pd.read_csv('papers.csv')
sentences = pd.read_csv('sentence.csv')


links = pd.DataFrame(columns={'sentence_id', 'paper_id'})

for p_idx, paper_row in papers.iterrows():
	paired = sentences['reference'].str.contains(paper_row['title'], na=True, regex=False)
	paired_sentences = sentences.loc[paired]

	for s_idx, sentence_row in paired_sentences.iterrows():
		links = links.append({'sentence_id': sentence_row['sentence_id'], 'paper_id': paper_row['paper_id']}, ignore_index=True)

links.to_csv('links.csv', index = False)
