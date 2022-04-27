import pandas as pd
import numpy as np

sentences = pd.read_csv('sentence.csv')
sentences['sentence'] = sentences['sentence'].str.lower()

current_sentence = ""
sentence_idx = 0
for idx, sentence in sentences.iterrows():
	if sentence['sentence'] != current_sentence:
		current_sentence = sentence['sentence']
		sentence_idx += 1

	sentences.loc[idx, "sentence_id"] = sentence_idx


sentences.to_csv('new_sentence.csv', index=False)
