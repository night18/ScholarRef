from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework import status
from random import sample
from decimal import Decimal
import csv
import json as JSON
import sys
from .model import model
import pandas as pd
import numpy as np
from scipy import spatial
from scholarly import scholarly, ProxyGenerator
import requests
from bs4 import BeautifulSoup

papers = pd.read_csv('./search/model/data/papers.csv')
pg = ProxyGenerator()
# success = pg.FreeProxies()
success = pg.ScraperAPI('e3a5c006022fe99fcdc0564e5b953a88')
scholarly.use_proxy(pg)

@api_view(['POST'])
def find_abstract(request):

	sentence = request.POST.get('sentence', None)

	try:
		Siamese_test = model.concept_encoder()
	except Exception as e:
		print(e)
		model.train_triplet(train_data['sentence'].to_numpy(), train_data['positive_abstract'].to_numpy(), train_data['negative_abstract'].to_numpy(), epochs=30)
		Siamese_test = model.concept_encoder()

	siamese = model.concept_encoder()
	embedding = siamese.predict([sentence])[0]
	# print(embedding)
	K_largest = []

	for idx, row in papers.iterrows():
		abs_embedding = row['embedding']
		abs_embedding = abs_embedding[1:]
		abs_embedding = abs_embedding[:-1]
		tmp = abs_embedding.split()
		abs_embedding = []
		for t in tmp:
			abs_embedding.append(float(t))
		abs_embedding = np.array(abs_embedding)

		result = 1 - spatial.distance.cosine([embedding], [abs_embedding])
		
		if len(K_largest) < 5:
			K_largest.append(tuple((result, row['paper_id'])))
			K_largest = sorted(K_largest, key=lambda x: x[0], reverse=True)
		else:
			if result > K_largest[4][0]:
				K_largest[4] = tuple((result, row['paper_id']))
				K_largest = sorted(K_largest, key=lambda x: x[0], reverse=True)

	K_title = []
	for k, v in K_largest:
		paper_title = papers[papers['paper_id'] == v]['title'].values[0]
		
		search_query = scholarly.search_pubs(paper_title)
		query_result = scholarly.fill(next(search_query))
		
		# print(type(query_result))
		# print(query_result)

		# url = "https://scholar.google.com" + query_result['url_scholarbib']
		# page = requests.get(url)
		# html = BeautifulSoup(page.text, 'html.parser')
		# print(html)
		# bib_url = html.find('a', class_= 'gs_citi')[0]['href']

		# page = requests.get(url)
		# html = BeautifulSoup(page.text, 'html.parser')
		# bibtex = html.get_text()
		bibtex = ""

		query_result['bibtex'] = bibtex

		K_title.append(query_result)

	json = {"titles" : K_title}
	return JsonResponse(json)