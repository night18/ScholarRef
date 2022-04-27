import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import random
import re
import nltk.data
import numpy as np
from webdriver_manager.chrome import ChromeDriverManager
tokenizer = nltk.data.load('/home/chun/nltk_data/tokenizers/punkt/english.pickle')


urls = ['https://dl.acm.org/doi/proceedings/10.1145/3411764']
collect_list = []

for url in urls:
	driver = webdriver.Chrome(ChromeDriverManager().install())
	driver.implicitly_wait(10)
	driver.get(url)
	driver.implicitly_wait(5)

	python_button = driver.find_elements_by_css_selector('.login-link a')[0]
	python_button.click()
	time.sleep(5)
	# driver.execute_script("arguments[0].scrollIntoView()", python_button)

	python_button = driver.find_elements_by_css_selector('.login-form button')[0]
	python_button.click()
	time.sleep(5)

	python_username = driver.find_elements_by_css_selector('#username')[0]
	python_username.send_keys("chun-weichiang")
	python_password = driver.find_elements_by_css_selector('#pword')[0]
	python_password.send_keys("Ckbaby32!")
	python_button = driver.find_elements_by_css_selector('.login-page-wrapper input.blue')[0]
	python_button.click()
	time.sleep(10)


	session_buttons = driver.find_elements_by_css_selector('a.accordion-tabbed__control[aria-expanded="false"]')
	for button in session_buttons:
		button.click()
		time.sleep(3)

	

	# Access the paper
	html = BeautifulSoup(driver.page_source, 'html.parser')
	papers = html.select('.issue-item-container')
	sentence_id = 0
	for paper in papers:
		link = paper.find('a', attrs={"data-title":'HTML'})

		if link != None:
			link = link['href']
			if 'fullHtml' in link:
					link = 'https://dl.acm.org' + link

					driver.execute_script("window.open('" + link + "', 'new_window')")
					driver.switch_to.window(window_name=driver.window_handles[1])
					time.sleep(2)
					
					
					paper_html = BeautifulSoup(driver.page_source, 'html.parser')

					# references_html = paper_html.find('section.back-matter')
					# paper_html = BeautifulSoup(paper_data, 'html.parser')
					# references_html = BeautifulSoup(reference_data, 'html.parser')
					# text = html.select('section.body')
					paragraphs = paper_html.select('section.body p')
					references = paper_html.find('ul', class_= 'bibUl')

					for paragraph in paragraphs:
						text = paragraph.get_text()
						for sentence in tokenizer.tokenize(text):
							sentence = re.sub(' ',' ',sentence)
							
							cite_nums = re.findall(r'\[[0-9]+(?:\,\s[0-9]+)*\]', sentence)
							if len(cite_nums) > 0:
								try:
									sentence = re.sub(r' \[[0-9]+(?:\,\s[0-9]+)*\]','',sentence)

									cite_nums = np.concatenate(np.array([np.array(re.sub(r'\[|\]', '', x).split(', ')) for x in cite_nums])).flatten().astype(np.int)

									for cite_no in cite_nums:
										reference = references.find('li', attrs={"label": "["+ str(cite_no) +"]"}).get_text()
										# reference = references.find('li', attrs={"label": cite_no}).get_text()
										reference = re.sub(r'  Navigate to.+', '', reference)
										reference = re.sub(' ',' ',reference)
										
										collect_list.append({'sentence': sentence, 'reference': reference, 'sentence_id':sentence_id})
								except: 
									print(link)
									print(cite_nums)

							sentence_id += 1

					sleeping_time = random.randrange(7, 20)
					time.sleep(sleeping_time)

					driver.close()
					driver.switch_to.window(window_name=driver.window_handles[0])
					time.sleep(1)
				


collect_data = pd.DataFrame(collect_list)
collect_data.to_csv('sentence.csv', index=False)


