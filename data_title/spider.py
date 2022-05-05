import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
import random
dir_loc = 'new_collection/'
# urls = ['https://st.sigchi.org/publications/toc/chi-2018.html']
urls = ['https://dl.acm.org/doi/proceedings/10.1145/3290605']
# urls_name = ['chi2018']
urls_name = ['chi2019']
for index, url in enumerate(urls):
	collect_list = []

	driver = webdriver.Chrome('/home/chun/chromedriver')
	driver.implicitly_wait(10)
	driver.get(url)
	# driver.implicitly_wait(10)

	python_button = driver.find_element_by_class_name('showAllProceedings')
	python_button.click()
	time.sleep(5)
	driver.execute_script("arguments[0].scrollIntoView()", python_button)
	time.sleep(5)

	html = BeautifulSoup(driver.page_source, 'html.parser')
	papers = html.select('.issue-item__title')
	# papers = html.select('.DLtitleLink')

	for paper in papers:
		title = paper.get_text()
		link = paper['href']
		# link = 'https://dl.acm.org' + link
		
		paper_data = requests.get(link)
		paper_data = BeautifulSoup(paper_data.text, 'html.parser')
		abstract = paper_data.find('div', class_= 'abstractInFull').get_text()

		collect_list.append({'title': title, 'link': link, 'abstract': abstract})
		sleeping_time = random.randrange(7, 20)
		time.sleep(sleeping_time)

	collect_data = pd.DataFrame(collect_list)

	collect_data.to_csv(dir_loc + urls_name[index] + '.csv')