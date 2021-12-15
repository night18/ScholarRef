import pandas as pd
import re

chi2018 = pd.read_csv('chi2018.csv')
chi2019 = pd.read_csv('chi2019.csv')
chi2020 = pd.read_csv('chi2020.csv')

cscw2018 = pd.read_csv('cscw2018.csv')
cscw2019 = pd.read_csv('cscw2019.csv')
cscw2020 = pd.read_csv('cscw2020.csv')

papers = pd.concat([chi2018, chi2019, chi2020, cscw2018, cscw2019, cscw2020], ignore_index=True)
for index, row in papers.iterrows():
	papers.iloc[index]['title'] = re.sub(r"\s+", " ", row['title'])
	
papers['abstract'] = papers['abstract'].str.strip()

papers = papers[['title','abstract']]
papers = papers[papers['title'] != "Editors' Message"]
papers = papers[papers['abstract'] != "No abstract available."]


papers.to_csv('papers.csv', index=False)