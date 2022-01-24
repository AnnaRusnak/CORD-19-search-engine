# CORD-19-search-engine

This script downloads, organizes, and created a search engine for CORD-19, a dataset of 500,000 articles about COVID-19, SARS-CoV-2, and related coronaviruses. 

It analyzes the composition of CORD-19, which is language of the article, which institution the article is coming from, if it's a full article. Then, ten-topic LDA is ran on the titles and abstracts of articles written in English. 

BioSentVec, sent2vec trained on 30 millions of biomedical documents, is used to embed the dataset and the user query. Ball-tree kNN is used to find a hundred of nearest neighbors to the embedded query. 
