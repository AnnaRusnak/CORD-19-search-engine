# This script was written by Anna Rusnak, Ruiqi Hu, and Dongjun Shin
# It extracts and organizes the metadata about CORD-19 dataset and constructs a search engine based on 
#sentence embedding and kNN algorithms


#import modules for getData and extract
import os
import urllib.request
import tarfile
import json
from langdetect import detect
import pandas as pd

#import modules for organize
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#import modules for retrieve
from nltk.tokenize import sent_tokenize
import sent2vec
import numpy as np
from sklearn.neighbors import NearestNeighbors


#Download and unpack the collection
def getData():
    urls = ['https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/comm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/noncomm_use_subset.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/custom_license.tar.gz', 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/biorxiv_medrxiv.tar.gz']

    # Create data directory
    try:
        os.mkdir('./data')
        print('Directory created')
    except FileExistsError:
        print('Directory already exists')

    #Download all files
    for i in range(len(urls)):
        urllib.request.urlretrieve(urls[i], './data/file'+str(i)+'.tar.gz')
        print('Downloaded file '+str(i+1)+'/'+str(len(urls)))
        tar = tarfile.open('./data/file'+str(i)+'.tar.gz')
        tar.extractall('./data')
        tar.close()
        print('Extracted file '+str(i+1)+'/'+str(len(urls)))
        os.remove('./data/file'+str(i)+'.tar.gz')



#extract stuff for LDA and KNN (articles in English only)
def extract():
     
    all_abstracts = [] # put all abstracts in a list for LDA
    title_files = 0 # count files with a title
    no_title_files = 0 # count files with no title
    non_covid19_count = 0 # count articles not related to COVID
    covid19_count=0 # count articles related to COVID
    files_count = 0 # count total files
    abstract_files = 0 # count articles that have an abstract
    no_abstract_files = 0 # count articles that have no abstract
    lang_list = [] #create a dictionary in a form {language:number of files in this language}
    lang_dict = {}
   
    df = pd.DataFrame() #create a dataframe for KNN query search
    
    all_sorts_of_counts = {} #dictionary of all the counted things


    #Iterate through all files in the data directory
    for subdir, dirs, files in os.walk('./data'):
        for file in files:
            files_count += 1 #count total files in a database
            with open(os.path.join(subdir, file), encoding = 'ISO-8859-1') as f:
                try:
                    dict = {}
                    data=json.load(f)             #load files as json files for ease of access 
                    try:
                        language = detect(data['metadata']['title'])  #detect the language of a file
                        lang_list.append(language) #append the language to the list

                        if language == 'en':                #look at only english language titles
                            paper_id = data['paper_id']
                            title = data['metadata']['title']  # define a title
                            
                            if len(title) > 0:                 # count files with/without title
                                title_files += 1 
                            else:
                                no_title_files += 1 
                                
                            dict['paper_id']=paper_id
                            dict['title'] = title
                            
                            try:
                                abstract = data['abstract'][0]['text'] #define abstract
                                abstract_files += 1                    #count files with/without abstract
                                dict['abstract']=abstract
                                all_abstracts.append(abstract) #append abstract for LDA
                            except IndexError:
                                no_abstract_files += 1 
                                
                           # count files that mention covid 19 in title or abstract 
                            if ('COVID' in title or 'COVID-19' in title or 
                            '2019-nCoV' in title or 'SARS-CoV-2' in title or 'Novel coronavirus' in title) or ('COVID' in abstract or 'COVID-19' in abstract or '2019-nCoV' in abstract 
                            or 'SARS-CoV-2' in abstract or 'Novel coronavirus' in abstract):
                                covid19_count += 1 
                            else:
                                non_covid19_count += 1
                                    
                            df = df.append(dict,ignore_index=True) #create a dataframe for KNN with paper id, title, abstract 
                       
                        else:
                            pass
                    
                    except:
                        language == 'langdetect.lang_detect_exception.LangDetectException: No features in text.' #error in case language is undetectable 
                        pass   
                        
                except ValueError: #error raised when json is not readable 
                    pass
    df['text'] = df['title'].fillna('') + df['abstract'].fillna('') #combine title and abstract in dataframe, fill out NaN values 

    all_sorts_of_counts['total files']=files_count
    all_sorts_of_counts['english articles with a title'] = title_files       # combine all the counts in one dictionary 
    all_sorts_of_counts['english articles without a title'] = no_title_files
    all_sorts_of_counts['english articles with an abstract'] = abstract_files
    all_sorts_of_counts['english articles with no abstract'] = no_abstract_files
    all_sorts_of_counts['COVID related articles'] = covid19_count
    all_sorts_of_counts['COVID unrelated articles'] = non_covid19_count
    
     #count occurrences of diff languages in dict           
    for lang in lang_list:
        if lang in lang_dict:
            lang_dict[lang] += 1
        else:
            lang_dict[lang] = 1
    
    return(all_abstracts, df, all_sorts_of_counts, lang_dict)


all_abstracts, df, all_sorts_of_counts, lang_dict = extract()
print(all_sorts_of_counts)
print(lang_dict)



#run LDA and graph the composition of the english side of the collection
def organize():
    from gensim import corpora, models
    nltk.download('punkt')
    nltk.download('stopwords')
    
    def cleanse(text):
        tokens = word_tokenize(text)
        tokens = [t.lower() for t in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [t.translate(table) for t in tokens]
        alphabetic_words = [token for token in stripped if token.isalpha()]
        stop_words = set(stopwords.words('english'))
        alphabetic_words = [word for word in alphabetic_words if not word in stop_words]
        alphabetic_words = [word for word in alphabetic_words if len(word) > 1]
        return alphabetic_words
    
    cleansed_abstracts = [] #format the abstracts
    for abstract in all_abstracts:
        abstract = cleanse(abstract)
        cleansed_abstracts.append(abstract)
        
    stemmed_abstracts = []
    ps = PorterStemmer()
    for abstract in cleansed_abstracts:
        abstract = [ps.stem(i) for i in abstract]
        stemmed_abstracts.append(abstract)
        
    dictionary = corpora.Dictionary(stemmed_abstracts)
    dictionary.filter_extremes(no_below = 50, no_above = 0.5)
    corpus = [dictionary.doc2bow(abstract) for abstract in stemmed_abstracts]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=5)
    
    return(ldamodel.show_topics(formatted=True, num_topics=10, num_words=10))

#print out the results of LDA 
topics = organize()  
print("10 LDA topics:")
for i,topic in topics:
    print(str(i)+": "+ topic)
    print()
    
query = input('Please enter your queries separated by comma, with no brackets or quotation marks: \n').split(', ')
print(f'Your query is {query} ')   
    
#embed the query, embed the titles and abstract, and use kNN to find 100 nearest neighbors to a query    
def retrieve(query): 

    model_path = input("Please specify model path: ")
    model = sent2vec.Sent2vecModel()

    try:
        model.load_model(model_path)  
    except Exception as e:
        print(e)
    print('model successfully loaded')

    print('Searching for 100 nearest neighbors, Godspeed')

#clean and tokenize title, abstract. Embed with Biosentvec. 
    new_df = pd.DataFrame()
    all_vectors = []
    for index, row in df.iterrows():
        dict={}
        row['text'] = sent_tokenize(row['text'])
        dict['paper_id'] =  row['paper_id']
        for i in range(0, len(row['text'])):
            row['text'][i] = row['text'][i].lower()
            row['text'][i] = row['text'][i].translate(str.maketrans('', '', string.punctuation))
            dict['sentence']=row['text'][i]
            new_df = new_df.append(dict,ignore_index=True)
            all_vectors.append(model.embed_sentence(row['text'][i]))

    array = np.array(all_vectors)
    reshape = array.reshape(array.shape[0],array.shape[2])
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(reshape) #neighbors database
    
#embed query and find neighbors
    results = []
    for x in range(0,len(query)):
        embed = model.embed_sentence(query[x])
        embed = np.array(embed)
        distances, ind = nbrs.kneighbors(embed)
        list_of_lists=[]
        for num in ind[0]:
            paper_id = new_df['paper_id'].iloc[num]
            list_of_lists.append(paper_id)
        results.append(list_of_lists)

    return(results)
    
results = retrieve(query)

for query in range(len(results)):
        for rank in range(len(results[query])):
            print(str(query+1)+'\t'+str(rank+1)+'\t'+str(results[query][rank]))
    
    
    
    
    
    
    
    