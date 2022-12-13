#!/usr/bin/env python
# coding: utf-8

# # UAS PPW Crawling Data Comment Youtube

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/webmining/')


# In[ ]:


get_ipython().run_line_magic('cd', 'contents')


# #1. Import Library

# In[ ]:


import pandas as pd
from googleapiclient.discovery import build


# #2. Fungsi untuk crawling komentar

# In[ ]:


def video_comments(video_id):
	# empty list for storing reply
	replies = []

	# creating youtube resource object
	youtube = build('youtube', 'v3', developerKey=api_key)

	# retrieve youtube video results
	video_response = youtube.commentThreads().list(part='snippet,replies', videoId=video_id).execute()

	# iterate video response
	while video_response:
		
		# extracting required info
		# from each result object
		for item in video_response['items']:
			
			# Extracting comments ()
			published = item['snippet']['topLevelComment']['snippet']['publishedAt']
			user = item['snippet']['topLevelComment']['snippet']['authorDisplayName']

			# Extracting comments
			comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
			likeCount = item['snippet']['topLevelComment']['snippet']['likeCount']

			replies.append([published, user, comment, likeCount])
			
			# counting number of reply of comment
			replycount = item['snippet']['totalReplyCount']

			# if reply is there
			if replycount>0:
				# iterate through all reply
				for reply in item['replies']['comments']:
					
					# Extract reply
					published = reply['snippet']['publishedAt']
					user = reply['snippet']['authorDisplayName']
					repl = reply['snippet']['textDisplay']
					likeCount = reply['snippet']['likeCount']
					
					# Store reply is list
					#replies.append(reply)
					replies.append([published, user, repl, likeCount])

			# print comment with list of reply
			#print(comment, replies, end = '\n\n')

			# empty reply list
			#replies = []

		# Again repeat
		if 'nextPageToken' in video_response:
			video_response = youtube.commentThreads().list(
					part = 'snippet,replies',
					pageToken = video_response['nextPageToken'], 
					videoId = video_id
				).execute()
		else:
			break
	#endwhile
	return replies


# #3. Jalankan Proses Crawling

# In[ ]:


# isikan dengan api key Anda
api_key = 'AIzaSyARpR_UNoGRRkODrdCLxu7tExhhk64bv5I'

# Enter video id
# contoh url video = https://www.youtube.com/watch?v=5tucmKjOGi8
video_id = "mve26ZiedmA" #isikan dengan kode / ID video

# Call function
comments = video_comments(video_id)

comments


# #4. Ubah Hasil Crawling ke Dataframe

# In[ ]:


df = pd.DataFrame(comments, columns=['publishedAt', 'authorDisplayName', 'textDisplay', 'likeCount'])
df


# #5. Simpan Hasil Crawling ke file CSV

# In[ ]:


df.to_csv('youtube-comments.csv', index=False)


# In[ ]:


get_ipython().system('pip install numpy')


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_excel('youtube-comments-cut.xlsx')
# data=data.loc[:100]
# data.to_excel('youtube-comments-cut.xlsx', index=False)
data


# In[ ]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# In[ ]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[ ]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/webmining/webmining/contents/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# In[ ]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# In[ ]:


def preprocessing(text):
    #case folding
    text = text.lower()

    #remove non ASCII (emoticon, chinese word, .etc)
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")

    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')

    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())

    #replace weird characters
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('-', ' ')

    #tokenization and remove stopwords
    text = remove_stopwords(text)

    #remove punctuation    
    text = [''.join(c for c in s if c not in string.punctuation) for s in text]  

    #stemming
    text = stemming(text)

    #remove empty string
    text = list(filter(None, text))
    return text


# In[ ]:


# data['textDisplay'].apply(preprocessing).to_excel('preprocessing-youtube.xlsx')


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_excel('/content/drive/MyDrive/webmining/webmining/contents/preprocessing-youtube.xlsx')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['comment'])
dataTextPre


# In[ ]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[ ]:


matrik_vsm[0]


# In[ ]:


a=vectorizer.get_feature_names()
dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[ ]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[ ]:


label = pd.read_csv('/content/drive/MyDrive/webmining/webmining/contents/youtube-comments.csv')
dj = pd.concat([dataTF.reset_index(), dj["label"]], axis=1)
dj


# In[ ]:


dj['label'].unique()


# In[ ]:


from sklearn.model_selection import train_test_split
#membagi kumpulan data menjadi data pelatihan dan data pengujian.
X_train,X_test,y_train,y_test=train_test_split(dj.drop(labels=['label'], axis=1),
    dj['label'],
    test_size=0.3,
    random_state=0)


# In[ ]:


X_train


# In[ ]:


from sklearn.model_selection import train_test_split
#membagi kumpulan data menjadi data pelatihan dan data pengujian.
X_train,X_test,y_train,y_test=train_test_split(dj.drop(labels=['label'], axis=1),
    dj['label'],
    test_size=0.3,
    random_state=0)


# In[ ]:


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
estimators = [
    ('rf', RandomForestClassifier(random_state=42,max_features='auto', n_estimators= 100, max_depth=8, criterion='gini')),
    ('rf2', RandomForestClassifier(random_state=42,max_features='auto', n_estimators= 100, max_depth=8, criterion='entropy'))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=10, random_state=42)
)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, stratify=y, random_state=42
# )
clf.fit(X_train.values, y_train.values).score(X_test.values, y_test.values)

