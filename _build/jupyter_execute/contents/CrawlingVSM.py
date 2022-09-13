#!/usr/bin/env python
# coding: utf-8

# # CRAWLING DATA TWITTER MENGGUNAKAN METODE VECTOR SPACE MODEL

# Untuk Tahap-Tahap sebagai berikut:

# 1.Lakukan Connect Google colab dengan goole Drive sebagai penyimpanan

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# 2.Pindah Path ke /content/drive/MyDrive/webmining/webmining

# In[40]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/webmining/')


# 3.Clone Twint dari Github Twint Project

# In[5]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# ## Penjelasan Twint

# Twint adalah alat pengikis Twitter canggih yang ditulis dengan Python yang memungkinkan untuk menggores Tweet dari profil Twitter tanpa menggunakan API Twitter.

# 4.install Library Twint

# In[3]:


get_ipython().system('pip install twint')


# 5.install aiohttp versi 3.7.0

# In[4]:


get_ipython().system('pip install aiohttp==3.7.0')


# 6.melalukan Import Twint

# 

# In[6]:


import twint


# In[7]:


get_ipython().system('pip install nest_asyncio')
import nest_asyncio
nest_asyncio.apply() 


# 7.configurasi Twint

# In[15]:


c = twint.Config()
c.Search = '#puanmaharani'
c.Pandas = True
c.Limit = 60
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "dataPuan.csv"
twint.run.Search(c)


# ## Penjelasan Pandas

# **Pandas adalah paket Python open source yang paling sering dipakai untuk menganalisis data serta membangun sebuah machine learning. Pandas dibuat berdasarkan satu package lain bernama Numpy**

# 8.melakukan Import Pandas

# In[16]:


import pandas as pd


# 9.Baca data excel dataPuan.xlsx yang telah dilabeli yang disimpan di Google Drive

# In[17]:


data = pd.read_excel('dataPuan.xlsx')
data


# ## Penjelasan NLTK

# **NLTK adalah singkatan dari Natural Language Tool Kit, yaitu sebuah library yang digunakan untuk membantu kita dalam bekerja dengan teks. Library ini memudahkan kita untuk memproses teks seperti melakukan classification, tokenization, stemming, tagging, parsing, dan semantic reasoning.**

# ## Penjelasan Sastrawi

# **Python Sastrawi adalah pengembangan dari proyek PHP Sastrawi. Python Sastrawi merupakan library sederhana yang dapat mengubah kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya. Sastrawi juga dapat diinstal melalui “pip”**

# 10.Install Library nltk dan Sastrawi

# In[18]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# ## Penjelasan RE

# **Re module Python menyediakan seperangkat fungsi yang memungkinkan kita untuk mencari sebuah string untuk match (match).**

# 11.Lakukan Import beberapa Library seperti Pandas,re,nltk,string dan Sastrawi

# In[19]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# 12.Selanjutnya membuat Function Remove Stopwords yang fungsinya adalah menghapus kata-kata yang tidak diperlukan dalam proses nantinya,sehingga dapat mempercepat proses VSM

# In[53]:


def remove_stopwords(text):
    with open('stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# 13.Steming merupakan proses mengubah kata dalam bahasa Indonesia ke akar katanya misalkan 'Mereka meniru-nirukannya' menjadi 'mereka tiru'

# In[21]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# 14.Selanjutnya tahap preprocessing,untuk tahap ini ada beberapa proses seperti:  
# 
# 
# > 1.Mengubah Text menjadi huruf kecil
# 
# 
# > 2.Menghilangkan Url didalam Text
# 
# 
# > 3.Mengubah/menghilangkan tanda (misalkan garis miring menjadi spasi)
# 
# 
# > 4.Melakukan tokenization kata dan Penghapusan Kata yang tidak digunakan
# 
# 
# > 5.Memfilter kata dari tanda baca
# 
# > 6.Mengubah kata dalam bahasa Indonesia ke akar katanya
# 
# 
# > 7.Menghapus String kosong
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[22]:


def preprocessing(text):
    #case folding
    text = text.lower()
    #remove urls
    text = re.sub('http\S+', '', text)
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


# 15.Membuat Tabel dari hasil Preprocessing,disitu juga menambahkan nama column(Abstraksi) dan baris(kata)

# In[51]:


get_ipython().run_line_magic('cd', '../contents')


# In[52]:


tf = pd.DataFrame()
for i,v in enumerate(data['tweet']):
    cols = ["Doc " + str(i+1)]    
    doc = pd.DataFrame.from_dict(nltk.FreqDist(preprocessing(v)), orient='index',columns=cols) 
    #doc.columns = [data['Judul'][i]]    
    tf = pd.concat([tf, doc], axis=1, sort=False)


# In[54]:


tf.index.name = 'Term'
tf = pd.concat([tf], axis=1, sort=False)
tf = tf.fillna(0)
tf


# ## Penjelasan Scikit-learn

# Scikit-learn atau sklearn merupakan sebuah module dari bahasa pemrograman Python yang dibangun berdasarkan NumPy, SciPy, dan Matplotlib. Fungsi dari module ini adalah untuk membantu melakukan processing data ataupun melakukan training data untuk kebutuhan machine learning atau data science.

# 16.install scikit-learn

# In[55]:


get_ipython().system('pip install -U scikit-learn')


# 17.mengumpulkan data untuk di Train

# In[56]:


train = tf.iloc[:,:len(data)]


# In[65]:


cols = train.columns
df = pd.DataFrame(train[cols].gt(0).sum(axis=1), columns=['Document Frequency'])

idf = np.log10(len(cols)/df)
idf.columns = ['Inverse Document Frequency']
idf = pd.concat([df, idf], axis=1)
idf


# 18.Mengurutkan suku kata berdasarkan jumlah suku kata yang paling banyak keluar 

# In[70]:


# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(idf.drop(labels=['Document Frequency',],axis=1),
#                                                idf['Document Frequency'],
#                                                test_size=0.3,
#                                                random_state=0)
# from sklearn.feature_selection import mutual_info_classif
# #mutual_info=mutual_info_classif(idf['Document Frequency'],idf['Inverse Document Frequency'])
# X_test
pd.DataFrame(idf['Document Frequency'].sort_values(ascending=False)).head(15)

