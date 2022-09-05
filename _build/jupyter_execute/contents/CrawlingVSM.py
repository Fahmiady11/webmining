#!/usr/bin/env python
# coding: utf-8

# # **CRAWLING DATA PTA TRUNOJOYO.AC.ID MENGGUNAKAN METODE VECTOR SPACE MODEL**

# Untuk Tahap-Tahap sebagai berikut:

# 1.Lakukan Connect Google colab dengan goole Drive sebagai penyimpanan

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# 2.Pindah Path ke /content/drive/MyDrive/webmining/webmining/content

# In[8]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/webmining/content')


# 3.Selanjutnya Import Scrapy atau Install jika belum pernah Menginstall

# In[9]:


try:
    import scrapy
except:
    get_ipython().system('pip install scrapy')
    import scrapy


# 4.Import Pandas yang nantinya digunakan untuk membaca file Json maupun CSV

# In[10]:


import pandas as pd


# ## Scrapy adalah web crawling dan web scraping framework tingkat tinggi yang cepat, digunakan untuk merayapi situs web dan mengekstrak data terstruktur dari halaman mereka. Ini dapat digunakan untuk berbagai tujuan, mulai dari penambangan data hingga pemantauan dan pengujian otomatis.

# 5.Untuk Crawling Data pertama yang saya lakukan adalah untuk mendapatkan Link,untuk Link nya saya Crawling 50 lebih

# In[11]:


class LinkSpider(scrapy.Spider):
    name='link'
    start_urls=[]
    for i in range(1, 50+1):
        start_urls.append(f'https://pta.trunojoyo.ac.id/c_search/byprod/10/{i}')
    def parse(self, response):
        count=0
        link=[]
        for jurnal in response.css('#content_journal > ul'):
            count+=1
            for j in range(1,6):
                yield {
                    'link': response.css(f'li:nth-child({j}) > div:nth-child(3) > a::attr(href)').get(),
                }


# 6.selanjutnya ingin mengetahui hasil dari Crawling yang berbentuk Json

# In[12]:


df = pd.read_json('jurnal.json')
df


# 7.Setelah mendapatkan Data Link dari PTA.trunojoyo.ac.id lalu dicrawling lagi dari Link yang tadi,untuk mendapatkan Kumpulan Judul dan Abstraksi

# In[13]:


class Spider(scrapy.Spider):
    name = 'detail'
    data_csv = pd.read_json('jurnal.json').values
    start_urls = [ link[0] for link in data_csv ]

    def parse(self, response):
        yield {
            'Judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
            'Abstraksi': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
        }


# 8.Untuk menampilkan data CSV

# In[14]:


df = pd.read_csv('result.csv')


# ## **NLTK** adalah singkatan dari Natural Language Tool Kit, yaitu sebuah library yang digunakan untuk membantu kita dalam bekerja dengan teks. Library ini memudahkan kita untuk memproses teks seperti melakukan classification, tokenization, stemming, tagging, parsing, dan semantic reasoning.

# ## **Python Sastrawi** adalah pengembangan dari proyek PHP Sastrawi. Python Sastrawi merupakan library sederhana yang dapat mengubah kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya. Sastrawi juga dapat diinstal melalui “pip”

# 9.Install Library nltk dan Sastrawi

# In[15]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# ## **Pandas** adalah paket Python open source yang paling sering dipakai untuk menganalisis data serta membangun sebuah machine learning. Pandas dibuat berdasarkan satu package lain bernama Numpy

# ## **Re** module Python menyediakan seperangkat fungsi yang memungkinkan kita untuk mencari sebuah string untuk match (match).

# 10.Lakukan Import beberapa Library seperti Pandas,re,nltk,string dan Sastrawi

# In[24]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# 11.Menampilkan data CSV dengan bantuan Library Pandas

# In[17]:


data = pd.read_csv('result.csv').dropna()
data


# 12.Selanjutnya membuat Function Remove Stopwords yang fungsinya adalah menghapus kata-kata yang tidak diperlukan dalam proses nantinya,sehingga dapat mempercepat proses VSM

# In[18]:


def remove_stopwords(text):
    with open('stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# 13.Steming merupakan proses mengubah kata dalam bahasa Indonesia ke akar katanya misalkan 'Mereka meniru-nirukannya' menjadi 'mereka tiru'

# In[19]:


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

# In[26]:


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

# In[21]:


tf = pd.DataFrame()
for i,v in enumerate(data['Abstraksi']):
    cols = ["Doc " + str(i+1)]    
    doc = pd.DataFrame.from_dict(nltk.FreqDist(preprocessing(v)), orient='index',columns=cols) 
    #doc.columns = [data['Judul'][i]]    
    tf = pd.concat([tf, doc], axis=1, sort=False)


# In[22]:


tf.index.name = 'Term'
tf = pd.concat([tf], axis=1, sort=False)
tf = tf.fillna(0)
tf


# 16.Menampilkan nilai tabel berdasarkan hasil Asli tanpa pembulatan

# In[25]:


tf[tf != 0] = 1 + np.log10(tf)
tf

