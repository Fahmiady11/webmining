#!/usr/bin/env python
# coding: utf-8

# # CRAWLING DATA TWITTER MENGGUNAKAN METODE VECTOR SPACE MODEL

# Crawling Data adalah teknik pengumpulan data yang digunakan untuk mengindeks informasi pada halaman menggunakan URL (Uniform Resource Locator) dengan menyertakan API (Application Programming Interface) untuk melakukan penambangan dataset yang lebih besar.
# 
# Data yang dapat kamu kumpulkan dapat berupa text, audio, video, dan gambar. Kamu dapat memulai dengan melakukan penambangan data pada API yang bersifat open source seperti yang disediakan oleh Twitter. Untuk melakukan crawling data di Twitter kamu dapat menggunakan library scrapy ataupun twint pada python.

# Untuk Tahap-Tahap sebagai berikut:

# Lakukan Connect Google colab dengan goole Drive sebagai penyimpanan

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# Pindah Path ke /content/drive/MyDrive/webmining/webmining

# In[2]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/webmining/')


# Clone Twint dari Github Twint Project

# In[3]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# ## Penjelasan Twint

# Twint adalah alat pengikis Twitter canggih yang ditulis dengan Python yang memungkinkan untuk menggores Tweet dari profil Twitter tanpa menggunakan API Twitter.

# install Library Twint

# In[4]:


get_ipython().system('pip install twint')


# install aiohttp versi 3.7.0

# In[5]:


get_ipython().system('pip install aiohttp==3.7.0')


# melakukan Import Twint

# 

# In[6]:


import twint


# Install Nest Asyncio dan lakukan Import

# In[7]:


get_ipython().system('pip install nest_asyncio')
import nest_asyncio
nest_asyncio.apply() 


# configurasi Twint dengan value seperti dibawah

# In[8]:


c = twint.Config()
c.Search = 'tragedi kanjuruhan'
c.Pandas = True
c.Limit = 60
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "tragediKanjuruhan127.csv"
twint.run.Search(c)


# ## Penjelasan Pandas

# **Pandas adalah paket Python open source yang paling sering dipakai untuk menganalisis data serta membangun sebuah machine learning. Pandas dibuat berdasarkan satu package lain bernama Numpy**

# melakukan Import Pandas

# In[10]:


import pandas as pd


# Baca data excel Tragedi Kanjuruhan yang telah simpan di Google Drive

# In[12]:


data = pd.read_csv('tragediKanjuruhan127.csv')
data


# ## Penjelasan NLTK

# **NLTK adalah singkatan dari Natural Language Tool Kit, yaitu sebuah library yang digunakan untuk membantu kita dalam bekerja dengan teks. Library ini memudahkan kita untuk memproses teks seperti melakukan classification, tokenization, stemming, tagging, parsing, dan semantic reasoning.**

# ## Penjelasan Sastrawi

# **Python Sastrawi adalah pengembangan dari proyek PHP Sastrawi. Python Sastrawi merupakan library sederhana yang dapat mengubah kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya. Sastrawi juga dapat diinstal melalui “pip”**

# Install Library nltk dan Sastrawi

# In[13]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# ## Penjelasan RE

# **Re module Python menyediakan seperangkat fungsi yang memungkinkan kita untuk mencari sebuah string untuk match (match).**

# Lakukan Import beberapa Library seperti Pandas,re,nltk,string dan Sastrawi

# In[14]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Selanjutnya membuat Function Remove Stopwords yang fungsinya adalah menghapus kata-kata yang tidak diperlukan dalam proses nantinya,sehingga dapat mempercepat proses VSM

# In[15]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/webmining/webmining/contents/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# Steming merupakan proses mengubah kata dalam bahasa Indonesia ke akar katanya misalkan 'Mereka meniru-nirukannya' menjadi 'mereka tiru'

# In[16]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# Selanjutnya tahap preprocessing,untuk tahap ini ada beberapa proses seperti:  
# 
# 
# > 1.Mengubah Text menjadi huruf kecil
# 
# > 2.Menghapus Kata non Ascii
# 
# > 4.Menghapus Hastag,Link dan Mention
# 
# > 5.Mengubah/menghilangkan tanda (misalkan garis miring menjadi spasi)
# 
# > 6.Melakukan tokenization kata dan Penghapusan Kata yang tidak digunakan
# 
# > 7.Memfilter kata dari tanda baca
# 
# > 8.Mengubah kata dalam bahasa Indonesia ke akar katanya
# 
# > 9.Menghapus String kosong
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
# 
# 
# 

# In[17]:


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


# Selanjutnya pindah Path ke Folder contents

# In[18]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/webmining/contents')


# Simpan hasil dari preprocessing ke dalam bentuk CSV

# In[19]:


# data['tweet'].apply(preprocessing).to_excel('preprocessingTK.xlsx')


# Tokenizing adalah proses pemisahan teks menjadi potongan-potongan yang disebut sebagai token untuk kemudian di analisa. Kata, angka, simbol, tanda baca dan entitas penting lainnya dapat dianggap sebagai token.

# In[20]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_excel('/content/drive/MyDrive/webmining/webmining/contents/preprocessingTK.xlsx')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])
dataTextPre


# Melihat Jumlah Baris dan Kata

# In[21]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[22]:


matrik_vsm[0]


# In[23]:


a=vectorizer.get_feature_names()


# Tampilan data VSM dengan labelnya 

# In[24]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# ## Penjelasan Matplotib

# Matplotlib adalah library Python yang fokus pada visualisasi data seperti membuat plot grafik. Matplotlib pertama kali diciptakan oleh John D. Hunter dan sekarang telah dikelola oleh tim developer yang besar. Awalnya matplotlib dirancang untuk menghasilkan plot grafik yang sesuai pada publikasi jurnal atau artikel ilmiah. Matplotlib dapat digunakan dalam skrip Python, Python dan IPython shell, server aplikasi web, dan beberapa toolkit graphical user interface (GUI) lainnya.

# In[25]:


#import plt
import matplotlib.pyplot as plt
#import metrics
from sklearn import metrics


# ## Penjelasan K-Means

# K-Means Clustering merupakan algoritma yang efektif untuk menentukan cluster dalam sekumpulan data, di mana pada algortima tersebut dilakukan analisis kelompok yang mengacu pada pemartisian N objek ke dalam K kelompok (Cluster) berdasarkan nilai rata-rata (means) terdekat. Adapun persamaan yang sering digunakan dalam pemecahan masalah dalam menentukan jarak terdekat adalah persamaan Euclidean berikut :

# 
# $$
# d(p,q) = \sqrt{(p_{1}-q_{1})^2+(p_{2}-q_{2})^2+(p_{3}-q_{3})^2}
# $$
# 
# 
# d = jarak obyek
# 
# p = data 
# 
# q = centroid

# TruncatedSVD adalah Teknik pengurangan dimensi menggunakan SVD terpotong

# In[26]:


from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


# In[59]:


# Latih Kmeans dengan n cluster terbaik
modelKm = KMeans(n_clusters=3, random_state=12)
modelKm.fit(dataTF.values)
prediksi = modelKm.predict(dataTF.values)

# Pengurangan dimensi digunakan untuk memplot dalam representasi 2d
pc=TruncatedSVD(n_components=2)
X_new=pc.fit_transform(dataTF.values)
centroids=pc.transform(modelKm.cluster_centers_)
print(centroids)
plt.scatter(X_new[:,0],X_new[:,1],c=prediksi, cmap='viridis')
plt.scatter(centroids[:,0] , centroids[:,1] , s = 50, color = 'red')


# ## Perangkingan Kalimat Berita dengan Method Page Rank

# ## Penjelasan Scrapy

# Scrapy adalah web crawling dan web scraping framework tingkat tinggi yang cepat, digunakan untuk merayapi situs web dan mengekstrak data terstruktur dari halaman mereka. Ini dapat digunakan untuk berbagai tujuan, mulai dari penambangan data hingga pemantauan dan pengujian otomatis.

# In[28]:


get_ipython().system('pip install scrapy')
get_ipython().system('pip install crochet')


# In[29]:


import scrapy


# In[30]:


import scrapy
from scrapy.crawler import CrawlerRunner
import re
from crochet import setup, wait_for
setup()

class QuotesToCsv(scrapy.Spider):
    name = "MJKQuotesToCsv"
    start_urls = [
        'https://nasional.tempo.co/read/1642981/usai-tragedi-kanjuruhan-jokowi-klaim-indonesia-tak-dikenai-sanksi-dari-fifa',
    ]
    custom_settings = {
        'ITEM_PIPELINES': {
            '__main__.ExtractFirstLine': 1
        },
        'FEEDS': {
            'news.csv': {
                'format': 'csv',
                'overwrite': True
            }
        }
    }

    def parse(self, response):
        """parse data from urls"""
        for quote in response.css('#isi > p'):
            yield {'news': quote.extract()}


class ExtractFirstLine(object):
    def process_item(self, item, spider):
        """text processing"""
        lines = dict(item)["news"].splitlines()
        first_line = self.__remove_html_tags__(lines[0])

        return {'news': first_line}

    def __remove_html_tags__(self, text):
        """remove html tags from string"""
        html_tags = re.compile('<.*?>')
        return re.sub(html_tags, '', text)

@wait_for(10)
def run_spider():
    """run spider with MJKQuotesToCsv"""
    crawler = CrawlerRunner()
    d = crawler.crawl(QuotesToCsv)
    return d


# In[31]:


# run_spider()


# Mengambil dan Membaca data CSV yang bernama news.csv

# In[32]:


dataNews = pd.read_csv('news.csv')
dataNews


# PyPDF2 adalah pustaka PDF python murni gratis dan open-source yang mampu memisahkan, menggabungkan , memotong, dan mengubah halaman file PDF.

# Install PyPDF2

# In[33]:


get_ipython().system('pip install PyPDF2')


# import PyPDF2

# In[34]:


import PyPDF2


# Membaca Pdf dari file lalu dibuat menjadi bentuk document Text

# In[35]:


pdfReader = PyPDF2.PdfFileReader('/content/drive/MyDrive/webmining/webmining/contents/news.pdf')
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()
document


# PunktSentenceTokenizer adalah Sebuah tokenizer kalimat yang menggunakan algoritma tanpa pengawasan untuk membangun model untuk kata-kata singkatan, kolokasi, dan kata-kata yang memulai kalimat dan kemudian menggunakan model itu untuk menemukan batas kalimat.

# In[36]:


from nltk.tokenize.punkt import PunktSentenceTokenizer


# In[37]:


def tokenize(document):
    # Kita memecahnya menggunakan  PunktSentenceTokenizer
    doc_tokenizer = PunktSentenceTokenizer()
    # sentences_list adalah daftar masing masing kalimat dari dokumen yang ada.
    sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list


# In[38]:


sentences_list = tokenize(document)
sentences_list


# Merapikan data di atas sehingga lebih enak dibaca

# In[39]:


kal=1
for i in sentences_list:
    print('\nKalimat {}'.format(kal))
    kal+=1
    print(i)


# Tokenizing adalah proses pemisahan teks menjadi potongan-potongan yang disebut sebagai token untuk kemudian di analisa. Kata, angka, simbol, tanda baca dan entitas penting lainnya dapat dianggap sebagai token.

# In[40]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
vectorizer = CountVectorizer()
cv_matrix=vectorizer.fit_transform(sentences_list)


# Menampilkan jumlah Kosa Kata dari Data

# In[41]:


print ("Banyaknya kosa kata = ", len((vectorizer.get_feature_names_out())))


# Menampilkan jumlah Kalimat dari Data

# In[42]:


print ("Banyaknya kalimat = ", (len(sentences_list)))


# Menampilkan Kosa Kata dari Data

# In[43]:


print ("kosa kata = ", (vectorizer.get_feature_names_out()))


# In[44]:


# mengubah kumpulan dokumen mentah menjadi matriks fitur TF-IDF
normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())


# Menampilkan Jumlah Kalimat dan Kosa Kata

# In[45]:


normal_matrix.shape


# NetworkX adalah paket Python untuk pembuatan, manipulasi, dan studi tentang struktur, dinamika, dan fungsi jaringan yang kompleks. Ini menyediakan:

# In[46]:


import networkx as nx


# Graph adalah kumpulan dati titik (node) dan garis dimana pasangan – pasangan titik (node) tersebut dihubungkan oleh segmen garis. Node ini biasa disebut simpul (vertex) dan segmen garis disebut ruas (edge)

# In[47]:


res_graph = normal_matrix * normal_matrix.T
print(res_graph)


# In[48]:


nx_graph = nx.from_scipy_sparse_matrix(res_graph)


# In[49]:


nx.draw_circular(nx_graph)


# Jumlah Banyak Sisi 

# In[50]:


print('Banyaknya sisi {}'.format(nx_graph.number_of_edges()))


# Menkalikan data dengan data Transpose

# In[51]:


res_graph = normal_matrix * normal_matrix.T


# ## Penjelasan PageRabk

# PageRank adalah menghitung peringkat node dalam grafik G berdasarkan struktur tautan masuk. Awalnya dirancang sebagai algoritma untuk menentukan peringkat halaman web.

# In[52]:


ranks=nx.pagerank(nx_graph,)


# memasukkan data ke array

# In[53]:


arrRank=[]
for i in ranks:
    arrRank.append(ranks[i])


# menjadikan data kedalam bentuk tabel lalu digabungkan 

# In[54]:


dfRanks = pd.DataFrame(arrRank,columns=['PageRank'])
dfSentence = pd.DataFrame(sentences_list,columns=['News'])
dfJoin = pd.concat([dfSentence,dfRanks], axis=1)
dfJoin


# Mengurutkan data berdasarkan hasil tertinggi

# In[55]:


sortSentence=dfJoin.sort_values(by=['PageRank'],ascending=False)
sortSentence


# Menampilkan data dari 5 ke atas

# In[56]:


sortSentence.head(5)

