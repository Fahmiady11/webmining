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


# \configurasi Twint dengan value seperti dibawah

# In[8]:


c = twint.Config()
c.Search = 'tragedi kanjuruhan'
c.Pandas = True
c.Limit = 60
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "dataGanjar.csv"
twint.run.Search(c)


# ## Penjelasan Pandas

# **Pandas adalah paket Python open source yang paling sering dipakai untuk menganalisis data serta membangun sebuah machine learning. Pandas dibuat berdasarkan satu package lain bernama Numpy**

# melakukan Import Pandas

# In[9]:


import pandas as pd


# Baca data excel dataGanjar.xlsx yang telah diberi label (Positif,Negatif dan Netral) yang telah simpan di Google Drive

# In[10]:


data = pd.read_excel('dataGanjar.xlsx')
data


# ## Penjelasan NLTK

# **NLTK adalah singkatan dari Natural Language Tool Kit, yaitu sebuah library yang digunakan untuk membantu kita dalam bekerja dengan teks. Library ini memudahkan kita untuk memproses teks seperti melakukan classification, tokenization, stemming, tagging, parsing, dan semantic reasoning.**

# ## Penjelasan Sastrawi

# **Python Sastrawi adalah pengembangan dari proyek PHP Sastrawi. Python Sastrawi merupakan library sederhana yang dapat mengubah kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya. Sastrawi juga dapat diinstal melalui “pip”**

# Install Library nltk dan Sastrawi

# In[11]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# ## Penjelasan RE

# **Re module Python menyediakan seperangkat fungsi yang memungkinkan kita untuk mencari sebuah string untuk match (match).**

# Lakukan Import beberapa Library seperti Pandas,re,nltk,string dan Sastrawi

# In[12]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Selanjutnya membuat Function Remove Stopwords yang fungsinya adalah menghapus kata-kata yang tidak diperlukan dalam proses nantinya,sehingga dapat mempercepat proses VSM

# In[13]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/webmining/webmining/contents/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# Steming merupakan proses mengubah kata dalam bahasa Indonesia ke akar katanya misalkan 'Mereka meniru-nirukannya' menjadi 'mereka tiru'

# In[14]:


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

# In[15]:


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

# In[16]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/webmining/contents')


# Simpan hasil dari preprocessing ke dalam bentuk CSV

# In[17]:


#data['tweet'].apply(preprocessing).to_excel('preprocessing.xlsx')


# Tokenizing adalah proses pemisahan teks menjadi potongan-potongan yang disebut sebagai token untuk kemudian di analisa. Kata, angka, simbol, tanda baca dan entitas penting lainnya dapat dianggap sebagai token.

# In[18]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_excel('/content/drive/MyDrive/webmining/webmining/contents/preprocessing.xlsx')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])
dataTextPre


# Melihat Jumlah Baris dan Kata

# In[19]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[20]:


matrik_vsm[0]


# In[21]:


a=vectorizer.get_feature_names()


# Tampilan data VSM dengan labelnya 

# In[22]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# 
# 
# ```
# # Ini diformat sebagai kode
# ```
# 
# lalu data diatas ditambahkan dengan label (positif,netral dan negatif)

# In[23]:


label = pd.read_excel('/content/drive/MyDrive/webmining/webmining/twint/dataGanjar.xlsx')
dj = pd.concat([dataTF.reset_index(), label["label"]], axis=1)
dj


# In[24]:


dj['label'].unique()


# ## Penjelasan Scikit-learn

# Scikit-learn atau sklearn merupakan sebuah module dari bahasa pemrograman Python yang dibangun berdasarkan NumPy, SciPy, dan Matplotlib. Fungsi dari module ini adalah untuk membantu melakukan processing data ataupun melakukan training data untuk kebutuhan machine learning atau data science.

# install scikit-learn

# In[25]:


get_ipython().system('pip install -U scikit-learn')


# ## Penjelasan Information Gain

# Information Gain merupakan teknik seleksi fitur yang memakai metode scoring untuk nominal
# ataupun pembobotan atribut kontinue yang didiskretkan menggunakan maksimal entropy. Suatu entropy
# digunakan untuk mendefinisikan nilai Information Gain. Entropy menggambarkan banyaknya informasi
# yang dibutuhkan untuk mengkodekan suatu kelas. Information Gain (IG) dari suatu term diukur
# dengan menghitung jumlah bit informasi yang diambil dari prediksi kategori dengan ada atau tidaknya
# term dalam suatu dokumen.

# 
# $$
# Entropy \ (S) \equiv \sum ^{c}_{i}P_{i}\log _{2}p_{i}
# $$
# 
# c : jumlah nilai yang ada pada atribut target (jumlah kelas klasifikasi).
# 
# Pi : porsi sampel untuk kelas i.

# 
# $$
# Gain \ (S,A) \equiv Entropy(S) - \sum _{\nu \varepsilon \ values } \dfrac{\left| S_{i}\right| }{\left| S\right|} Entropy(S_{v})
# $$
# 
# A : atribut
# 
# V : menyatakan suatu nilai yang mungkin untuk atribut A
# 
# Values (A) : himpunan nilai-nilai yang mungkin untuk atribut A
# 
# |Sv| : jumlah Sampel untuk nilai v
# 
# |S| : jumlah seluruh sample data Entropy 
# 
# (Sv) : entropy untuk sampel sampel yang memiliki nilai v
# 

# In[26]:


from sklearn.model_selection import train_test_split
#membagi kumpulan data menjadi data pelatihan dan data pengujian.
X_train,X_test,y_train,y_test=train_test_split(dj.drop(labels=['label'], axis=1),
    dj['label'],
    test_size=0.3,
    random_state=0)


# In[27]:


# from sklearn import model_selection
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier
# import pandas as pd

# X = X_train
# Y = y_train

# # seed = 8
# # kfold = model_selection.KFold(n_splits = 3,
# # 					random_state = seed)

# # initialize the base classifier
# base_cls = DecisionTreeClassifier()

# # no. of base classifier
# num_trees = 500

# # bagging classifier
# model = BaggingClassifier(base_estimator = base_cls,
# 						n_estimators = num_trees)

# results = model_selection.cross_val_score(model, X, Y)
# print("accuracy :")
# print(results.mean())


# In[28]:


# from sklearn import model_selection
# from sklearn.ensemble import BaggingClassifier
# from sklearn.svm import SVC
# import pandas as pd

# X = X_train
# Y = y_train

# # seed = 8
# # kfold = model_selection.KFold(n_splits = 3,
# # 					random_state = seed)

# # initialize the base classifier
# base_cls = SVC()

# # no. of base classifier
# num_trees = 500

# # bagging classifier
# model = BaggingClassifier(base_estimator = base_cls,
# 						n_estimators = num_trees)

# results = model_selection.cross_val_score(model, X, Y)
# print("accuracy :")
# print(results.mean())


# In[29]:


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import GridSearchCV

# n_estimators = [i for i in (5,21,1)]
# parameters = dict(n_estimators)
# # Instantiating the GridSearchCV object
# rand=RandomForestClassifier()
# logreg_cv = GridSearchCV(rand, parameters)
# logrec_cv

# # X = X_train
# # Y = y_train
# # clf = RandomForestClassifier(n_estimators, random_state=0)
# # clf.fit(X, Y)

# # # performing predictions on the test dataset
# # y_pred = clf.predict(X_test)
 
# # # metrics are used to find accuracy or error
# # from sklearn import metrics 
 
# # # using metrics module for accuracy calculation
# # print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


# **Penjelasan mutual_info_classif**
# mengukur ketergantungan antara variabel. Itu sama dengan nol jika dan hanya jika dua variabel acak independen, dan nilai yang lebih tinggi berarti ketergantungan yang lebih tinggi.

# In[30]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# merangking fitur(Kata) sesuai dengan fitur(Kata) yang paling banyak keluar

# In[31]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# menvisualkan data dengan grafik bar dengan urutan paling besar ke rendah

# In[32]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(50, 20))


# Import SelectKBest

# In[33]:


from sklearn.feature_selection import SelectKBest


# Pilih fitur menurut k skor tertinggi.

# In[34]:


sel_five_cols = SelectKBest(mutual_info_classif, k=100)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


# In[35]:


X_train=X_train.values
y_train=y_train.values
X_test=X_test.values
y_test=y_test.values


# ## Penjelasan Naive Bayes

# Naive Bayes adalah algoritma machine learning yang digunakan untuk keperluan klasifikasi atau pengelompokan suatu data. Algoritma ini didasarkan pada teorema probabilitas yang dikenalkan oleh ilmuwan Inggris Thomas Bayes. Naive Bayes berfungsi memprediksi probabilitas di masa depan berdasarkan pengalaman sebelumnya, sehingga dapat digunakan untuk pengambilan keputusan.

# In[36]:


# from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
gauss = GaussianNB()
gauss.fit(X_train, y_train)


# Menampilkan accuracy dari nilai test dengan method Gaussion Naive Bayes

# In[37]:


from sklearn.metrics import make_scorer, accuracy_score,precision_score
testing = gauss.predict(X_test) 
accuracy_gauss=round(accuracy_score(y_test,testing)* 100, 2)
accuracy_gauss


# ## Penjelasan Matplotib

# Matplotlib adalah library Python yang fokus pada visualisasi data seperti membuat plot grafik. Matplotlib pertama kali diciptakan oleh John D. Hunter dan sekarang telah dikelola oleh tim developer yang besar. Awalnya matplotlib dirancang untuk menghasilkan plot grafik yang sesuai pada publikasi jurnal atau artikel ilmiah. Matplotlib dapat digunakan dalam skrip Python, Python dan IPython shell, server aplikasi web, dan beberapa toolkit graphical user interface (GUI) lainnya.

# In[38]:


#import plt
import matplotlib.pyplot as plt
#import metrics
from sklearn import metrics


# ## Penjelasan Confusion Matrix

# Confusion matrix juga sering disebut error matrix. Pada dasarnya confusion matrix memberikan informasi perbandingan hasil klasifikasi yang dilakukan oleh sistem (model) dengan hasil klasifikasi sebenarnya. Confusion matrix berbentuk tabel matriks yang menggambarkan kinerja model klasifikasi pada serangkaian data uji yang nilai sebenarnya diketahui.

# membuat Confusion Matrix dengan column vertical (negatif,netral dan positif) dan column horizontal (negatif,netral dan positif)

# In[39]:


conf_matrix =metrics.confusion_matrix(y_true=y_test, y_pred=testing)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['negatif', 'netral','positif'])
cm_display.plot()
plt.show()


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

# In[40]:


from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


# In[41]:


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

# In[42]:


get_ipython().system('pip install scrapy')
get_ipython().system('pip install crochet')


# In[43]:


import scrapy


# In[44]:


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


# In[45]:


# run_spider()


# Mengambil dan Membaca data CSV yang bernama news.csv

# In[46]:


dataNews = pd.read_csv('news.csv')
dataNews


# PyPDF2 adalah pustaka PDF python murni gratis dan open-source yang mampu memisahkan, menggabungkan , memotong, dan mengubah halaman file PDF.

# Install PyPDF2

# In[47]:


get_ipython().system('pip install PyPDF2')


# import PyPDF2

# In[48]:


import PyPDF2


# Membaca Pdf dari file lalu dibuat menjadi bentuk document Text

# In[49]:


pdfReader = PyPDF2.PdfFileReader('/content/drive/MyDrive/webmining/webmining/contents/news.pdf')
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()
document


# PunktSentenceTokenizer adalah Sebuah tokenizer kalimat yang menggunakan algoritma tanpa pengawasan untuk membangun model untuk kata-kata singkatan, kolokasi, dan kata-kata yang memulai kalimat dan kemudian menggunakan model itu untuk menemukan batas kalimat.

# In[50]:


from nltk.tokenize.punkt import PunktSentenceTokenizer


# In[51]:


def tokenize(document):
    # Kita memecahnya menggunakan  PunktSentenceTokenizer
    doc_tokenizer = PunktSentenceTokenizer()
    # sentences_list adalah daftar masing masing kalimat dari dokumen yang ada.
    sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list


# In[52]:


sentences_list = tokenize(document)
sentences_list


# Merapikan data di atas sehingga lebih enak dibaca

# In[53]:


kal=1
for i in sentences_list:
    print('\nKalimat {}'.format(kal))
    kal+=1
    print(i)


# Tokenizing adalah proses pemisahan teks menjadi potongan-potongan yang disebut sebagai token untuk kemudian di analisa. Kata, angka, simbol, tanda baca dan entitas penting lainnya dapat dianggap sebagai token.

# In[54]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
vectorizer = CountVectorizer()
cv_matrix=vectorizer.fit_transform(sentences_list)


# Menampilkan jumlah Kosa Kata dari Data

# In[55]:


print ("Banyaknya kosa kata = ", len((vectorizer.get_feature_names_out())))


# Menampilkan jumlah Kalimat dari Data

# In[56]:


print ("Banyaknya kalimat = ", (len(sentences_list)))


# Menampilkan Kosa Kata dari Data

# In[57]:


print ("kosa kata = ", (vectorizer.get_feature_names_out()))


# In[58]:


# mengubah kumpulan dokumen mentah menjadi matriks fitur TF-IDF
normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())


# Menampilkan Jumlah Kalimat dan Kosa Kata

# In[59]:


normal_matrix.shape


# NetworkX adalah paket Python untuk pembuatan, manipulasi, dan studi tentang struktur, dinamika, dan fungsi jaringan yang kompleks. Ini menyediakan:

# In[60]:


import networkx as nx


# Graph adalah kumpulan dati titik (node) dan garis dimana pasangan – pasangan titik (node) tersebut dihubungkan oleh segmen garis. Node ini biasa disebut simpul (vertex) dan segmen garis disebut ruas (edge)

# In[61]:


res_graph = normal_matrix * normal_matrix.T
print(res_graph)


# In[62]:


nx_graph = nx.from_scipy_sparse_matrix(res_graph)


# In[63]:


nx.draw_circular(nx_graph)


# Jumlah Banyak Sisi 

# In[64]:


print('Banyaknya sisi {}'.format(nx_graph.number_of_edges()))


# Menkalikan data dengan data Transpose

# In[65]:


res_graph = normal_matrix * normal_matrix.T


# PageRank menghitung peringkat node dalam grafik G berdasarkan struktur tautan masuk. Awalnya dirancang sebagai algoritma untuk menentukan peringkat halaman web.

# In[66]:


ranks=nx.pagerank(nx_graph,)


# memasukkan data ke array

# In[67]:


arrRank=[]
for i in ranks:
    arrRank.append(ranks[i])


# menjadikan data kedalam bentuk tabel lalu digabungkan 

# In[68]:


dfRanks = pd.DataFrame(arrRank,columns=['PageRank'])
dfSentence = pd.DataFrame(sentences_list,columns=['News'])
dfJoin = pd.concat([dfSentence,dfRanks], axis=1)
dfJoin


# Mengurutkan data berdasarkan hasil tertinggi

# In[69]:


sortSentence=dfJoin.sort_values(by=['PageRank'],ascending=False)
sortSentence


# Menampilkan data dari 5 ke atas

# In[70]:


sortSentence.head(5)


# ## Latent Semantic Indexing(LSI) Topik Berita

# In[71]:


get_ipython().system('pip install nltk')


# In[72]:


get_ipython().system('pip install PySastrawi')


# In[73]:


get_ipython().system('pip install Sastrawi')


# In[74]:


import PyPDF2


# In[75]:


pdfReader = PyPDF2.PdfFileReader('/content/drive/MyDrive/webmining/webmining/contents/news.pdf')
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()
print(document)


# In[76]:


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')


# In[77]:


word_tokens = word_tokenize(document)
print(word_tokens)


# In[78]:


stop_words = set(stopwords.words('indonesian'))
word_tokens_no_stopwords = [w for w in word_tokens if not w in stop_words]
print(word_tokens_no_stopwords)


# In[79]:


import os
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[80]:


# Vectorize document using TF-IDF
tfidf = TfidfVectorizer(lowercase=True,
                        ngram_range = (1,1))

# Fit and Transform the documents
train_data = tfidf.fit_transform(word_tokens_no_stopwords)
train_data


# In[81]:


num_components=10

# Create SVD object
lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)

# Fit SVD model on data
lsa.fit_transform(train_data)

# Get Singular values and Components 
Sigma = lsa.singular_values_ 
V_transpose = lsa.components_.T
V_transpose


# In[82]:


# Print the topics with their terms
terms = tfidf.get_feature_names()

for index, component in enumerate(lsa.components_):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:5]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index+1)+": ",top_terms_list)


# ## 

# In[83]:


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

X = X_train
Y = y_train

# seed = 8
# kfold = model_selection.KFold(n_splits = 3,
# 					random_state = seed)

# initialize the base classifier
base_cls = DecisionTreeClassifier()

# no. of base classifier
num_trees = 500

# bagging classifier
model = BaggingClassifier(base_estimator = base_cls,
						n_estimators = num_trees)

results = model_selection.cross_val_score(model, X, Y)
print("accuracy :")
print(results.mean())


# In[84]:


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import pandas as pd

X = X_train
Y = y_train

# seed = 8
# kfold = model_selection.KFold(n_splits = 3,
# 					random_state = seed)

# initialize the base classifier
base_cls = SVC()

# no. of base classifier
num_trees = 500

# bagging classifier
model = BaggingClassifier(base_estimator = base_cls,
						n_estimators = num_trees)

results = model_selection.cross_val_score(model, X, Y)
print("accuracy :")
print(results.mean())


# In[156]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
# 'n_estimators': [i for i in range(800)],
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [50,100,200,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)


# In[160]:


CV_rfc.best_params_


# In[173]:


rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=8, criterion='entropy')
rfc1.fit(X_train, y_train)


# In[174]:


pred=rfc1.predict(X_test)
print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))

