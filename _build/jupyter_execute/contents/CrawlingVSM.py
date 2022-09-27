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


# configurasi Twint seperti halnya example di website library Twint

# In[8]:


c = twint.Config()
c.Search = '#ganjarpranowo'
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


# Baca data excel dataGanjar.xlsx yang telah diberi label yang telah simpan di Google Drive

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


# data['tweet'].apply(preprocessing).to_csv('preprocessing.csv')


# berikutnya jadikan data menjadi bentuk Matrik numerik

# In[18]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('/content/drive/MyDrive/webmining/webmining/contents/preprocessing.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])


# In[19]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[20]:


matrik_vsm[0]


# In[21]:


a=vectorizer.get_feature_names()


# selanjutnya data Matrik dikasih nama column sesuai Fitur

# In[22]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# lalu data fitur ditambahkan dengan label positif,netral dan negatif

# In[23]:


label = pd.read_excel('/content/drive/MyDrive/webmining/webmining/twint/dataGanjar.xlsx')
dj = pd.concat([dataTF.reset_index(), label["label"]], axis=1)
dj


# In[24]:


dj['label'].unique()


# In[25]:


dj.info()


# ## Penjelasan Scikit-learn

# Scikit-learn atau sklearn merupakan sebuah module dari bahasa pemrograman Python yang dibangun berdasarkan NumPy, SciPy, dan Matplotlib. Fungsi dari module ini adalah untuk membantu melakukan processing data ataupun melakukan training data untuk kebutuhan machine learning atau data science.

# install scikit-learn

# In[26]:


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

# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dj.drop(labels=['label'], axis=1),
    dj['label'],
    test_size=0.3,
    random_state=0)


# In[28]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info


# merangking fitur sesuai dengan fitur yang paling banyak keluar

# In[29]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False)


# menvisualkan data dengan grafik bar dengan urutan paling besar ke rendah

# In[30]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(50, 20))


# In[31]:


from sklearn.feature_selection import SelectKBest


# In[32]:


sel_five_cols = SelectKBest(mutual_info_classif, k=100)
sel_five_cols.fit(X_train, y_train)
X_train.columns[sel_five_cols.get_support()]


# In[33]:


X_train=X_train.values
y_train=y_train.values
X_test=X_test.values
y_test=y_test.values


# ## Penjelasan Naive Bayes

# Naive Bayes adalah algoritma machine learning yang digunakan untuk keperluan klasifikasi atau pengelompokan suatu data. Algoritma ini didasarkan pada teorema probabilitas yang dikenalkan oleh ilmuwan Inggris Thomas Bayes. Naive Bayes berfungsi memprediksi probabilitas di masa depan berdasarkan pengalaman sebelumnya, sehingga dapat digunakan untuk pengambilan keputusan.

# In[34]:


# from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
gauss = GaussianNB()
gauss.fit(X_train, y_train)
Y_pred = gauss.predict(X_test) 
Y_pred


# In[39]:


from sklearn.metrics import make_scorer, accuracy_score,precision_score
testing = gauss.predict(X_test) 
accuracy_gauss=round(accuracy_score(y_test,testing)* 100, 2)
accuracy_gauss


# ## Penjelasan Matplotib

# Matplotlib adalah library Python yang fokus pada visualisasi data seperti membuat plot grafik. Matplotlib pertama kali diciptakan oleh John D. Hunter dan sekarang telah dikelola oleh tim developer yang besar. Awalnya matplotlib dirancang untuk menghasilkan plot grafik yang sesuai pada publikasi jurnal atau artikel ilmiah. Matplotlib dapat digunakan dalam skrip Python, Python dan IPython shell, server aplikasi web, dan beberapa toolkit graphical user interface (GUI) lainnya.

# In[36]:


import matplotlib.pyplot as plt
from sklearn import metrics


# ## Penjelasan Confusion Matrix

# Confusion matrix juga sering disebut error matrix. Pada dasarnya confusion matrix memberikan informasi perbandingan hasil klasifikasi yang dilakukan oleh sistem (model) dengan hasil klasifikasi sebenarnya. Confusion matrix berbentuk tabel matriks yang menggambarkan kinerja model klasifikasi pada serangkaian data uji yang nilai sebenarnya diketahui.

# In[37]:


conf_matrix =metrics.confusion_matrix(y_true=y_test, y_pred=Y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['negatif', 'netral','positif'])
cm_display.plot()
plt.show()

