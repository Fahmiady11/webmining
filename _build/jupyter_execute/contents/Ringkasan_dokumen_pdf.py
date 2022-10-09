#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining')


# In[2]:


get_ipython().system('pip install PyPDF2')


# In[3]:


get_ipython().system('pip install docx2txt')


# In[4]:


import numpy as np
import PyPDF2
import docx2txt
import sys


# In[5]:


name = input('Masukkan nama file: ') 
print('Anda telah memanggil dokument  {}'.format(name))


# In[ ]:


pdfFileObj = open(name, 'rb')


# In[ ]:


pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
document = pageObj.extractText()


# In[ ]:


from nltk.tokenize.punkt import PunktSentenceTokenizer


# In[ ]:


document


# In[ ]:


def tokenize(document):
    # Kita memecahnya menggunakan  PunktSentenceTokenizer
    # 
    doc_tokenizer = PunktSentenceTokenizer()
    
    # metode tokenize() memanggil dokument kita
    # sebagai input dan menghasilkan daftar kalimat dalam dokumen
    
    # sentences_list adalah daftar masing masing kalimat dari dokumen yang ada.
    sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list


# In[ ]:


sentences_list = tokenize(document)


# In[ ]:


for i in sentences_list:
    print('------')
    print(i)


# In[ ]:


print ("Banyaknya kosa kata = ", len((cv.get_feature_names_out())))


# In[ ]:


print ("Banyaknya kalimat = ", (len(sentences_list)))


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# In[ ]:


cv = CountVectorizer()
cv_matrix = cv.fit_transform(sentences_list)


# In[ ]:


print ("Banyaknya kosa kata = ", len((cv.get_feature_names_out())))


# In[ ]:


print ("kosa kata = ", (cv.get_feature_names_out()))


# In[ ]:


print(cv_matrix)


# In[ ]:


normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())


# In[ ]:


import networkx as nx


# In[ ]:


print(normal_matrix.T.toarray)
res_graph = normal_matrix * normal_matrix.T


# In[ ]:


nx_graph = nx.from_scipy_sparse_matrix(res_graph)


# In[ ]:


nx.draw_circular(nx_graph)


# In[ ]:


print('Banyaknya sisi {}'.format(nx_graph.number_of_edges()))


# In[ ]:


res_graph = normal_matrix * normal_matrix.T


# In[ ]:


normal_matrix.shape


# In[ ]:


import scipy
A=scipy.sparse.csr_matrix.toarray(normal_matrix)


# In[ ]:


cv_matrix


# In[ ]:


print(sentences_list[1])


# In[ ]:


ranks = nx.pagerank(nx_graph)


# In[ ]:


for i in ranks:
    print(i, ranks[i])


# In[ ]:


print(res_graph)

