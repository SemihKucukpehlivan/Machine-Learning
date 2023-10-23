# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:50:30 2023

@author: Semih
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% import twitter data
# encoding= "latin1" latin harflerinin bulunduğu anlamına geliyor
data = pd.read_csv(r"gender_classifier.csv",encoding="latin1")
data = pd.concat([data.gender,data.description],axis=1)

#axis = 0 row olarak bul , dropna = nan değerleri drop et,
# inplace yazılmazsa 
# data = data.dropna(axis= 0)  
data.dropna(axis= 0,inplace=True)

#male ve female sınıfımız string ifade olduğu için onları 0 ve 1 lere eşitledik. 
data.gender = [1 if each == "female" else 0 for each in data.gender]

# %% cleanin data
# regular expression RE

import re 

first_description = data.description[4]
# ^ bu işaret küçük a dan küçük z ye A dan Z ye BULMA diyor. ^ olmazsa BUL demek
description = re.sub("[^a-zA-Z]"," ",first_description)

# pre process
description = description.lower() #buyuk harften küçük harfe çevirme

# %% stopwords (irrelavent words) gereksiz ifadeler(kelimeler)(the, and, as)

import nltk # naturel language tool kit

nltk.download("stopwords") #corpus diye bir klasöre indiriliyor.
from nltk.corpus import stopwords #corpus klasöründen import ediyorum

description = description.split() 

#split yerine tokenizer  kullanabiliriz
#description = nltk.word_tokenize(description)

#split kullanırsak "shouldn't gibi kelimeler ikiye ayrılmaz

# %%

description = [word for word in description if not word in set(stopwords("english"))]

# %%

import nltk as nlp

lemma  = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description]

 




















