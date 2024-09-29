# -*- coding: utf-8 -*-
"""Data_preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Os4ZMMPxvV_N_b21_sy8956iqvf5wQ4U
"""

# !pip install rapidfuzz

# !pip install transformers

from transformers import AutoTokenizer
import torch

model_name = "aubmindlab/bert-base-arabertv02-twitter"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# from rapidfuzz.string_metric import levenshtein, normalized_levenshtein

from rapidfuzz.distance import Levenshtein
import rapidfuzz.fuzz as fuzz

import pandas as pd



i=2
domain = ["Iraqi", "Gulf", "Nile_Basin", "North_Africa"]

data = pd.read_csv("./data_selected_translated_filtered_{}".format(domain[i]), sep="\t")

data.columns

for i in domain:
    
    data = pd.read_csv("./data".format(i) , sep ="\t")
    print(i, data.shape)


len(data.source_lang[0].split())

data.source_lang[0]

tokens_length = len(tokenizer.tokenize(data.source_lang[0]))

import re
import unicodedata

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)

    s = re.sub(r"[^a-zA-Z؀-ۿ?.!,¿]+", " ", s)
    s = re.sub(r"([.،!?])", r"", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

normalizeString(data.source_lang[0])

len(tokenizer.tokenize(normalizeString(data.source_lang[0])))

tokens_length
data =[]

data =[]
domain = ["Iraqi", "Levantine", "Gulf", "Nile_Basin", "North_Africa"]
for i, _ in enumerate(domain):
    data.append(pd.read_csv("./data_selected_translated_filtered_{}".format(domain[i]), sep="\t"))
    
    data[i] = data[i][["source_lang_{}".format(domain[i]), "target_lang"]]



data_1 = data[0].merge( data[1], on ="target_lang", how="right")
data_2 = data[2].merge(data[3],on ="target_lang", how="right")


data_3 = data_1.merge(data_2,on ="target_lang", how="right")
data_4 = data_3.merge(data[4],on ="target_lang", how="right")
data_4.shape
data_5 = data_4.dropna()
len(data_5.columns)
data_5.to_csv("./data_selected_translated_filtered_all", sep="\t")
import pandas as pd
from sklearn.model_selection import train_test_split

domain = ["Iraqi", "Levantine", "Gulf", "Nile_Basin", "North_Africa"]
# data= pd.DataFrame(data ={"source_lang" : src, "target_lang" : trgt})

for i in domain:
    
    data = pd.read_csv("./data_{}_preprocessed".format(i) , sep ="\t")
    data_train, data_test = train_test_split(data, test_size = 0.2, shuffle = True, random_state=42)
    
    data_train, data_valid = train_test_split(data_train, test_size = data_test.shape[0]/data_train.shape[0] , shuffle = True, random_state=42)
    
    print(data.shape, data_train.shape, data_valid.shape, data_test.shape)
    data_train.to_csv("./data_{}_preprocessed_train".format(i), sep="\t")
    data_test.to_csv("./data_{}_preprocessed_test".format(i), sep="\t")
    data_valid.to_csv("./data_{}_preprocessed_valid".format(i), sep="\t")

