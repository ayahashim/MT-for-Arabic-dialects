# -*- coding: utf-8 -*-

#This is the training for Sentence-AraBERT

from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import ( AutoModel, AutoTokenizer)
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
# from Data_preprocessing import data_preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel
# model_name = 'distilbert-base-uncased'

# model_name ="aubmindlab/bert-large-arabertv02"
# tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = 'CAMeL-Lab/bert-base-arabic-camelbert-da'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained(model_name)


# max_length= 128

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)

    s = re.sub(r"[^a-zA-Z؀-ۿ?.!,¿]+", " ", s)
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub("[a-zA-Z.,?؟]")
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class Arabert_dataset(Dataset):
  def __init__(self, data, tokenizer_arabert,tokenizer_sbert , max_length):
    self.data = data
    self.tokenizer_sbert = tokenizer_sbert
    self.tokenizer_arabert = tokenizer_arabert
    self.max_length = max_length

  def __len__(self):
    return (self.data.shape[0])

  def __getitem__(self, idx):
    text = self.data.target_lang[idx]
    text = normalizeString(text)
    inputs_arabert = self.tokenizer_arabert(
        text,
        add_special_tokens=True,
        return_tensors = "pt",
        max_length=self.max_length,
        padding='max_length',
        truncation=True
    )
    inputs_sbert = self.tokenizer_sbert(
        text,
        add_special_tokens=True,
        return_tensors = "pt",
        max_length=self.max_length,
        padding='max_length',
        truncation=True
    )

    # dataset = TensorDataset(torch.tensor(input_ids, dtype = torch.long).to(device), torch.tensor(attention_mask, dtype = torch.long).to(device))
    return inputs_arabert, inputs_sbert

import pandas as pd
data_select = pd.read_csv("./data_selected_translated_filtered_all", sep="\t")
domain = ["Gulf","Iraqi", "Levantine", "Nile_Basin", "North_Africa" ]

data_train = []

for i in range(len(domain)):
  data_train.append(pd.read_csv("./data_{}_preprocessed_train".format(domain[i]), sep="\t"))
data_1 = pd.concat([data_train[0], data_train[1]], ignore_index = True)
data_2 = pd.concat([data_train[2], data_train[3]], ignore_index = True)

data_train_src = pd.concat([data_1, data_2], ignore_index = True)
data_train_all =pd.concat([data_train_src, data_train[4]], ignore_index = True)

data_select.shape

data_train_all = data_train_all[["target_lang"]]
data_select = data_select[["target_lang"]]
data = pd.concat([data_train_all, data_select[:20000]], ignore_index= True)

data.shape



data.dropna(inplace = True)
data.drop_duplicates(inplace = True)
data.reset_index(inplace = True)
print(data.shape)

data.head()

# import re
# import pandas as pd
# for idx in data.index:
#   sent = data.loc[idx, "target_lang"]
#   sent = re.findall(r"\d+", sent)
#   if sent:
#     print(sent)
#     data.drop(idx, inplace = True)

# data.to_csv("./data_selected_translated_filtered", sep="\t")

data.head(-10)

data.target_lang



from transformers import AutoModel, AutoTokenizer
import torch
from peft import get_peft_model
from peft import LoraConfig, TaskType

# Load Sentence-BERT
model_sbert = AutoModel.from_pretrained("sentence-transformers/distiluse-base-multilingual-cased-v2")
tokenizer_sbert = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Load Arabert_model
model_arabert = AutoModel.from_pretrained('aubmindlab/bert-large-arabertv02')
tokenizer_arabert = AutoTokenizer.from_pretrained('aubmindlab/bert-large-arabertv02')
lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=1, lora_dropout=0.01)
model_arabert_lora = get_peft_model(model_arabert, lora_config)
# data.reset_index(inplace = True)
dataset = Arabert_dataset(data, tokenizer_arabert, tokenizer_sbert, max_length=200)
train_sampler = RandomSampler(dataset)
arabert_dataloader = DataLoader(dataset,sampler= train_sampler , batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse_loss = nn.MSELoss()
model_arabert_lora.to(device)
model_sbert.to(device)
mse_loss.to(device)
optimizer = torch.optim.Adam(model_arabert.parameters(), lr=1e-4,)

torch.cuda.empty_cache()

# checkpoint = torch.load("./checkpoint_sent_arabert.pt")
# model_arabert_lora.load_state_dict(checkpoint["model_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_dict"])
# loss = checkpoint["loss"]

from tqdm.auto import tqdm
import time
epochs =10

for epoch in tqdm(range(epochs)):
  total_loss =0
  model_arabert_lora.train()
  model_sbert.eval()
  for i, batch in tqdm(enumerate(arabert_dataloader)):
    start_time = time.time()
    inputs_arabert, inputs_sbert = batch

    optimizer.zero_grad()

    output_arabert = model_arabert_lora(inputs_arabert["input_ids"].squeeze(1).to(device),
                                   attention_mask=inputs_arabert["attention_mask"].squeeze(1).to(device))
    embeddings_arabert = output_arabert.last_hidden_state.mean(1)
    embeddings_arabert = embeddings_arabert[:, :768]
    with torch.no_grad():
      output_sbert = model_sbert(inputs_sbert["input_ids"].squeeze(1).to(device),
                                   attention_mask=inputs_sbert["attention_mask"].squeeze(1).to(device))
      embeddings_sbert = output_sbert.last_hidden_state.mean(1)

    loss = mse_loss(embeddings_arabert, embeddings_sbert)

    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print("saving")
  torch.save({"model_dict" : model_arabert_lora.state_dict(),
            "optimizer_dict" : optimizer.state_dict(),
            "loss" : loss},"./checkpoint_sent_arabert_large_teacher_student.pt")
  end_time =  time.time() - start_time
  print("Epoch processing time is {}".format(end_time))
  print("Total loss is {} in epoch {}".format(total_loss/ len(arabert_dataloader), epoch))
