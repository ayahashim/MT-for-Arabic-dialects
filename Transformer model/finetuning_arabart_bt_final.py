# -*- coding: utf-8 -*-
"""Finetuning_Arabart_BT_Final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N6EYIDNFmD4_jRsx5xqt4Pgk8MvcsIZk
"""

# !sudo echo -ne '\n' | sudo add-apt-repository ppa:alessandro-strada/ppa >/dev/null 2>&1 # note: >/dev/null 2>&1 is used to supress printing
# !sudo apt update >/dev/null 2>&1
# !sudo apt install google-drive-ocamlfuse >/dev/null 2>&1
# !google-drive-ocamlfuse
# !sudo apt-get install w3m >/dev/null 2>&1 # to act as web browser
# !xdg-settings set default-web-browser w3m.desktop >/dev/null 2>&1 # to set default browser
# %cd /content
# !mkdir gdrive
# %cd gdrive
# !mkdir "My Drive"
# !google-drive-ocamlfuse "/content/gdrive/My Drive"


import torch
from torch.utils.data import DataLoader, Dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoTokenizer, BartForConditionalGeneration, BartTokenizer
import pandas as pd
from timeit import default_timer as timer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import re
import unicodedata

# Assuming you have source and target texts as lists
domain = ["Gulf","Iraqi", "Levantine", "Nile_Basin", "North_Africa" ]
data_train =[]
for d in domain:
  data = pd.read_csv("../Data/data_{}_preprocessed_train".format(d), sep="\t")
  data_train.append(data)

data_1 = pd.concat([data_train[0], data_train[1]], ignore_index = True)
data_2 = pd.concat([data_train[2], data_train[3]], ignore_index = True)

data_train_src = pd.concat([data_1, data_2], ignore_index = True)
data_train_all =pd.concat([data_train_src, data_train[4]], ignore_index = True)

domain = ["Gulf","Iraqi", "Levantine", "Nile_Basin", "North_Africa" ]
data_valid =[]
for d in domain:
  data = pd.read_csv("../Data/data_{}_preprocessed_valid".format(d), sep="\t")
  data_valid.append(data)

data_1 = pd.concat([data_valid[0], data_valid[1]], ignore_index = True)
data_2 = pd.concat([data_valid[2], data_valid[3]], ignore_index = True)

data_valid_src = pd.concat([data_1, data_2], ignore_index = True)
data_valid_all =pd.concat([data_valid_src, data_valid[4]], ignore_index = True)

domain = ["Gulf","Iraqi", "Levantine", "Nile_Basin", "North_Africa" ]
data_test =[]
for d in domain:
  data = pd.read_csv("../Data/data_{}_preprocessed_test".format(d), sep="\t")
  data_test.append(data)

data_1 = pd.concat([data_test[0], data_test[1]], ignore_index = True)
data_2 = pd.concat([data_test[2], data_test[3]], ignore_index = True)

data_test_src = pd.concat([data_1, data_2], ignore_index = True)
data_test_all =pd.concat([data_test_src, data_test[4]], ignore_index = True)


device = "cuda" if torch.cuda.is_available() else "cpu"



# from adapters import AutoAdapterModel

# Load the tokenizer and model
model_name="moussaKam/AraBART"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

model.to(device)

import unicodedata
import re
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFKC', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)

    s = re.sub(r"[^a-zA-Z؀-ۿ?.!,¿]+", " ", s)
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Create a Dataset
class TranslationDataset(Dataset):
    def __init__(self, tokenizer, data_train, max_len=200):
        self.tokenizer = tokenizer
        self.source_texts = data_train["target_lang"]
        self.target_texts = data_train["source_lang"]

        self.max_len = max_len

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        inputs = self.tokenizer(normalizeString(source_text), return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
        targets = self.tokenizer(normalizeString(target_text), return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
        }

# Create DataLoader
i= 2
dataset = TranslationDataset(tokenizer, data_train[i])
data_loader = DataLoader(dataset, batch_size= 8)

#Create validation dataset
dataset_valid = TranslationDataset(tokenizer, data_valid[i])
data_loader_valid = DataLoader(dataset_valid, batch_size=8)

# Define optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
torch.cuda.empty_cache()

scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose = True)

data_valid[i].shape

torch.cuda.empty_cache()

# checkpoint= torch.load("./checkpoint_Arabart_{}_BT.pt".format(domain[i]))
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Training Loop
model.to(device)
best_valid_loss = 1000
early_stop_threshold = 10
best_epoch =-1
Epochs= 10
for epoch in range(Epochs):  # number of epochs
    losses=0
    start_time = timer()
    model.train()

    if epoch-best_epoch < early_stop_threshold:

      for batch in data_loader:
          # Move batch to device
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)

          # Forward pass
          outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss
          losses +=loss.item()
          # Backward pass
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      scheduler.step(losses/len(data_loader))

      end_time = timer()
      valid_losses =0

      valid_start_time = timer()
      model.eval()
      for batch in data_loader_valid:
          # Move batch to device
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)

          # Forward pass
          with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

          loss = outputs.loss
          valid_losses +=loss.item()

      valid_end_time = timer()
      print("Saving ...")
      torch.save({

              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),

              }, "./checkpoint_Arabart_{}_BT.pt".format(domain[i]))
      if (valid_losses < best_valid_loss):

        best_model = model
        best_epoch = epoch
        best_valid_loss = valid_losses
    else:
      break

    print(f"Epoch: {epoch+1}, Training Loss: {losses/ len(data_loader)} , ({(end_time-start_time)/ 60} m),\
            Validation Loss: {valid_losses/ len(data_loader_valid)} , ({(valid_end_time-valid_start_time)/ 60} m)")

def translate(model,tokenizer, data , n=None):
    targets =[]
    outputs =[]
    count=0
    for i in data.index:
        sentence= data["target_lang"][i]
        target= data["source_lang"][i]

        if count == n:
            break
        encoded_ar = tokenizer(normalizeString(sentence), return_tensors="pt").to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_ar,
            )
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        print("input: ", sentence)
        # print("target: ", normalizeString(target))
        print("output: ", output[0])
        # targets.append([normalizeString(target)])
        print("target: ", target)
        targets.append(target)
        outputs.append(output[0])
        count +=1


    return outputs, targets

checkpoint= torch.load("./checkpoint_Arabart_{}_BT.pt".format(domain[i]))
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

data_BT = pd.read_csv("./data_selected_translated_filtered", sep ="\t")

data_BT.head(100)

from nltk.translate.bleu_score import corpus_bleu

outputs, targets = translate(model,tokenizer, data_BT , )


data_BT["source_lang_{}".format(domain[i])] = outputs
data_BT.to_csv("../Data/data_selected_translated_filtered_{}".format(domain[i]), sep ="\t")

