# -*- coding: utf-8 -*-

from io import open
import unicodedata
import re
import random
# from arabert.preprocess import ArabertPreprocessor
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import ( AutoModel, AutoTokenizer)
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel

model_name ="aubmindlab/bert-large-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, do_lower_case=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

SOS_token = tokenizer.cls_token_id
EOS_token = tokenizer.sep_token_id
PAD_token = tokenizer.pad_token_id

import pandas as pd
data_train_src = pd.read_csv("./data_Nile_Basin_preprocessed_train" , sep="\t")
data_train_trgt= pd.read_csv("./data_selected_translated", sep="\t")
data_train_src["label"] = [1 for _ in range(data_train_src.shape[0])]
data_train_trgt["label"] = [0 for _ in range(data_train_trgt.shape[0])]
data_train_src.drop(columns = ["source_lang"], inplace = True)
data_train_trgt = data_train_trgt[["target_lang", "label"]]
data = pd.read_csv("./domain_finetune_data", sep="\t")







data_valid_src = pd.read_csv("./data_Nile_Basin_preprocessed_valid" , sep="\t")
data_valid_src["label"] = [1 for _ in range(data_valid_src.shape[0])]
data_trgt_test = pd.read_csv("./domain_finetune_data_test", sep="\t")
data_trgt_test_1 = data_trgt_test[3000:]
data_train_trgt_valid = data_trgt_test[:3000]
data_train_trgt_valid = data_train_trgt_valid[["target_lang", "label"]]
data_valid_src =  data_valid_src[["target_lang", "label"]]
data_valid = pd.concat([data_train_trgt_valid, data_valid_src], ignore_index = True)

max_len =200
from transformers import BertTokenizer, BertModel
from torch.autograd import Function

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        review = normalizeString(self.df.target_lang[index])
        label = self.df.label[index]


        encoded_input = self.tokenizer(
                review,
                add_special_tokens=True,
                max_length= max_len,
                pad_to_max_length=True,
                truncation=True,
                 return_token_type_ids=True
            )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        token_type_ids = encoded_input["token_type_ids"] if "token_type_ids" in encoded_input else None



        data_input = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "label" : torch.tensor(label, dtype=torch.long)

        }

        return data_input["input_ids"].to(device), data_input["attention_mask"].to(device), data_input["token_type_ids"].to(device) , data_input["label"].to(device)



    def __len__(self):
        return self.df.shape[0]

    def __len__(self):
        return self.df.shape[0]

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

from peft import get_peft_model
from peft import LoraConfig, TaskType
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained('aubmindlab/bert-large-arabertv02')
        self.lora_config = LoraConfig(
           task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1)

        self.model = get_peft_model(self.bert, self.lora_config)
        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1),
        )


    def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          labels=None,
          # grl_lambda = 1.0

          ):

        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
            )
        hidden_state = outputs[0]
        pooled_output = hidden_state.mean(dim=1)

        domain_pred = self.domain_classifier(pooled_output)

        return domain_pred.to(device)

def train_epoch(train_dataloader,valid_dataloader,
                   model,domain_optimizer, scheduler,
                   loss_fn_domain_classifier , epoch, n_epochs):

    total_loss = 0
    total_valid_loss = 0
    train_iterator =iter(train_dataloader)

    valid_iterator = iter(valid_dataloader)
    model.train()
    for idx in range(int(len(train_dataloader))):


        input_ids, attention_mask, token_type_ids, label= next(train_iterator)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids" : token_type_ids,
        }
        domain_optimizer.zero_grad()
        domain_pred = model(**inputs)

        loss= loss_fn_domain_classifier(domain_pred, label)

        loss.backward()
        domain_optimizer.step()

        total_loss += loss.item()

    model.eval()
    for idx in range(int(len(valid_dataloader))):

        input_ids, attention_mask, token_type_ids, label= next(valid_iterator)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids" : token_type_ids
        }

        with torch.no_grad():
          domain_pred = model(**inputs)

        valid_loss= loss_fn_domain_classifier(domain_pred, label)

        total_valid_loss += valid_loss.item()
    scheduler.step(total_valid_loss)

    return total_loss / len(train_dataloader), total_valid_loss / len(valid_dataloader), model,domain_optimizer

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def train(train_dataloader,valid_dataloader,
         model ,domain_optimizer, scheduler, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    best_loss = 10
    threshold = 5
    loss_fn_domain_classifier = torch.nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):

        print_loss_total = 0
        loss, valid_loss,model,domain_optimizer = train_epoch(train_dataloader,valid_dataloader,
                                                              model,domain_optimizer,scheduler,
                                                              loss_fn_domain_classifier , epoch, n_epochs)
        torch.save({
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': domain_optimizer.state_dict(),
          },"./checkpoint_domain_fintune_arabert.pt")

        print_loss_total += loss

        print('%s (%d %d%%) %.4f ' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, loss),
              "valid loss is:",  valid_loss)

torch.cuda.empty_cache()

from torch.optim import lr_scheduler
hidden_size = 256
learning_rate = 0.001
Load = True

model = DomainClassifier()
model.to(device)
domain_optimizer = optim.Adam(model.parameters(), lr= learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(domain_optimizer, mode='min', factor=0.1, patience=2)

model.model.print_trainable_parameters()

batch_size = 16
train_dataset = ReviewDataset(data)
train_dataloader= DataLoader(train_dataset, batch_size=batch_size,  shuffle= True)
valid_dataset = ReviewDataset(data_valid)
valid_dataloader= DataLoader(valid_dataset, batch_size=batch_size,  shuffle= True)

torch.cuda.empty_cache()

data.head(-10)

# train(train_dataloader,valid_dataloader,model,domain_optimizer,scheduler, 10, print_every=1, plot_every=5)
Load = True
if Load:
    checkpoint= torch.load("./checkpoint_domain_fintune_arabert_LORA.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    domain_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

def prediction(model, data_train):

  with torch.no_grad():

    sentences = []
    preds = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_train["prediction"] = [0 for _ in range(data_train.shape[0])]
    data_train["score"] = [0 for _ in range(data_train.shape[0])]
    for idx in data_train.index:
      sent = normalizeString(data_train.target_lang[idx])
      label = data_train.label[idx]
      encoded_input = tokenizer(
                sent,
                add_special_tokens=True,
                max_length= max_len,
                pad_to_max_length=True,
                truncation=True,
                 return_token_type_ids=True
            )
      input_ids = encoded_input["input_ids"]
      attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None
      token_type_ids = encoded_input["token_type_ids"] if "token_type_ids" in encoded_input else None

      src_inputs = {
          "input_ids": torch.tensor(input_ids).unsqueeze(0).to(device),
          "attention_mask": torch.tensor(attention_mask).unsqueeze(0).to(device),
          "token_type_ids" :torch.tensor(token_type_ids).unsqueeze(0).to(device),
          # "label" : torch.tensor(label, dtype = torch.long)

      }
      prob = model(**src_inputs)
      prob = F.softmax(prob, dim =-1)
      domain_pred, idx_pred = torch.max(prob, dim =-1)
      sentences.append(sent)
      preds.append(idx_pred.item())
      print(idx_pred, idx)
      # print( domain_pred.item(), idx_pred.item())
      data_train["prediction"][idx] = idx_pred.item()
      data_train["score"][idx] = domain_pred.item()

    return sentences, preds, data_train

sentences, preds, data_train_pred = prediction(model, data_trgt_test_1)

