
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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from timeit import default_timer as timer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "aubmindlab/bert-base-arabertv02-twitter"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

SOS_token = tokenizer.cls_token_id
EOS_token = tokenizer.sep_token_id
PAD_token = tokenizer.pad_token_id

domain = ["Gulf","Iraqi", "Levantine", "Nile_Basin", "North_Africa" ]
data_train =[]
for d in domain:
  data = pd.read_csv("../Data/data_{}_preprocessed_train".format(d), sep="\t")
  data_train.append(data)

data_1 = pd.concat([data_train[0], data_train[1]], ignore_index = True)
data_2 = pd.concat([data_train[2], data_train[3]], ignore_index = True)

data_train_src = pd.concat([data_1, data_2], ignore_index = True)
data_train_all =pd.concat([data_train_src, data_train[4]], ignore_index = True)
data_train_trgt = data_train[4]

domain = ["Gulf","Iraqi", "Levantine", "Nile_Basin", "North_Africa" ]
data_test =[]
for d in domain:
  data = pd.read_csv("../Data/data_{}_preprocessed_test".format(d), sep="\t")
  data_test.append(data)

data_1 = pd.concat([data_test[0], data_test[1]], ignore_index = True)
data_2 = pd.concat([data_test[2], data_test[3]], ignore_index = True)

data_test_src = pd.concat([data_1, data_2], ignore_index = True)
data_test_all =pd.concat([data_test_src, data_test[4]], ignore_index = True)

domain = ["Gulf","Iraqi", "Levantine", "Nile_Basin", "North_Africa" ]
data_valid =[]
for d in domain:
  data = pd.read_csv("../Data/data_{}_preprocessed_valid".format(d), sep="\t")
  data_valid.append(data)

data_1 = pd.concat([data_valid[0], data_valid[1]], ignore_index = True)
data_2 = pd.concat([data_valid[2], data_valid[3]], ignore_index = True)

data_valid_src = pd.concat([data_1, data_2], ignore_index = True)
data_valid_all =pd.concat([data_valid_src, data_valid[4]], ignore_index = True)
data_valid_trgt = data_valid[4]





data = pd.read_csv("../Data/data_domian_cosine_arabert_all", sep ="\t")
data_train_1 =[]

for i in range(5):
    data_train_1.append(data.sort_values("score_{}".format(domain[i]), ascending = False))
    data_train_1[i] = data_train_1[i][["source_lang_{}".format(domain[i]), "target_lang"]]
    data_train_1[i].rename(columns = {"source_lang_{}".format(domain[i]):"source_lang"}, inplace = True)




data_all_1 = pd.concat([data_train_1[0][:5000], data_train_1[1][:5000]], ignore_index = True)
data_all_2 = pd.concat([data_train_1[2][:5000], data_train_1[3][:5000]], ignore_index = True)
data_all_3 = pd.concat([data_all_2, data_all_1], ignore_index = True)
data_all_4 = pd.concat([data_all_3, data_train_1[4][:5000]], ignore_index = True)


# data_all_4.drop_duplicates(inplace = True)
# data_all_4.shape
# data_train_all_BT_src  = pd.concat([data_train_src, data_all_3], ignore_index = True)
data_train_all_BT = pd.concat([data_train_src, data_all_3], ignore_index = True)
print(data_all_3.shape)
data_all_3
data_train_src
data_train_all_BT.to_csv("./source_domain_BT", columns = ["target_lang", "source_lang"], index = False )
data_trgt_all_BT = pd.concat([data_train[4], data_all_4], ignore_index = True)

data_1 = pd.concat([data_valid_all, data_train_all], ignore_index = True)
data_all = pd.concat([data_test_all, data_1], ignore_index = True)

data_vocab_all_BT = pd.concat([data_all, data_all_4], ignore_index = True)
# data_vocab_all_BT = pd.concat([data_vocab_all_1, data_all_4], ignore_index = True)
data_vocab_all_BT.shape


MAX_LENGTH = 200
class Lang:
    def __init__(self, name):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.word2index = { self.tokenizer.pad_token : self.tokenizer.pad_token_id}
        self.word2count = {}
        self.index2word = {self.tokenizer.pad_token_id: self.tokenizer.pad_token}
        self.n_words = 1 # Count PAD token

    def addSentence(self, sentence):
        for word in self.tokenizer.tokenize(sentence, add_special_tokens= True):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.tokenizer.convert_tokens_to_ids(word)
            self.word2count[word] = 1
            self.index2word[self.tokenizer.convert_tokens_to_ids(word)] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
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

def readLangs(lang1, lang2, reverse=False, label ="train"):
    print("Reading lines...")

    # Read the file and split into lines
    if label=="vocab":
        data = data_vocab_all_BT
    if label =="train":
        data = data_train_all_BT
    if label =="valid":
        data = data_valid_trgt
    if label =="trgt":
        data = data_trgt_all_BT


    # Split every line into pairs and normalize
    pairs = [[normalizeString(data.source_lang[idx]), normalizeString(data.target_lang[idx])] for idx in data.index]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False, label="train"):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, label)
    print("Read %s sentence pairs" % len(pairs))
    # pairs = pairs[1:]
    # print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        input_lang.addSentence(pair[1])
    output_lang = input_lang
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs



# input_lang, output_lang, pairs = prepareData('Cairo', 'MSA')



def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word,0) for word in tokenizer.tokenize(sentence, add_special_tokens=True)]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size, label ="train"):
    input_lang, output_lang, _ = prepareData('ar', 'arz', label="vocab")
    _, _, pairs = prepareData('ar', 'arz', label = label)
    # pairs = pairs[1:]
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

from torch.autograd import Function
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainAdaptationModel(nn.Module):
    def __init__(self,embed_size= 512, hidden_size = 256, dropout=0.1):
        super(DomainAdaptationModel, self).__init__()



        self.dropout = nn.Dropout(dropout)

        self.domain_classifier = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
            # nn.LogSoftmax(dim=1),
        )


    def forward(
          self,
         pooled_output,
          grl_lambda = 1.0,
          ):
        pooled_output = self.dropout(pooled_output)


        reversed_pooled_output = GradientReversalFn.apply(pooled_output, grl_lambda)


        domain_pred = self.domain_classifier(reversed_pooled_output)

        return domain_pred.to(device)

class Disc_model(nn.Module):
    def __init__(self, hidden_size=256, input_size=512, dropout =0.1):
        super(Disc_model, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_outputs, label ="trgt"):

        h = self.tanh(self.linear(self.dropout(enc_outputs)))#(N,hid_dim)
        z = self.fc(h) #(N,2)

        # prob = self.logsoftmax(z)


        return z

# input_lang, output_lang, train_dataloader = get_dataloader(32)

# data = iter(train_dataloader)
# next(data)[0]

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den) # is an indexing operation that targets every other
                                                      #column of the pos_embedding tensor, starting from
                                                      #the second column (index 1) and going up to the last
                                                      #column (index emb_size-1).
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask, src_key_padding_mask = src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask, tgt_key_padding_mask = tgt_padding_mask )

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

PAD_IDX = 0
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

PAD_IDX=0
def train_epoch(transformer,src_dataloader,trgt_dataloader,valid_dataloader, optimizer, classifier,domain_optimizer ):
  # if (epoch-1) % int(len(src_dataloader)/ len(trgt_dataloader)) == 0:
  dataset_iter = iter(src_dataloader)
  transformer.train()
  classifier.train()

  total_loss = 0
  total_loss_domain =0
  loss_NMT =0
  trgt_iter = iter(trgt_dataloader)
  for idx in range(len(src_dataloader)):
    input_tensor,target_tensor = next(dataset_iter)

    inp_len = [ len(input_tensor[i][input_tensor[i]!=0]) for i in range(input_tensor.shape[0])]
    trgt_len = [len(target_tensor[i][target_tensor[i]!=0]) for i in range(target_tensor.shape[0]) ]
    input_tensor = input_tensor[:, :max(inp_len)]
    target_tensor = target_tensor[:, :max(trgt_len)]
    input_tensor = input_tensor.T.to(device) #(L,N)
    target_tensor = target_tensor.T.to(device)

    try:
      da_inp,_ = next(trgt_iter)
    except Exception as e:
      trgt_iter = iter(trgt_dataloader)
      da_inp,_ = next(trgt_iter)


    trgt_da_len = [ len(da_inp[i][da_inp[i]!=0]) for i in range(da_inp.shape[0])]
    da_inp_tensor = da_inp[:, :max(trgt_da_len)] #(N,L)
    da_inp_tensor = da_inp_tensor.T.to(device)


    tgt_inp = target_tensor[:-1, :]
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(input_tensor, tgt_inp)

    encoder_src_domain = transformer.encode(input_tensor, src_mask, src_padding_mask ) #(L,N,embed)
    decoder_src_domain = transformer.decode(tgt_inp, encoder_src_domain, tgt_mask, tgt_padding_mask)
    logits = transformer.generator(decoder_src_domain)

    domain_optimizer.zero_grad()
    src_out_domain = encoder_src_domain.mean(axis=0).to(device) #(N,E)




    y_s_domain = torch.zeros(src_out_domain.shape[0], dtype=torch.long).to(device)


    da_mask = torch.zeros((da_inp_tensor.shape[0], da_inp_tensor.shape[0]),device=DEVICE).type(torch.bool)

    da_padding_mask = (da_inp_tensor == PAD_IDX).transpose(0, 1)
    da_inp_domain = transformer.encode(da_inp_tensor, da_mask, da_padding_mask )

    da_out_domain = da_inp_domain.mean(axis=0).to(device)



    y_t_domain = torch.zeros(da_out_domain.shape[0], dtype=torch.long).to(device)
    y_domain = torch.cat((y_s_domain, y_t_domain))
    domain_out = torch.cat((src_out_domain, da_out_domain))
    domain_pred = classifier(domain_out)

    lprobs = F.log_softmax(domain_pred, dim=-1, dtype=torch.float32)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    loss_domain = F.nll_loss(lprobs, y_domain.view(-1))



    optimizer.zero_grad()
    tgt_out = target_tensor[1:, :]
    # logits = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
    loss_NMT +=loss
    loss = loss+ loss_domain

    loss.backward()

    optimizer.step()
    domain_optimizer.step()

    total_loss += loss.item()
    total_loss_domain += loss_domain
  print(total_loss_domain.item()/len(src_dataloader) ,  loss_NMT.item()/len(src_dataloader))
  scheduler.step(total_loss / len((src_dataloader)))
  scheduler_domain.step(total_loss / len(src_dataloader))
  train_end_time = timer()

  transformer.eval()
  valid_losses=0
  valid_start_time = timer()
  for src, tgt in valid_dataloader:
      # Move batch to device
      src = src.to(device)
      tgt = tgt.to(device)

      inp_len = [ len(src[i][src[i]!=0]) for i in range(src.shape[0])]
      trgt_len = [len(tgt[i][tgt[i]!=0]) for i in range(tgt.shape[0]) ]
      input_tensor = src[:, :max(inp_len)]
      target_tensor = tgt[:, :max(trgt_len)]
      src = input_tensor.T.to(device)
      tgt = target_tensor.T.to(device)
      tgt_input = tgt[:-1, :]

      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
      with torch.no_grad():
        logits = transformer(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)



      tgt_out = tgt[1:, :]
      loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

      valid_losses +=loss.item()
  valid_end_time = timer()

  #scheduler.step()


  return total_loss / len(list(src_dataloader)), train_end_time, valid_losses/ len(valid_dataloader), valid_start_time, valid_end_time, loss_NMT.item()/len(src_dataloader)


torch.cuda.empty_cache()
i=0
model ={}
best_loss = - np.inf



batch_size = 16

input_lang, output_lang, src_dataloader = get_dataloader(batch_size= batch_size)
input_lang, output_lang, trgt_dataloader = get_dataloader(batch_size= batch_size, label ="trgt")

input_lang, output_lang, valid_dataloader = get_dataloader(batch_size= batch_size, label ="valid")

# checkpoint = torch.load("./checkpoint_transformer_DC.pt")

# lora_model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# checkpoint = torch.load("./checkpoint_transformer_classifier_DC.pt")

# model.load_state_dict(checkpoint['model_state_dict'])
# domain_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


torch.cuda.empty_cache()



for z in [0,1,2]:
    torch.manual_seed(z)
    SRC_VOCAB_SIZE = tokenizer.vocab_size
    TGT_VOCAB_SIZE = tokenizer.vocab_size
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 1024
    
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    
    # input_lang, output_lang, src_dataloader = get_dataloader(batch_size= batch_size)
    # input_lang, output_lang, trgt_dataloader = get_dataloader(batch_size= batch_size, label ="trgt")
    
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                             NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
    loss_fn_domain_classifier = nn.CrossEntropyLoss()
    # model = DomainAdaptationModel(EMB_SIZE, 256).to(DEVICE) #ADC model
    model = Disc_model(hidden_size = 256, input_size =EMB_SIZE ).to(DEVICE)
    domain_optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose = True)
    scheduler_domain = ReduceLROnPlateau(domain_optimizer, patience=1, verbose = True)
    NUM_EPOCHS = 100
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 =NUM_EPOCHS , T_mult=1, verbose = True)
    torch.cuda.empty_cache()
    
    experiments = 3
    
    checkpoint = torch.load("./checkpoint_transformer_UDA_sorce_domain_paper_BT5000_FF1024_AH8_EDL2_{}.pt".format(z))
    
    transformer.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    from transformers import AutoModelForSeq2SeqLM
    from peft import LoraModel, LoraConfig
    
    config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r= 64,
        # lora_alpha= 2,
        target_modules=["out_proj", "k_proj","v_proj","q_proj", "embedding"],
        lora_dropout=0.01,
    )
    
    print(transformer)
    lora_model = LoraModel(transformer, config, "default")
    # lora_model = transformer
    optimizer = optim.Adam(lora_model.parameters(), lr=0.0001)
    print(lora_model)
    
    #training with r= 4
    NUM_EPOCHS = 50
    best_valid_loss = 10
    early_stop_threshold = 3
    best_epoch =-1
    j=0
    for epoch in range(1, NUM_EPOCHS+1):
        start_time = timer()
    
        if j < early_stop_threshold:
          train_loss,  train_end_time, valid_loss, valid_start_time, valid_end_time,loss_NMT = train_epoch(lora_model, src_dataloader, trgt_dataloader, valid_dataloader, optimizer, model,domain_optimizer )
          if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("Saving ...")
            torch.save({
    
            'model_state_dict': lora_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    
            }, "./checkpoint_transformer_UDA_DC_BT5000_paper_FF1024_AH8_EDL2_{}.pt".format(z))
            torch.save({
    
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': domain_optimizer.state_dict(),
    
            }, "./checkpoint_classifier_UDA_DC_BT5000_paper_FF1024_AH8_EDL2_{}.pt".format(z))
            j =0
          else:
    
            j +=1
        else:
          break
        print((f"Epoch: {epoch}, Train loss: {train_loss:.9f}, "f"Epoch time = {(train_end_time - start_time)/60:.3f}m"))
        print((f"Epoch: {epoch}, valid loss: {valid_loss:.9f}, "f"Epoch time = {(valid_end_time - valid_start_time)/60:.3f}m"))
    




def greedy_decode(model, src, src_mask,src_padding_mask, max_len, start_symbol, output_lang):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    decoded_words = []
    attn=[]
    with torch.no_grad():
        memory = model.encode(src, src_mask,src_padding_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        tgt_padding_mask = (ys == PAD_IDX).transpose(0, 1)
        out = model.decode(ys, memory, tgt_mask,tgt_padding_mask )
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
      
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        decoded_words.append(output_lang.index2word.get(next_word,output_lang.index2word[1]))
     
        if next_word == EOS_token:
            break
    return decoded_words, ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, data, input_lang, output_lang, n=None):
    targets =[]
    outputs =[]
    model.eval()
    i=0
    for idx in data.index:
      src_sentence = data.source_lang[idx]
      trgt = data.target_lang[idx]
      if i == n:
          break

      src = tensorFromSentence(input_lang, normalizeString(src_sentence)).view(-1, 1)
      num_tokens = src.shape[0]
      src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
      src_padding_mask = (src == PAD_IDX).transpose(0, 1)

      decoded_words, _ = greedy_decode(
          model,  src, src_mask,src_padding_mask,max_len=num_tokens + 5, start_symbol=SOS_token, output_lang=output_lang)
      output_words = tokenizer.convert_tokens_to_string(decoded_words)
      output_words = output_words.replace("[SEP]", "")
      output_words = output_words.replace("[UNK]", "")
      # print("input: ", src_sentence)
      # print("target: ", trgt)
      # print("prediction: ", output_words)
      # print('')
      i=i+1
      targets.append([normalizeString(trgt)])
      outputs.append(output_words)
    return outputs, targets


from nltk.translate.bleu_score import corpus_bleu



for i in [1]:
    bleu={
        "Gulf": [],
            "Iraqi": [],
            "Levantine":[],
            "Nile_Basin":[],
            "North_Africa":[]
          }
    
    # model_path_list = ["./checkpoint_transformer_{}_{}.pt".format(domain[i],x) for x in [0,1,2]]
    model_path_list = ["./checkpoint_transformer_UDA_DC_BT5000_paper_FF1024_AH8_EDL2_{}.pt".format(x)for x in [0,1,2]]
   
    
   
    # checkpoint = torch.load("./checkpoint_transformer_finetune_arabert_large_MADAR_2_{}_BT_4_2_{}.pt".format(domain[i],z))
    z=0
    
    # model_path_list= ["checkpoint_transformer_Gulf.pt"]
    for path in model_path_list:
        
        torch.manual_seed(z)
        z +=1
        transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        transformer = transformer.to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
        checkpoint = torch.load(path)
        config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r= 64,
            # lora_alpha= 2,
            target_modules=["out_proj", "k_proj","v_proj","q_proj", "embedding"],
            lora_dropout=0.01,
        )
        
       
        lora_model = LoraModel(transformer, config, "default")
        print("loading")
        lora_model.load_state_dict(checkpoint['model_state_dict'])
    
    
        for j, (k, v) in enumerate(bleu.items()):
        
            data = data_test[j]
            outputs, target= translate(lora_model, data, input_lang, output_lang)
            src_test = corpus_bleu(target, outputs)
            bleu[k].append(src_test)
            print(bleu)
            src_test

