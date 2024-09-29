
import torch
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

# Load data
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

data_1 = pd.concat([data_valid_all, data_train_all], ignore_index = True)
data_all = pd.concat([data_test_all, data_1], ignore_index = True)

data_vocab = []

data_vocab_all = data_all

# Load selected data from the monolingual corpus using domain cosine with arabert.
data = pd.read_csv("../Data/data_domian_cosine_arabert_all", sep ="\t")
data_train_1 =[]

for i in range(5):
    data_train_1.append(data.sort_values("score_{}".format(domain[i]), ascending = False))
    data_train_1[i] = data_train_1[i][["source_lang_{}".format(domain[i]), "target_lang"]]
    data_train_1[i].rename(columns = {"source_lang_{}".format(domain[i]):"source_lang"}, inplace = True)

data_all_1 = pd.concat([data_train_1[0][:20000], data_train_1[1][:20000]], ignore_index = True)
data_all_2 = pd.concat([data_train_1[2][:20000], data_train_1[3][:20000]], ignore_index = True)
data_all_3 = pd.concat([data_all_2, data_all_1], ignore_index = True)
data_all_4 = pd.concat([data_all_3, data_train_1[4][:20000]], ignore_index = True)


data_train_all_BT = pd.concat([data_train_all, data_all_4], ignore_index = True)
print(data_train_all.shape, data_train_all_BT.shape)

data_train_all_BT_src  = pd.concat([data_train_src, data_all_3], ignore_index = True)
data_vocab_all_BT = pd.concat([data_vocab_all, data_all_4], ignore_index = True)
data_vocab_all_BT.shape

data_vocab_all_BT_src = pd.concat([data_vocab_all, data_all_3], ignore_index = True)
print(data_train_all_BT.shape)


data_train_BT= data_train[i]
print(data_train_BT.shape)

data_train[3].shape
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

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFKC', s)
        if unicodedata.category(c) != 'Mn'
    )

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
        data = data_valid_all
   
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

MAX_LENGTH = 200
def prepareData(lang1, lang2, reverse=False, label="train"):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, label = label)
    print("Read %s sentence pairs" % len(pairs))
    # pairs = filterPairs(pairs)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    pairs = pairs[1:]
    for pair in pairs:
        input_lang.addSentence(pair[0])
        input_lang.addSentence(pair[1])
    output_lang = input_lang
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


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

def get_dataloader(batch_size, label ="train" ):
    input_lang, output_lang, _ = prepareData('ar', 'arz', label="vocab")
    _, _, pairs = prepareData('ar', 'arz', label = label)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        # inp_ids.append(EOS_token)
        # tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


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
                            self.src_tok_emb(src)), src_mask, src_key_padding_mask  = src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask, tgt_key_padding_mask = tgt_padding_mask)

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

def train_epoch(model, optimizer ):
    model.train()
    losses = 0
    valid_losses =0

    start_time = timer()
    for src, tgt in train_dataloader:
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

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        # scheduler.step()
        losses += loss.item()
    end_time = timer()
    model.eval()
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
          logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)



        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        valid_losses +=loss.item()
    valid_end_time = timer()




    # scheduler.step(losses / len(list(train_dataloader)))
    #scheduler.step()

    return   transformer, optimizer, start_time, end_time, valid_start_time, valid_end_time, losses / len(list(train_dataloader)), valid_losses /  len(list(valid_dataloader))

torch.cuda.empty_cache()

model ={}
best_loss = - np.inf



NUM_EPOCHS = 100
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 =NUM_EPOCHS , T_mult=1, verbose = True)
torch.cuda.empty_cache()

batch_size = 16
input_lang, output_lang, train_dataloader = get_dataloader(batch_size = batch_size, label ="train")
_, _, valid_dataloader = get_dataloader(batch_size = batch_size, label ="valid")



torch.cuda.empty_cache()

for z in [0]:
    torch.cuda.empty_cache()
    NUM_EPOCHS = 100
    best_valid_loss = 1000
    early_stop_threshold = 3
    best_epoch =-1
    j=0
    
    torch.manual_seed(z)
    SRC_VOCAB_SIZE = tokenizer.vocab_size
    TGT_VOCAB_SIZE = tokenizer.vocab_size
    
    EMB_SIZE = 512
    NHEAD = 2
    FFN_HID_DIM = 512

    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                              NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
    
    scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose = True)
    

    for epoch in range(1, NUM_EPOCHS+1):
    
        if j < early_stop_threshold:
            transformer, optimizer,  start_time, end_time, valid_start_time, valid_end_time, train_loss, valid_loss = train_epoch( transformer, optimizer )
    
            if (valid_loss < best_valid_loss ):
              best_model = transformer
              best_epoch = epoch
              best_valid_loss = valid_loss
              print("Saving ...")
              torch.save({
    
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
    
                    }, "./transformer_multi_dialect_BT_20k_{}.pt".format(z))
              j = 0
    
            else:
              j =j+1
    
            print((f"Epoch: {epoch}, Train loss: {train_loss:.9f}, "f"Epoch time = {(end_time - start_time)/60:.3f}m"),
                    (f"Validation loss: {valid_loss:.9f}, "f"Epoch time = {(valid_end_time - valid_start_time)/60:.3f}m") )
        else:
          break

def greedy_decode(model, src, src_mask,src_padding_mask, max_len, start_symbol, output_lang):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    decoded_words = []
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

def decode(model, src, src_mask,src_padding_mask, max_len, start_symbol, output_lang,
             p=None, greedy=None):
    """ Main decoding function, beam search is in a separate function """

    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    decoded_words = []
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
        logits = model.generator(out[:, -1])
        prob = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)

        if greedy:
          _, next_word = torch.max(prob, dim=1)
          next_word = next_word.item()


        if p is not None:

          sorted_probs, sorted_indices = torch.sort(prob, descending=True)
          cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
          sorted_indices_to_remove = cumulative_probs > p
          sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
          sorted_indices_to_remove[:, 0] = 0
          sorted_samp_probs = sorted_probs.clone()
          sorted_samp_probs[sorted_indices_to_remove] = 0



          sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)

          next_tokens = sorted_indices.gather(1, sorted_next_indices)
          next_word = next_tokens.item()

          next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()

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

      decoded_words, _ = decode(
          model,  src, src_mask,src_padding_mask,max_len=num_tokens + 5,
          start_symbol=SOS_token, output_lang=output_lang, greedy = True
          #p=0.95
          )
      output_words = tokenizer.convert_tokens_to_string(decoded_words)
      output_words = output_words.replace("[SEP]", "")
      output_words = output_words.replace("[UNK]", "")
      output_words = output_words.replace("[PAD]", "")
      # print("input: ", src_sentence)
      # print("target: ", trgt)
      # print("prediction: ", output_words)
      # print('')
      i=i+1
      targets.append([normalizeString(trgt)])
      outputs.append(output_words)
    return outputs, targets

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive



from nltk.translate.bleu_score import corpus_bleu
EMB_SIZE = 512
NHEAD = 2
FFN_HID_DIM = 512

NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
for i in [1]:
    bleu={
        "Gulf": [],
            "Iraqi": [],
            "Levantine":[],
            "Nile_Basin":[],
            "North_Africa":[]
          }

    z=0
   
    path = "./transformer_multi_dialect_BT_20k_{}.pt".format(z)
    model_path_list = [path]
    for j, (k, v) in enumerate(bleu.items()):
        
        data = data_test[j]
        outputs_all= []
        sent_list = {"source_sent":[], "Actual_sent":[],  "predicted_sent_base": [],"predicted_sent_BT": [],}
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
            print("loading")
            transformer.load_state_dict(checkpoint['model_state_dict'])
            outputs, target= translate(transformer, data[:100], input_lang, output_lang)
            outputs_all.append(outputs)
            
        sent_list["source_sent"] = data.source_lang[:100]
        sent_list["Actual_sent"] = data.target_lang[:100]
        sent_list["predicted_sent_base"] = outputs_all[0][:100]
        sent_list["predicted_sent_BT"] = outputs_all[1][:100]
        data_out = pd.DataFrame(sent_list)
        # data_out.to_csv(f"./{k}_Gulf_model.txt", sep=",", encoding="utf-8")
        src_test = corpus_bleu(target, outputs)
        bleu[k].append(src_test)
        print(bleu)
        src_test
