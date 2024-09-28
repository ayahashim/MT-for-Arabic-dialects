# -*- coding: utf-8 -*-

from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import numpy as np
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import (AutoModel, AutoTokenizer)

MAX_LENGTH = 200
model_name = "aubmindlab/bert-base-arabertv02-twitter"
# arabic_prep = ArabertPreprocessor(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
SOS_token = tokenizer.cls_token_id
EOS_token = tokenizer.sep_token_id
PAD_token = tokenizer.pad_token_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_Gulf = pd.read_csv("./data_Gulf_preprocessed_train", sep="\t")
data_Iraqi = pd.read_csv("./data_Iraqi_preprocessed_train", sep="\t")
data_Levantine = pd.read_csv("./data_Levantine_preprocessed_train", sep="\t")
data_Nile_Basin = pd.read_csv("./data_Nile_Basin_preprocessed_train", sep="\t")
data_North_Africa = pd.read_csv(
    "./data_North_Africa_preprocessed_train", sep="\t")

data_Gulf_valid = pd.read_csv("./data_Gulf_preprocessed_valid", sep="\t")
data_Iraqi_valid = pd.read_csv("./data_Iraqi_preprocessed_valid", sep="\t")
data_Levantine_valid = pd.read_csv(
    "./data_Levantine_preprocessed_valid", sep="\t")
data_Nile_Basin_valid = pd.read_csv(
    "./data_Nile_Basin_preprocessed_valid", sep="\t")
data_North_Africa_valid = pd.read_csv(
    "./data_North_Africa_preprocessed_valid", sep="\t")

data_Gulf_test = pd.read_csv("./data_Gulf_preprocessed_test", sep="\t")
data_Iraqi_test = pd.read_csv("./data_Iraqi_preprocessed_test", sep="\t")
data_Levantine_test = pd.read_csv(
    "./data_Levantine_preprocessed_test", sep="\t")
data_Nile_Basin_test = pd.read_csv(
    "./data_Nile_Basin_preprocessed_test", sep="\t")
data_North_Africa_test = pd.read_csv(
    "./data_North_Africa_preprocessed_test", sep="\t")

domain = ["Gulf", "Iraqi", "Levantine", "Nile_Basin", "North_Africa"]
data_train = []
for d in domain:
    data = pd.read_csv("./data_{}_preprocessed_train".format(d), sep="\t")
    data_train.append(data)

data_1 = pd.concat([data_train[0], data_train[1]], ignore_index=True)
data_2 = pd.concat([data_train[2], data_train[3]], ignore_index=True)

data_train_src = pd.concat([data_1, data_2], ignore_index=True)
data_train_all = pd.concat([data_train_src, data_train[4]], ignore_index=True)
# data_train_src = pd.concat([data_1, data_2], ignore_index=True)
# print(data_train[0].shape[0]+ data_train[1].shape[0] + data_train[2].shape[0]+data_train[3].shape[0])

domain = ["Gulf", "Iraqi", "Levantine", "Nile_Basin", "North_Africa"]
data_test = []
for d in domain:
    data = pd.read_csv("./data_{}_preprocessed_test".format(d), sep="\t")
    data_test.append(data)

data_1 = pd.concat([data_test[0], data_test[1]], ignore_index=True)
data_2 = pd.concat([data_test[2], data_test[3]], ignore_index=True)

data_test_all = pd.concat([data_1, data_2], ignore_index=True)
data_test_all = pd.concat([data_test_all, data_test[4]], ignore_index=True)

domain = ["Gulf", "Iraqi", "Levantine", "Nile_Basin", "North_Africa"]
data_valid = []
for d in domain:
    data = pd.read_csv("./data_{}_preprocessed_valid".format(d), sep="\t")
    data_valid.append(data)

data_1 = pd.concat([data_valid[0], data_valid[1]], ignore_index=True)
data_2 = pd.concat([data_valid[2], data_valid[3]], ignore_index=True)

data_valid_src = pd.concat([data_1, data_2], ignore_index=True)
data_valid_all = pd.concat([data_valid_src, data_valid[4]], ignore_index=True)

# print(data_train_src.shape, data_train_all.shape)

data_1 = pd.concat([data_valid_all, data_train_all], ignore_index=True)
data_all = pd.concat([data_test_all, data_1], ignore_index=True)



data_train_all.to_csv("./data_train_all", sep="\t")
data_test_all.to_csv("./data_test_all", sep="\t")
data_valid_all.to_csv("./data_valid_all", sep="\t")
data_all.to_csv("./data_all", sep="\t")

data_file = "./data_train_all"
vocab_file = "./data_all"
# data_trgt_file = "./Data/data_trgt_train_MAG"
data_valid_file = "./data_valid_all"


class Lang:
    def __init__(self, name):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.word2index = {
            self.tokenizer.pad_token: self.tokenizer.pad_token_id}
        self.word2count = {}
        self.index2word = {
            self.tokenizer.pad_token_id: self.tokenizer.pad_token}
        self.n_words = 1  # Count PAD token

    def addSentence(self, sentence):
        for word in self.tokenizer.tokenize(sentence, add_special_tokens=True):
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
    # s= arabic_prep.preprocess(s)
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z؀-ۿ?.!,¿]+", " ", s)
    s = re.sub(r"([.!?])", r" \1", s)
    return s


def readLangs(lang1, lang2, reverse=False, label="src"):
    print("Reading lines...")

    # Read the file and split into lines
    if label == "vocab":
        # lines = open(vocab_file, encoding='utf-8').\
        # read().strip().split('\n')
        data = data_all
    if label == "src":
        # lines = open(data_file, encoding='utf-8').\
        #     read().strip().split('\n')
        data = data_train_src
    # if label =="trgt":
    #     lines = open(data_trgt_file, encoding='utf-8').\
    #     read().strip().split('\n')
    if label == "valid":
        # lines = open(data_valid_file, encoding='utf-8').\
        # read().strip().split('\n')
        data = data_valid_src

    # Split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[normalizeString(data.source_lang[idx]),
              normalizeString(data.target_lang[idx])]
             for idx in data.index]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2, reverse=False, label="src"):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, label)
    print("Read %s sentence pairs" % len(pairs))
    pairs = pairs[1:]
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        input_lang.addSentence(pair[1])
    output_lang = input_lang
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, 0) for word in tokenizer.tokenize(sentence)]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(batch_size, label="src"):
    input_lang, output_lang, _ = prepareData('ar', 'arz', label="vocab")

    _, _, pairs = prepareData('ar', 'arz', label=label)
    # elif label == "trgt":
    #     _, _, pairs = prepareData('ar', 'arz', label="trgt")

    pairs = pairs[1:]
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
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


def beam_search(encoder, decoder, sentence, input_lang, output_lang, beam_width=5, max_length=MAX_LENGTH):
    with torch.no_grad():
        sentence = normalizeString(sentence)
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden)

        # Initialize the list to hold the completed beams
        completed_beams = []

        # Start the beam search process

        candidate_beams = []
        decoder_hidden = encoder_hidden

        # Get the top-k candidates and their probabilities
        topk_probs, topk_indices = decoder_outputs.squeeze().topk(beam_width)
        topk_probs = topk_probs.T  # (beam, max_length)
        topk_indices = topk_indices.T

        # Expand the current beam with the top-k candidates

        for i in range(beam_width):
            decoder_output = []
            candidate_prob = 0.0
            for j in range(MAX_LENGTH):

                new_candidate_idx = topk_indices[i][j].item()
                decoder_output.append(new_candidate_idx)

                candidate_prob += topk_probs[i][j].item()
                new_decoder_input = torch.tensor(
                    [decoder_output], device=device)
                new_beam_score = (candidate_prob)

                # Check if the candidate is the end token
                if new_candidate_idx == EOS_token:
                    completed_beams.append((decoder_output, new_beam_score))
                    break
                if j == (MAX_LENGTH-1):
                    completed_beams.append((decoder_output, new_beam_score))

        # Sort the completed beams and select the one with the highest score
        completed_beams.sort(key=lambda x: x[1], reverse=True)

        decoded_words = [output_lang.index2word.get(
            word, tokenizer.unk_token)for word in completed_beams[0][0]]

        return decoded_words  # No attention scores for beam search


def evaluateRandomly(encoder, decoder, input_lang, output_lang, data=None, beam=1, n=None):
    outputs = []
    targets = []
    i = 0
    for idx in data.index:
        if n != None:
            if i == n:
                break

        targets.append([normalizeString(data.target_lang[idx])])
        print('input', data.source_lang[idx])
        print('output', data.target_lang[idx])
        output_words = beam_search(
            encoder, decoder, data.source_lang[idx], input_lang, output_lang, beam_width=beam)

        output_sentence = tokenizer.convert_tokens_to_string(output_words)
        output_sentence = output_sentence.replace("[SEP]", "")
        output_sentence = output_sentence.replace("[PAD]", "")
        output_sentence = output_sentence.replace("[CLS]", "")
        output_sentence = output_sentence.replace("[UNK]", "")
        outputs.append(output_sentence)

        print('prediction', output_sentence)
        print('')
        i = i+1
    return targets, outputs


def load_checkpoit(checkpoint_file, model, optimizer):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def save_checkpoit(checkpoint_file, model, optimizer):
    print("Saving checkpoint...")
    torch.save({

        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),

    }, checkpoint_file)


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

# encoder class


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))

        output, hidden = self.gru(embedded)
        return output, hidden

# attention classes


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        # scores shape (N,L,1)
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        # weights shape is N,1,L
        # context shape is N,1,hid_size
        return context, weights


class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, query, keys):
        scores = torch.bmm(query, keys.permute(0, 2, 1))

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        # weights shape is N,1,L
        # context shape is N,1,hid_size
        return context, weights


teacher_force_ratio = 0.5


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, attention="Bahdanau"):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        if attention == "Bahdanau":
            self.attention = BahdanauAttention(hidden_size)
        else:
            self.attention = DotAttention()

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, MAX_LENGTH=MAX_LENGTH):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            use_teacher_forcing = True if random.random() < teacher_force_ratio else False
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                if use_teacher_forcing:
                    decoder_input = target_tensor[:, i].unsqueeze(
                        1)  # Teacher forcing
                else:
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


def train(train_dataloader, valid_dataloader, encoder, decoder, n_epochs,
          print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    criterion = nn.NLLLoss()
    best_valid_loss = 10
    early_stop_threshold = 3
    best_epoch = -1
    j = 0
    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for data in train_dataloader:
            input_tensor, target_tensor = data
            inp_len = [len(input_tensor[i][input_tensor[i] != 0])
                       for i in range(input_tensor.shape[0])]
            trgt_len = [len(target_tensor[i][target_tensor[i] != 0])
                        for i in range(target_tensor.shape[0])]
            input_tensor = input_tensor[:, :max(inp_len)]
            target_tensor = target_tensor[:, :max(trgt_len)]

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(
                encoder_outputs, encoder_hidden, target_tensor, MAX_LENGTH=max(trgt_len))

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.reshape(-1)
            )

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        scheduler_enc.step(total_loss / len(train_dataloader))
        scheduler_dec.step(total_loss / len(train_dataloader))
        print_loss_total = total_loss / len(train_dataloader)
        plot_loss_total = total_loss / len(train_dataloader)

        valid_total_loss = 0

        for data in valid_dataloader:
            input_tensor, target_tensor = data
            inp_len = [len(input_tensor[i][input_tensor[i] != 0])
                       for i in range(input_tensor.shape[0])]
            trgt_len = [len(target_tensor[i][target_tensor[i] != 0])
                        for i in range(target_tensor.shape[0])]
            input_tensor = input_tensor[:, :max(inp_len)]
            target_tensor = target_tensor[:, :max(trgt_len)]
            with torch.no_grad():
                encoder_outputs, encoder_hidden = encoder(input_tensor)
                decoder_outputs, _, _ = decoder(
                    encoder_outputs, encoder_hidden, target_tensor, MAX_LENGTH=max(trgt_len))

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.reshape(-1)
            )

            valid_total_loss += loss.item()
        valid_total_loss = valid_total_loss / len(valid_dataloader)
        if j < early_stop_threshold:
            if best_valid_loss > valid_total_loss:
                best_valid_loss = valid_total_loss
                best_epoch = epoch
                j = 0
                print("Saving ...")
                save_checkpoit(f"./checkpoint_enc_base_{z}",
                               encoder, encoder_optimizer)
                save_checkpoit(f"./checkpoint_dec_base_{z}",
                               decoder, decoder_optimizer)
            else:
                j += 1

        else:

            break

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every

            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
            print("valid loss: ", valid_total_loss)
            print_loss_total = 0
            valid_total_loss = 0

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


for z in [0,1]:
    hidden_size = 512
    batch_size = 16
    learning_rate = 0.0001
   
    torch.manual_seed(z)
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
    _, _, valid_dataloader = get_dataloader(batch_size, label="valid")
    
    encoder = EncoderRNN(tokenizer.vocab_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, tokenizer.vocab_size).to(device)
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    scheduler_enc = ReduceLROnPlateau(
        encoder_optimizer, patience=2, factor=0.1, verbose=True)
    scheduler_dec = ReduceLROnPlateau(
        decoder_optimizer, patience=2, factor=0.1, verbose=True)
    torch.cuda.empty_cache()
    
    
    train(train_dataloader, valid_dataloader, encoder,decoder, 60, print_every=1, plot_every=5)

z=1
Load= True

torch.manual_seed(z)
if Load:
      load_checkpoit(f"./checkpoint_enc_base_{z}", encoder, encoder_optimizer)
      load_checkpoit(f"./checkpoint_dec_base_{z}", decoder, decoder_optimizer)
      
bleu={
    "Gulf": [],
        "Iraqi": [],
        "Levantine":[],
        "Nile_Basin":[],
        "North_Africa":[]
       }

from nltk.translate.bleu_score import corpus_bleu
for j, (k,v) in enumerate(bleu.items()):
    targets, outputs= evaluateRandomly(encoder, decoder,input_lang,output_lang, data_test[j])
    src_test = corpus_bleu(targets, outputs)
    bleu[k].append(src_test)
    print(bleu)


#z=0, {'Gulf': [0.524567439009065], 'Iraqi': [0.6397466542969094], 'Levantine': [0.48049319602442003],
       # 'Nile_Basin': [0.5263656147151258], 'North_Africa': [0.2841222724836363]}
#z=1, {'Gulf': [0.5249207911722588], 'Iraqi': [0.6413246651757423], 'Levantine': [0.4802711729601389], 
        # 'Nile_Basin': [0.5275433764972391], 'North_Africa': [0.2834026916040478]}


# 0.2824961013992457, North_Africa, z=1
# 

# 284810669463694


# 0.2854881701564368 North Africa

# targets_trgt, outputs_trgt= evaluateRandomly(encoder, decoder,input_lang,output_lang, data_trgt_test)
# trgt_test = corpus_bleu(targets_trgt, outputs_trgt )

# print("Bley score for target domain is {} and for source domain is {}".format(
#     trgt_test, src_test))

# 20m 51s (- 396m 20s) (3 5%) 2.5290
# valid loss:  2.53274218434269
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 27m 46s (- 388m 57s) (4 6%) 2.3451
# valid loss:  2.380791461294137
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 34m 40s (- 381m 21s) (5 8%) 2.1563
# valid loss:  2.2349102672153305
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 41m 35s (- 374m 15s) (6 10%) 1.9761
# valid loss:  2.0933535480962218
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 48m 31s (- 367m 25s) (7 11%) 1.8074
# valid loss:  1.9779538565035004
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 55m 29s (- 360m 39s) (8 13%) 1.6417
# valid loss:  1.8644170533250837
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 62m 27s (- 353m 58s) (9 15%) 1.5181
# valid loss:  1.7763019984642279
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 69m 27s (- 347m 16s) (10 16%) 1.4052
# valid loss:  1.7192494448961564
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 76m 25s (- 340m 26s) (11 18%) 1.2940
# valid loss:  1.6425226337267358
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 83m 22s (- 333m 31s) (12 20%) 1.2105
# valid loss:  1.6011189689624656
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 90m 18s (- 326m 29s) (13 21%) 1.1238
# valid loss:  1.5529060208971062
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 97m 16s (- 319m 36s) (14 23%) 1.0461
# valid loss:  1.5181382208773233
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 104m 14s (- 312m 44s) (15 25%) 0.9748
# valid loss:  1.4767674151265506
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 111m 18s (- 306m 6s) (16 26%) 0.9130
# valid loss:  1.4496185469974592
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 118m 15s (- 299m 7s) (17 28%) 0.8473
# valid loss:  1.44424220937693
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 125m 19s (- 292m 26s) (18 30%) 0.7839
# valid loss:  1.4260231621780441
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 132m 15s (- 285m 22s) (19 31%) 0.7357
# valid loss:  1.405092654225317
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 139m 13s (- 278m 26s) (20 33%) 0.6760
# valid loss:  1.3993631177627057
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 146m 9s (- 271m 25s) (21 35%) 0.6304
# valid loss:  1.360076682696354
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 153m 3s (- 264m 23s) (22 36%) 0.5821
# valid loss:  1.3508746263399287
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 160m 2s (- 257m 26s) (23 38%) 0.5364
# valid loss:  1.3439355070874528
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 167m 0s (- 250m 30s) (24 40%) 0.4979
# valid loss:  1.342866928000184
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 174m 7s (- 243m 46s) (25 41%) 0.4622
# valid loss:  1.337723237941566
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 181m 5s (- 236m 48s) (26 43%) 0.4214
# valid loss:  1.331753376979851
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 188m 2s (- 229m 49s) (27 45%) 0.3846
# valid loss:  1.3081053505534108
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 194m 58s (- 222m 49s) (28 46%) 0.3548
# valid loss:  1.3024696555820483
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 201m 54s (- 215m 49s) (29 48%) 0.3247
# valid loss:  1.3016445344693741
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 208m 52s (- 208m 52s) (30 50%) 0.2938
# valid loss:  1.282628129534785
# 215m 47s (- 201m 52s) (31 51%) 0.2677
# valid loss:  1.3049773452351394
# 222m 45s (- 194m 54s) (32 53%) 0.2460
# valid loss:  1.289009770505058
# Saving ...
# Saving checkpoint...
# Saving checkpoint...
# 229m 52s (- 188m 4s) (33 55%) 0.2200
# valid loss:  1.2718950749817983
# 236m 54s (- 181m 9s) (34 56%) 0.1990
# valid loss:  1.2765962724546784
# 243m 55s (- 174m 13s) (35 58%) 0.1785
# valid loss:  1.2735043300261486