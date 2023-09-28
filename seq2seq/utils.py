from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import numpy as np
import time 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datasets import load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math
MAX_LENGTH = 128
SOS_token = 0
EOS_token = 1

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence,EOS_token):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(input_lang,output_lang,pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def dataLoader(lang_in_class,lang_out_class,lang_in_corpus,lang_out_corpus,batch_size,max_length = 50,reverse=False):
      pairs = []
      in_index = []
      out_index = []

      input_ids = np.zeros((len(lang_in_corpus), MAX_LENGTH), dtype=np.int32)
      target_ids = np.zeros((len(lang_out_corpus), MAX_LENGTH), dtype=np.int32)

      assert len(lang_in_corpus) == len(lang_out_corpus)
      for idx, (inn, out) in enumerate(zip(lang_in_corpus, lang_out_corpus)):
        pairs.append([inn,out])
        inp_ids = indexesFromSentence(lang_in_class,normalizeString(inn))
        out_ids = indexesFromSentence(lang_out_class,normalizeString(out))
        inp_ids.append(EOS_token)
        out_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(out_ids)] = out_ids

      train_data = TensorDataset(torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device))
      loader = DataLoader(train_data, batch_size=batch_size)
      return loader,pairs
def evaluate(model, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        decoder_outputs, decoder_hidden, decoder_attn = model(input_tensor)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(input_lang,output_lang,model,pairs, n=3):
    for i in range(n):
        pair = pairs[i]
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(model, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def train_epoch(dataloader, model,optimizer, criterion):

    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data

        optimizer.zero_grad()


        decoder_outputs, _, _ = model(input_tensor, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()


        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


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

def train(train_dataloader,model, n_epochs,input_lang,output_lang, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
 #   decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()





    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader,model, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        #f epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                    epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        evaluateRandomly(input_lang,output_lang,model, n=3)
    showPlot(plot_losses)