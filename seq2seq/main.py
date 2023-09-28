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
import time
import math
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from datasets import load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
from model import *
MAX_LENGTH = 128
SOS_token = 0
EOS_token = 1

def main():
    data = load_dataset('bentrevett/multi30k')
    train_en =  [normalizeString(text) for text in data['train']['en']  ]
    train_de =   [normalizeString(text) for text in data['train']['de'] ]



    de_en =   [normalizeString(text) for text in data['validation']['en'] ]
    dev_de =   [normalizeString(text) for text in  data['validation']['de'] ]

    test_en =    [normalizeString(text) for text in data['test']['en'] ]
    test_de =   [normalizeString(text) for text in data['test']['de']  ]
    english = Lang('en')
    germany = Lang('de')
    for i in tqdm(range(len(train_en))):
        english.addSentence(train_en[i])

    for i in tqdm(range(len(train_de))):
        germany.addSentence(train_de[i])
    train_dataloader,pairs = dataLoader(germany,english,train_de,train_en,32)
    hidden_size = MAX_LENGTH

    encoder = Encoder(germany.n_words, hidden_size).to(device)
    decoder = Decoder(hidden_size, english.n_words).to(device)

    model = Seq2Seq(encoder,decoder).to(device)
    train(train_dataloader,model, 80,germany, english,learning_rate=0.001, print_every=2, plot_every=5)

if __name__ == "__main__":
    main()



