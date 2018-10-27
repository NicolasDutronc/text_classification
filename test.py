import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl

from torch.utils.data import DataLoader

from dataset import TextDataset
from models import SimpleModel


labels = pkl.load(open('./data/train/labels.pkl', 'rb'))
sentences = pkl.load(open('./data/train/sentences.pkl', 'rb'))
sequences = pkl.load(open('./data/train/sequences.pkl', 'rb'))
vocab = pkl.load(open('./data/dict.pkl', 'rb'))


data = TextDataset(labels, sequences)

print(data[1])

model = SimpleModel(len(vocab), 5)
model.cuda()

# data loader
dataloader = DataLoader(data, shuffle=True)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# loss
loss = nn.CrossEntropyLoss(torch.DoubleTensor([0.86]))

num_epoch = 10
losses = []

for i in range(num_epoch):
    print("Epoch:", num_epoch)
    for sequence, label in dataloader:

        # print(sequence)
        # print(label)

        sequence = sequence.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        prediction = model(sequence)
        output = F.binary_cross_entropy(prediction, label)
        print('loss:', output)
        losses.append(output)
        output.backward()
        optimizer.step()
