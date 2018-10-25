import pandas as pd
import torch
import pickle as pkl

from dataset import TextDataset
from models import SimpleModel


labels = pd.read_pickle('./data/train/labels.pkl')
sentences = pd.read_pickle('./data/train/sentences.pkl')
sequences = pd.read_pickle('./data/train/sequences.pkl')
vocab = pkl.load(open('./data/dict.pkl', 'rb'))

data = TextDataset(labels, sequences)
print(data.df.info())

print(data[1])

model = SimpleModel(len(vocab), 5)
print(model(data[1][0]))

# TODO: data loaders
