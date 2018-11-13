from dataset import TorchTextDataset
from train import train
from models import CNNModel
import torch
import pickle as pkl


labels = pkl.load(open('./data/train/labels.pkl', 'rb'))
sentences = pkl.load(open('./data/train/sentences.pkl', 'rb'))

data = TorchTextDataset(sentences, labels)
train_iter, valid_iter = data.get_iterators()

# model parameters
vocab_size = len(data.text_field.vocab)
embeddings_size = 128
kernel_sizes = [2, 4, 8, 16]
kernel_num = 100
dropout = 0.5

# training parameters
num_epoch = 10
lr = 1e-3
cuda = torch.cuda.is_available()
log_interval = 100
valid_interval = 100

model = CNNModel(vocab_size, embeddings_size, kernel_sizes, kernel_num, dropout)

train_losses, train_auc, valid_losses, valid_auc = train(model, train_iter, valid_iter, num_epoch, lr, cuda, log_interval, valid_interval)
